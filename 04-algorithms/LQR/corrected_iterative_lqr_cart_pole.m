function iterative_lqr_cart_pole()
% ITERATIVE_LQR_CART_POLE
% Corrected iLQR / finite-horizon LQR-style implementation for cart-pole.
%
% State: x = [cart_position; pole_angle; cart_velocity; pole_angular_velocity]
% Input: u = cart force
%
% The controller locally linearizes nonlinear dynamics:
%   x_{k+1} = f_d(x_k,u_k)
% and locally quadratizes the cost, then performs the Riccati-style backward
% pass:
%   Qx, Qu, Qxx, Qux, Quu
%   k = -Quu^{-1} Qu
%   K = -Quu^{-1} Qux
%
% Forward pass:
%   u_new = u_old + alpha*k + K*(x_new - x_old)

    close all; clc;

    T = 5.0;           % time horizon
    dt = 0.05;         % time step
    N = floor(T/dt)+1; % number of state samples

    nX = 4;            % number of states
    nU = 1;            % number of inputs

    % Cart-pole physical parameters
    param.mc = 10;
    param.mp = 2;
    param.l  = 0.5;
    param.g  = 9.8;
    param.b  = 0.1;
    param.d  = 0.1;

    % Initial and desired states
    x0 = [0; 0; 0; 0];
    xd = [0; pi; 0; 0];

    % Nominal trajectories
    xtraj = zeros(nX, N);
    utraj = zeros(nU, N-1);

    % Feedforward and feedback gains
    ktraj = zeros(nU, N-1);
    Ktraj = zeros(nU, nX, N-1);

    % Cost matrices
    Q  = diag([0.1, 1, 0.1, 1]);
    Qf = diag([1.2, 30000, 1.5, 30000]);
    R  = 1e-3;  % Avoid using an extremely tiny R; it can make Quu ill-conditioned.

    % Run iLQR
    maxIter = 300;
    tol = 1e-5;
    [xtraj, utraj, ktraj, Ktraj, Jhist] = ilqr( ...
        x0, xtraj, utraj, ktraj, Ktraj, N, dt, param, Q, R, Qf, xd, maxIter, tol);

    fprintf('Final cost: %.6f\n', Jhist(end));

    % Simulate using optimized open-loop controls
    t = 0:dt:T;
    x = zeros(nX, N);
    x(:,1) = x0;

    for k = 1:N-1
        xdot = cartpole_dynamics(t(k), x(:,k), utraj(:,k), param);
        x(:,k+1) = x(:,k) + dt * xdot;
    end

    % Plot results
    figure;
    subplot(3,1,1);
    plot(t, x(1,:), 'LineWidth', 1.5); grid on;
    ylabel('cart position');

    subplot(3,1,2);
    plot(t, x(2,:), 'LineWidth', 1.5); grid on;
    ylabel('pole angle');
    yline(pi, '--');

    subplot(3,1,3);
    stairs(t(1:end-1), utraj(1,:), 'LineWidth', 1.5); grid on;
    ylabel('u');
    xlabel('time');

    figure;
    semilogy(Jhist, 'LineWidth', 1.5); grid on;
    xlabel('iLQR iteration');
    ylabel('cost');
    title('Cost convergence');

    % Animate
    for k = 1:N
        draw_cartpole(t(k), x(:,k), param);
        drawnow;
    end
end

%==========================================================================
%   iLQR function
%==========================================================================
function [xtraj, utraj, ktraj, Ktraj, Jhist] = ilqr( ...
    x0, xtraj, utraj, ktraj, Ktraj, N, dt, param, Q, R, Qf, xd, maxIter, tol)

    % Initial rollout with initial control trajectory
    [xtraj, J] = rollout(x0, utraj, N, dt, param, Q, R, Qf, xd);

    Jhist = J;

    for iter = 1:maxIter
        % Riccati-style backward pass
        [Ktraj, ktraj] = backward_pass(xtraj, utraj, Q, R, Qf, xd, param, N, dt);

        % Line-search forward pass
        [xnew, unew, Jnew, accepted] = forward_pass( ...
            x0, xtraj, utraj, ktraj, Ktraj, N, dt, param, Q, R, Qf, xd, J);

        if ~accepted
            fprintf('Iteration %d: line search failed. Stop.\n', iter);
            break;
        end

        improvement = J - Jnew;

        xtraj = xnew;
        utraj = unew;
        J = Jnew;
        Jhist(end+1) = J; %#ok<AGROW>

        fprintf('Iteration %3d | cost = %.6f | improvement = %.6f\n', iter, J, improvement);

        if abs(improvement) < tol
            break;
        end
    end
end

%==========================================================================
%   Rollout with a fixed open-loop input sequence
%==========================================================================
function [xtraj, J] = rollout(x0, utraj, N, dt, param, Q, R, Qf, xd)
    nX = length(x0);
    xtraj = zeros(nX, N);
    xtraj(:,1) = x0;

    J = 0;
    t = 0;

    for k = 1:N-1
        x = xtraj(:,k);
        u = utraj(:,k);

        J = J + dt * stage_cost(x, u, xd, Q, R);

        xdot = cartpole_dynamics(t, x, u, param);
        xtraj(:,k+1) = x + dt * xdot;

        t = t + dt;
    end

    J = J + final_cost(xtraj(:,N), xd, Qf);
end

%==========================================================================
%   Forward pass with line search
%==========================================================================
function [xtraj, utraj, J, accepted] = forward_pass( ...
    x0, xtraj0, utraj0, ktraj, Ktraj, N, dt, param, Q, R, Qf, xd, J0)

    alphaList = [1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001];

    nX = length(x0);
    nU = size(utraj0,1);

    accepted = false;
    bestJ = inf;
    bestX = xtraj0;
    bestU = utraj0;

    for alpha = alphaList
        xtraj_try = zeros(nX, N);
        utraj_try = zeros(nU, N-1);

        x = x0;
        t = 0;
        Jtry = 0;
        xtraj_try(:,1) = x;

        for k = 1:N-1
            dx = x - xtraj0(:,k);

            % Correct iLQR forward update:
            % u_new = u_old + alpha*k + K*(x_new - x_old)
            u = utraj0(:,k) + alpha * ktraj(:,k) + Ktraj(:,:,k) * dx;

            utraj_try(:,k) = u;
            Jtry = Jtry + dt * stage_cost(x, u, xd, Q, R);

            xdot = cartpole_dynamics(t, x, u, param);
            x = x + dt * xdot;

            xtraj_try(:,k+1) = x;
            t = t + dt;
        end

        Jtry = Jtry + final_cost(xtraj_try(:,N), xd, Qf);

        if Jtry < bestJ
            bestJ = Jtry;
            bestX = xtraj_try;
            bestU = utraj_try;
        end

        if Jtry < J0
            accepted = true;
            xtraj = xtraj_try;
            utraj = utraj_try;
            J = Jtry;
            return;
        end
    end

    % Return the best trial even when not accepted
    xtraj = bestX;
    utraj = bestU;
    J = bestJ;
end

%==========================================================================
%   Backward pass
%==========================================================================
function [Ktraj, ktraj] = backward_pass(xtraj, utraj, Q, R, Qf, xd, param, N, dt)

    nX = size(xtraj,1);
    nU = size(utraj,1);

    Ktraj = zeros(nU, nX, N-1);
    ktraj = zeros(nU, N-1);

    % Terminal value function:
    % V_N(x) = 0.5*(x - xd)'*Qf*(x - xd)
    xN = xtraj(:,N);
    Vx  = Qf * (xN - xd);
    Vxx = Qf;

    reg = 1e-6;

    for k = N-1:-1:1
        x = xtraj(:,k);
        u = utraj(:,k);

        % Stage cost gradients
        [gx, gu, gxx, gux, guu] = cost_gradients(x, u, xd, Q, R);

        % Continuous-time dynamics gradients
        [Ac, Bc] = cartpole_grads(0, x, u, param);

        % Discrete-time linearization for Euler integration:
        % x_{k+1} = x_k + dt*f(x_k,u_k)
        fx = eye(nX) + dt * Ac;
        fu = dt * Bc;

        % Because stage cost is multiplied by dt during rollout,
        % its gradients should also be multiplied by dt here.
        gx  = dt * gx;
        gu  = dt * gu;
        gxx = dt * gxx;
        gux = dt * gux;
        guu = dt * guu;

        % Q-function derivatives
        Qx  = gx  + fx' * Vx;
        Qu  = gu  + fu' * Vx;
        Qxx = gxx + fx' * Vxx * fx;
        Qux = gux + fu' * Vxx * fx;
        Quu = guu + fu' * Vxx * fu;

        % Regularize Quu to keep it positive definite numerically
        Quu = 0.5 * (Quu + Quu') + reg * eye(nU);

        % Correct signs:
        % delta_u* = k + K*delta_x
        % k = -Quu^{-1}Qu
        % K = -Quu^{-1}Qux
        kff = -Quu \ Qu;
        Kfb = -Quu \ Qux;

        ktraj(:,k) = kff;
        Ktraj(:,:,k) = Kfb;

        % Value function update
        Vx = Qx + Kfb' * Quu * kff + Kfb' * Qu + Qux' * kff;
        Vxx = Qxx + Kfb' * Quu * Kfb + Kfb' * Qux + Qux' * Kfb;

        % Symmetrize to reduce numerical drift
        Vxx = 0.5 * (Vxx + Vxx');
    end
end

%==========================================================================
%   Stage cost
%==========================================================================
function J = stage_cost(x, u, xd, Q, R)
    e = x - xd;
    J = 0.5 * e' * Q * e + 0.5 * u' * R * u;
end

%==========================================================================
%   Final cost
%==========================================================================
function Jf = final_cost(x, xd, Qf)
    e = x - xd;
    Jf = 0.5 * e' * Qf * e;
end

%==========================================================================
%   Cost gradients
%==========================================================================
function [gx, gu, gxx, gux, guu] = cost_gradients(x, u, xd, Q, R)
    gx = Q * (x - xd);
    gu = R * u;
    gxx = Q;
    gux = zeros(length(u), length(x));
    guu = R;
end

%==========================================================================
%   Cart-pole dynamics
%==========================================================================
function xdot = cartpole_dynamics(~, x, u, param)

    mc = param.mc;
    mp = param.mp;
    l  = param.l;
    g  = param.g;
    b  = param.b;
    d  = param.d;

    x2 = x(2);
    x3 = x(3);
    x4 = x(4);

    s = sin(x2);
    c = cos(x2);

    den = mc + mp * s^2;

    xdot = zeros(4,1);
    xdot(1) = x3;
    xdot(2) = x4;
    xdot(3) = (u - b*x3 + d*x4*c/l + mp*s*(l*x4^2 + g*c)) / den;
    xdot(4) = (-u*c + b*x3*c - d*(mc+mp)*x4/(mp*l) ...
              - mp*l*x4^2*c*s - (mc+mp)*g*s/(mp*l)) / (l*den);
end

%==========================================================================
%   Cart-pole gradients: continuous-time Jacobians df/dx and df/du
%==========================================================================
function [dfdx, dfdu] = cartpole_grads(~, x, u, param)

    mc = param.mc;
    mp = param.mp;
    l  = param.l;
    g  = param.g;
    b  = param.b;
    d  = param.d;

    x2 = x(2);
    x3 = x(3);
    x4 = x(4);

    dfdx = zeros(4,4);
    dfdu = zeros(4,1);

    dfdx(1,3) = 1;
    dfdx(2,4) = 1;

    % Existing analytical derivatives, cleaned up with initialized matrices.
    df3dx1 = 0;

    df3dx2 = - (g*mp*sin(x2)^2 - mp*cos(x2)*(l*x4^2 + g*cos(x2)) ...
             + (d*x4*sin(x2))/l) / (mp*sin(x2)^2 + mc) ...
             - (2*mp*cos(x2)*sin(x2)*(u - b*x3 ...
             + mp*sin(x2)*(l*x4^2 + g*cos(x2)) ...
             + (d*x4*cos(x2))/l)) / (mp*sin(x2)^2 + mc)^2;

    df3dx3 = -b / (mp*sin(x2)^2 + mc);

    df3dx4 = ((d*cos(x2))/l + 2*l*mp*x4*sin(x2)) ...
             / (mp*sin(x2)^2 + mc);

    df4dx1 = 0;

    df4dx2 = (2*mp*cos(x2)*sin(x2)*(l*mp*cos(x2)*sin(x2)*x4^2 ...
             + (d*(mc + mp)*x4)/(l*mp) + u*cos(x2) ...
             - b*x3*cos(x2) + (g*sin(x2)*(mc + mp))/(l*mp))) ...
             / (l*(mp*sin(x2)^2 + mc)^2) ...
             - (b*x3*sin(x2) - u*sin(x2) ...
             + l*mp*x4^2*cos(x2)^2 - l*mp*x4^2*sin(x2)^2 ...
             + (g*cos(x2)*(mc + mp))/(l*mp)) ...
             / (l*(mp*sin(x2)^2 + mc));

    df4dx3 = (b*cos(x2)) / (l*(mc - mp*(cos(x2)^2 - 1)));

    df4dx4 = -((d*(mc + mp))/(l*mp) + l*mp*x4*sin(2*x2)) ...
             / (l*(mp*sin(x2)^2 + mc));

    df3du = 1 / (mp*sin(x2)^2 + mc);
    df4du = -cos(x2) / (l*(mc - mp*(cos(x2)^2 - 1)));

    dfdx(3,1) = df3dx1;
    dfdx(3,2) = df3dx2;
    dfdx(3,3) = df3dx3;
    dfdx(3,4) = df3dx4;

    dfdx(4,1) = df4dx1;
    dfdx(4,2) = df4dx2;
    dfdx(4,3) = df4dx3;
    dfdx(4,4) = df4dx4;

    dfdu(3) = df3du;
    dfdu(4) = df4du;
end

%==========================================================================
%   Draw cart-pole
%==========================================================================
function draw_cartpole(t, x, param)
    l = param.l;
    persistent hFig base raarm lwheel;

    % Create or recover the figure safely. In newer MATLAB versions, a
    % stale/deleted graphics handle can cause figure(hFig) to error.
    if isempty(hFig) || ~isgraphics(hFig, 'figure')
        hFig = figure(25);
        set(hFig, 'DoubleBuffer', 'on');
    else
        figure(hFig);
    end

    if isempty(base) || isempty(raarm) || isempty(lwheel)
        a1 = l + 0.25;
        av = pi*(0:0.05:1);
        theta = pi*(0:0.05:2);
        wb = 0.3;
        hb = 0.15;
        aw = 0.01;
        wheelr = 0.05;

        lwheel = [-wb/2 + wheelr*cos(theta); ...
                  -hb-wheelr + wheelr*sin(theta)]';

        base = [wb*[1 -1 -1 1]; hb*[1 1 -1 -1]]';

        arm = [aw*cos(av-pi/2), -a1+aw*cos(av+pi/2); ...
               aw*sin(av-pi/2),  aw*sin(av+pi/2)]';

        raarm = [(arm(:,1).^2 + arm(:,2).^2).^0.5, atan2(arm(:,2), arm(:,1))];
    end

    clf(hFig);
    axes('Parent', hFig);
    hold on;
    view(0,90);

    patch(x(1)+base(:,1), base(:,2), 0*base(:,1), 'b', 'FaceColor', [.3 .6 .4]);
    patch(x(1)+lwheel(:,1), lwheel(:,2), 0*lwheel(:,1), 'k');
    patch(x(1)+0.3+lwheel(:,1), lwheel(:,2), 0*lwheel(:,1), 'k');

    patch(x(1)+raarm(:,1).*sin(raarm(:,2)+x(2)-pi), ...
         -raarm(:,1).*cos(raarm(:,2)+x(2)-pi), ...
          1+0*raarm(:,1), 'r', 'FaceColor', [.9 .1 0]);

    plot3(x(1)+l*sin(x(2)), -l*cos(x(2)), 1, 'ko', ...
        'MarkerSize', 10, 'MarkerFaceColor', 'b');

    plot3(x(1), 0, 1.5, 'k.');

    title(['t = ', num2str(t, '%.2f'), ' sec']);
    set(gca, 'XTick', [], 'YTick', []);
    axis image;
    axis([-2.5 2.5 -2.5*l 2.5*l]);
end
