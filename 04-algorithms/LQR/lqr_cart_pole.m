function manual_lqr_cartpole_with_animation()

    clc; clear; close all;

    % ============================================================
    % 1. Time setup
    % ============================================================
    dt = 0.05;
    T = 5.0;
    N = floor(T/dt) + 1;

    % ============================================================
    % 2. Linearized discrete-time cart-pole system
    % 
    % State:
    % x = [cart position error;
    %      pole angle error from upright;
    %      cart velocity;
    %      pole angular velocity]
    %
    % Control:
    % u = cart force
    % ============================================================
    % using a linear approximation of cart-pole behavior near the upright equilibrium
    A = [
        1, 0, dt, 0;
        0, 1, 0, dt;
        0, 0.5*dt, 1, 0;
        0, 10*dt, 0, 1
    ];

    B = [
        0;
        0;
        dt;
        2*dt
    ];

    % ============================================================
    % 3. Cost matrices
    % ============================================================

    Q = diag([1, 100, 1, 10]);
    R = 0.1;
    Qf = Q;

    % ============================================================
    % 4. Manually solve finite-horizon LQR
    %
    % Riccati recursion:
    %
    % P_t = Q + A'P_{t+1}A
    %       - A'P_{t+1}B(R+B'P_{t+1}B)^(-1)B'P_{t+1}A
    %
    % K_t = (R+B'P_{t+1}B)^(-1)B'P_{t+1}A
    %
    % u_t = -K_t x_t
    % ============================================================

    P = zeros(4, 4, N);
    K = zeros(1, 4, N-1);

    % Terminal condition
    P(:, :, N) = Qf;

    % Backward Riccati recursion
    for k = N-1:-1:1

        Pnext = P(:, :, k+1);

        S = R + B' * Pnext * B;

        K(:, :, k) = S \ (B' * Pnext * A);

        P(:, :, k) = Q ...
            + A' * Pnext * A ...
            - A' * Pnext * B * K(:, :, k);

    end

    disp("Initial LQR gain K(:,:,1):");
    disp(K(:, :, 1));

    disp("Initial cost-to-go matrix P(:,:,1):");
    disp(P(:, :, 1));

    % ============================================================
    % 5. Simulate closed-loop system
    % ============================================================

    x = zeros(4, N);
    u = zeros(1, N-1);
    t = zeros(1, N);

    % Initial condition
    x(:,1) = [
        0.3;    % cart position error
        1.21;    % pole angle error from upright, radians
        0.0;    % cart velocity
        0.0     % pole angular velocity
    ];

    for k = 1:N-1

        % Manual LQR control law
        u(:,k) = -K(:, :, k) * x(:,k);

        % Linear system update
        x(:,k+1) = A * x(:,k) + B * u(:,k);

        % Time update
        t(k+1) = t(k) + dt;

    end

    % ============================================================
    % 6. Plot state trajectories
    % ============================================================

    figure;
    plot(t, x(1,:), 'LineWidth', 1.5); hold on;
    plot(t, x(2,:), 'LineWidth', 1.5);
    plot(t, x(3,:), 'LineWidth', 1.5);
    plot(t, x(4,:), 'LineWidth', 1.5);
    grid on;

    xlabel('Time [s]');
    ylabel('State value');
    title('Manual LQR Closed-Loop State Trajectory');
    legend('cart position', ...
           'pole angle error', ...
           'cart velocity', ...
           'pole angular velocity');

    % ============================================================
    % 7. Plot control input
    % ============================================================

    figure;
    stairs(t(1:end-1), u, 'LineWidth', 1.5);
    grid on;

    xlabel('Time [s]');
    ylabel('Control input u');
    title('Manual LQR Control Force');

    % ============================================================
    % 8. Animate cart-pole
    % ============================================================

    figure;
    for k = 1:N
        draw_cartpole(t(k), x(:,k));
        pause(0.03);
    end

end


% ============================================================
% Draw cart-pole animation
% ============================================================

function draw_cartpole(t, x)

    clf;
    hold on;
    axis equal;
    grid on;

    % Drawing parameters
    cart_width = 0.4;
    cart_height = 0.2;
    wheel_radius = 0.05;
    pole_length = 0.8;

    % State variables
    cart_pos = x(1);

    % x(2) is angle error from upright.
    % Upright corresponds to theta = pi.
    theta = pi + x(2);

    % Ground
    plot([-3, 3], [-0.15, -0.15], 'k', 'LineWidth', 1.5);

    % Cart body
    cart_x = cart_pos - cart_width/2;
    cart_y = 0;

    rectangle('Position', ...
        [cart_x, cart_y, cart_width, cart_height], ...
        'FaceColor', [0.3 0.7 0.9], ...
        'EdgeColor', 'k', ...
        'LineWidth', 1.5);

    % Wheels
    wheel_y = cart_y - wheel_radius;

    rectangle('Position', ...
        [cart_pos - cart_width/3 - wheel_radius, ...
         wheel_y - wheel_radius, ...
         2*wheel_radius, ...
         2*wheel_radius], ...
        'Curvature', [1, 1], ...
        'FaceColor', 'k');

    rectangle('Position', ...
        [cart_pos + cart_width/3 - wheel_radius, ...
         wheel_y - wheel_radius, ...
         2*wheel_radius, ...
         2*wheel_radius], ...
        'Curvature', [1, 1], ...
        'FaceColor', 'k');

    % Pivot point
    pivot_x = cart_pos;
    pivot_y = cart_y + cart_height;

    % Pole end point
    pole_x = pivot_x + pole_length * sin(theta);
    pole_y = pivot_y - pole_length * cos(theta);

    % Pole
    plot([pivot_x, pole_x], [pivot_y, pole_y], ...
        'r', 'LineWidth', 4);

    % Pole mass
    plot(pole_x, pole_y, 'ko', ...
        'MarkerSize', 10, ...
        'MarkerFaceColor', 'y');

    % Pivot point
    plot(pivot_x, pivot_y, 'ko', ...
        'MarkerSize', 6, ...
        'MarkerFaceColor', 'k');

    % Plot settings
    xlim([-2.5, 2.5]);
    ylim([-1.0, 1.5]);

    xlabel('Cart position');
    ylabel('Height');
    title(['Manual LQR Cart-Pole Animation, t = ', num2str(t, '%.2f'), ' s']);

    drawnow;

end