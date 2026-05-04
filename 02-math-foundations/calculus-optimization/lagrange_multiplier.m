clear; clc; close all;

syms x y

% Objective and constraint
f = x^2 + y^2;
g = x + y - 1;

% Optimum point
x0 = 0.5;
y0 = 0.5;
z0 = double(subs(f, [x y], [x0 y0]));

figure

% Plot objective surface
fsurf(f, [-1 2 -1 2])
hold on
fsurf(g, [-1 2 -1 2])
hold on
% Plot constraint curve on the surface
t = linspace(-1, 2, 200);
xc = t;
yc = 1 - t;
zc = xc.^2 + yc.^2;

plot3(xc, yc, zc, 'r', 'LineWidth', 3)

% Plot optimum point
plot3(x0, y0, z0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k')

% Labels
xlabel('x')
ylabel('y')
zlabel('f(x,y)')
title('Lagrange Multiplier: Constraint Curve and Tangent Optimum')

legend('Objective surface f(x,y)', ...
       'Constraint curve x + y = 1', ...
       'Optimum point')

grid on
view(45, 30)