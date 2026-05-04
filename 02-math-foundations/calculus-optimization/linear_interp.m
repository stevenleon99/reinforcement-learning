% f = @(input) expression
f = @(x) x.^2;

x1 = 1;
x2 = 5;

x = linspace(x1, x2, 100);

% function values
y = f(x);

% line between two points
t = linspace(0, 1, 100);
x_line = t*x1 + (1-t)*x2;
% if the interpolation line is linear
y_line = t*f(x1) + (1-t)*f(x2);

plot(x, y, 'b', 'LineWidth', 2); hold on;
plot(x_line, y_line, 'r--', 'LineWidth', 2);

legend('f(x)=x^2 (convex)', 'line between points');
title('Convexity visualization');
grid on;