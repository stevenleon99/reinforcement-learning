clear; clc; close all;

syms x y

% Saddle function
f = x^2 - y^2;

% Hessian matrix
H = jacobian(f, [x y])

% Plot surface
fsurf(f)
xlabel('x')
ylabel('y')
zlabel('f(x,y)')
title('Saddle Surface: f(x,y) = x^2 - y^2')
grid on