clear; clc; close all;

syms x y

% Saddle function
f = x^2 - y^2;

% Hessian matrix
H = hessian(f, [x y])

% Eigenvalues of Hessian
% Eigenvalues tell you curvature directions
% Mixed signs = saddle = optimization gets tricky
eig_H = eig(H)

% Plot surface
fsurf(f)
xlabel('x')
ylabel('y')
zlabel('f(x,y)')
title('Saddle Surface: f(x,y) = x^2 - y^2')
grid on