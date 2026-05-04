clear; clc;

syms x lambda

% Objective
f = x^2;

% Inequality constraint: x >= 1
% Convert to g(x) <= 0
g = 1 - x;

% Lagrangian
L = f + lambda*g;

% KKT conditions
stationarity = diff(L, x) == 0;
primal_feasibility = g <= 0;
dual_feasibility = lambda >= 0;
complementary_slackness = lambda*g == 0;

% Solve stationarity + complementary slackness
sol = solve([stationarity, complementary_slackness], [x, lambda], 'Real', true);

% Display candidates
sol.x
sol.lambda

for i = 1:length(sol.x)
    x_val = sol.x(i);
    lambda_val = sol.lambda(i);

    fprintf('Candidate %d:\n', i)
    fprintf('x = %s\n', string(x_val))
    fprintf('lambda = %s\n', string(lambda_val))

    g_val = subs(g, x, x_val);

    fprintf('g(x) = %s\n', string(g_val))
    % constraint is tight
    % OR multiplier is zero
    fprintf('lambda >= 0: %d\n', isAlways(lambda_val >= 0))
    fprintf('g(x) <= 0: %d\n', isAlways(g_val <= 0))
    fprintf('f(x) = %s\n\n', string(subs(f, x, x_val)))
end