clear; clc;

%% 1. Build grid map
nRow = 10;
nCol = 10;
nS = nRow * nCol;

goal = sub2ind([nRow, nCol], 10, 10);

gamma = 0.9;
theta = 1e-6;

% Actions: up, down, left, right
actions = [-1 0; 1 0; 0 -1; 0 1];
nA = size(actions, 1);

%% 2. Define a simple policy
% Random policy: each action has probability 0.25
pi = ones(nS, nA) / nA;

% Goal state has no action cost
pi(goal, :) = 0;

%% 3. Initialize value function
V = zeros(nS, 1);

%% 4. Bellman Expectation Iteration
num_iter = 0;
while true
    num_iter = num_iter + 1;
    delta = 0;

    for s = 1:nS
        if s == goal
            continue;
        end

        old_v = V(s);
        new_v = 0;

        [r, c] = ind2sub([nRow, nCol], s);

        for a = 1:nA
            nr = r + actions(a, 1);
            nc = c + actions(a, 2);

            % If hit wall, stay in same state
            if nr < 1 || nr > nRow || nc < 1 || nc > nCol
                s_next = s;
            else
                s_next = sub2ind([nRow, nCol], nr, nc);
            end

            % Reward means consumption
            if s_next == goal
                reward = 0;
            else
                reward = -1;
            end

            new_v = new_v + pi(s,a) * (reward + gamma * V(s_next));
        end

        V(s) = new_v;
        delta = max(delta, abs(old_v - V(s)));
    end

    if delta < theta
        break;
    end
end

%% 5. Display value function as grid
fprintf(">> number of iteration: %d", num_iter)
V_grid = reshape(V, [nRow, nCol]);
disp(V_grid);

%% 6. Heatmap visualization

figure(1);
V_grid = reshape(V, [nRow, nCol]);

imagesc(V_grid);         % plot heatmap
colormap('jet');         % color style
colorbar;                % show legend (color scale)

% Flip Y-axis so (1,1) is bottom-left like grid
set(gca, 'YDir', 'normal');

% Add labels
xlabel('Column');
ylabel('Row');
title('Value Function Heatmap (Expected Cost-to-Go)');

% Show values on each cell
for i = 1:nRow
    for j = 1:nCol
        text(j, i, sprintf('%.2f', V_grid(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'Color', 'w', 'FontWeight', 'bold');
    end
end

% Highlight goal
[row_g, col_g] = ind2sub([nRow, nCol], goal);
text(col_g, row_g, 'GOAL', ...
    'HorizontalAlignment', 'center', ...
    'Color', 'k', 'FontWeight', 'bold');

figure(2);
surf(1:nRow, 1:nCol, V_grid);
title('Surface Plot');
xlabel('x'); ylabel('y'); zlabel('Value');