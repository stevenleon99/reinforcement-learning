clc; clear;

% Grid size
nRow = 10;
nCol = 10;
nState = nRow * nCol;

% Discount factor
gamma = 0.9;

% Actions: up, down, left, right
actions = [-1 0;   % up
            1 0;   % down
            0 -1;  % left
            0 1];  % right

nAction = size(actions, 1);

% Goal state: bottom-right corner
goalRow = nRow;
goalCol = nCol;
goalState = sub2ind([nRow, nCol], goalRow, goalCol);

% Initialize value function
V = zeros(nRow, nCol);

% Value iteration parameters
maxIter = 100;
theta = 1e-6;

for iter = 1:maxIter
    delta = 0;
    V_old = V;

    for r = 1:nRow
        for c = 1:nCol

            currentState = sub2ind([nRow, nCol], r, c);

            % Goal state value remains 0
            if currentState == goalState
                V(r, c) = 0;
                continue;
            end

            actionValues = zeros(nAction, 1);

            for a = 1:nAction
                newRow = r + actions(a, 1);
                newCol = c + actions(a, 2);

                % If action goes outside grid, stay in same state
                if newRow < 1 || newRow > nRow || newCol < 1 || newCol > nCol
                    newRow = r;
                    newCol = c;
                end

                % Reward
                reward = -1;

                % Bellman Optimality Equation
                actionValues(a) = reward + gamma * V_old(newRow, newCol);
            end

            % Choose the best action
            V(r, c) = max(actionValues);

            delta = max(delta, abs(V(r, c) - V_old(r, c)));
        end
    end

    if delta < theta
        break;
    end
end

disp("Optimal Value Function:");
disp(V);

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
[row_g, col_g] = ind2sub([nRow, nCol], [goalRow, goalCol]);
text(col_g, row_g, 'GOAL', ...
    'HorizontalAlignment', 'center', ...
    'Color', 'k', 'FontWeight', 'bold');

figure(2);
surf(1:nRow, 1:nCol, V_grid);
title('Surface Plot');
xlabel('x'); ylabel('y'); zlabel('Value');