%% Load vars
[RXTraining, RYTraining, RXValidation, RYValidation, RXTest, RYTest] = loadVars();
%% Define helpers
knnWrapper = @(X, Y, k) knn(X, Y, RXTraining, RYTraining, k, @mean, @meanSquaredError);
linearRegressionWrapper = @(X, Y) linearRegression(X, Y, 0.01, 100000, RXTraining, RYTraining, @meanSquaredError);
%% 1a
% a) Nearest neighbors. Use the nearest neighbor method to make predictions for the 100 test cases 
% using the 50 training cases. Compute and report the mean squared error (MSE) on the test set. 

mse = knnWrapper(RXTest, RYTest, 1);
fprintf('MSE for test set on k=1 is %f\n', mse);
%% 1b 
% b) k-nearest neighbors. For k ranging from 1 to 10, use the k-nearest neighbor method to make 
% predictions for the 50 validation cases. Compute and report the 10 corresponding values of the 
% MSE. Which value of k achieves the lowest validation error? Use that value of k to make 
% predictions for the 100 test cases and report the MSE on the test set. Discuss how this value 
% compares to the MSE reported in (1a).

MSEs = zeros(10, 1);
for k = 1:10
    MSEs(k) = knnWrapper(RXValidation, RYValidation, k);
    fprintf('MSE for validation set on k=%d is %f\n', k, MSEs(k));
end
[minMSE, bestK] = min(MSEs);
fprintf('Lowest MSE for k=%d with MSE value of %f\n', bestK, minMSE);

% Using the best k, try on the test set
mse = knnWrapper(RXTest, RYTest, bestK);
fprintf('On the test data, MSE for k=%d is %f\n' , bestK, mse);

% The MSE is lower than in part 1a.
%% 1c
% c) Linear regression. Use the steepest descent method to fit a linear regression model (two 
% parameters, randomly initialized uniformly between -0.1 and 0.1) to the training data. Make 
% predictions for the test cases and report the MSE.

mse = linearRegressionWrapper(RXTest, RYTest);
fprintf('MSE on test data is %f\n', mse);
%% 1d
% d) Linear regression using polynomial inputs. Expand the input x to form a vector of inputs (x, 
% ), for a positive integer k. Use steepest descent to fit linear regression models to the 
% training data, for k ranging from 1 to 10. Note that for a given value of k, there will be k+1 
% parameters. For each model, make predictions for the validation cases. Compute and report the 
% 10 corresponding values of the MSE. Which value of k achieves the lowest validation error? Use 
% that value of k to make predictions for the 100 test cases and report the MSE on the test set. 
% Discuss how this value compares to the MSE reported in (1c).

weights = cell(10, 1);
MSEs = zeros(10, 1);

[XTrain, XValid, XTest] = polyExpand(load('RegressionX.txt'));

for k = 1:10
    [MSEs(k), ~, weights{k}] = linearRegression(XValid(:, 1:k), RYValidation, 0.01, 100000, XTrain(:, 1:k), RYTraining, @meanSquaredError);
end

display(MSEs);
[minMSE, bestK] = min(MSEs);
fprintf('Minimum mse for k=%d is %f\n', bestK, minMSE);

mse = meanSquaredError(applyWeights(XTest(:, 1:bestK), weights{bestK}), RYTest);
fprintf('MSE for test set is %f\n', mse);

% Value is better than what was recorded in 1c

