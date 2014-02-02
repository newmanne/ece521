[RXTraining, RYTraining, RXValidation, RYValidation, RXTest, RYTest] = loadVars();

%%
% e) Neural networks: Plotting training error and selecting the learning rate. Implement a steepest 
% descent learning algorithm for training a neural network with a single input, one hidden layer of 
% k logistic units, and a single linear output unit. The parameters (10 in all) should be initialized 
% using random values uniformly distributed between -.1 and .1. Your software should compute 
% and store the training error after each epoch (pass through the dataset), so you can plot the 
% training error against the number of epochs. For k=5, train the neural network for 10,000 epochs, 
% using a variety of learning rates. High learning rates may cause unwanted oscillations in the 
% training error, whereas low learning rates will lead to very slow learning. Submit a single figure 
% that plots the training error vs the logarithm of the number of epochs, for a variety of learning 
% rates (there should be one curve for each learning rate). Select a learning rate that balances these 
% two concerns and report its value. Use this learning rate in subsequent experiments, but get in the 
% habit of plotting the training error so as to make sure the learning rate you’re using is 
% appropriate. Using the model that was trained using the selected learning rate, report the test 
% error. 

% plot(RegressionX, RegressionY, 'ro');
k = 5;
nIters = 10000;
learningRates = zeros(1, 5);
figure
title('Training error vs Epoch for different LR');
xlabel('Epoch');
ylabel('Training error (MSE)');
cc = hsv(length(learningRates));
hold on
for i = 1:length(learningRates)
    learningRates(i) = 0.01./(10^i);
    [w, error] = trainNN(k, learningRates(i), nIters, RXTraining, RYTraining);       
    semilogx(1:nIters, error, 'color', cc(i, :));
end
legend(cellfun(@num2str,num2cell(learningRates),'uniformoutput',0));
hold off

%% select lr = 0.001
learningRate = 0.001;
[w, ~] = trainNN(k, learningRate, nIters, RXTraining, RYTraining);       
YPred = applyNN(w, RXTest);
error = meanSquaredError(YPred, RYTest);
fprintf('Error %f\n', error);
% plot(RXTest, RYTest, 'bo', RXTest, YPred, 'g+');

%%
% f) Neural networks: Random restarts. Sometimes, the training algorithm can get stuck in poor 
% local minima of the training error. To help avoid this, add a loop in your software so that training 
% is repeated 20 times using 20 different random parameter initializations and the model with 
% lowest training error is selected. For k=5, submit a histogram of the 20 resulting training error 
% values. How much variability is there? Compute and report the test error and compare it to the 
% test error obtained in (1e).

k = 5;
nIters = 10000;
numTrials = 20;
learningRate = 0.001;
errors = zeros(numTrials, 1);
weights = cell(20, 1);

for i = 1:numTrials
    [weights{i}, error] = trainNN(k, learningRate, nIters, RXTraining, RYTraining);       
    errors(i) = error(end);
end

hist(errors);
title('Histogram of Training Error');
xlabel('Training error');
ylabel('Counts');

[~, bestInd] = min(errors);

YPred = applyNN(weights{bestInd}, RXTest);
error = meanSquaredError(YPred, RYTest);
fprintf('Error %f\n', error);


%%
% g) Neural networks: Selecting the number of hidden units. Using the learning rate selected in (e) 
% and the 20-random-restart method developed in (f), train neural networks with k ranging from 1 
% to 10. (Note that for each value of k, you will train 20 models and pick the one with the lowest 
% training error.) Compute and report the validation error for these 10 models. Which value of k 
% achieves the lowest validation error? Use the model with lowest validation error to make 
% predictions for the 100 test cases and report the MSE on the test set. Discuss how this value 
% compares to the MSE reported in (1a) to (1f). 

nIters = 10000;
numTrials = 20;
maxK = 10;
learningRate = 0.001;
weights = cell(maxK, 1);
validationErrors = zeros(maxK, 1);
for k = 1:maxK
    minError = intmax;
    for i = 1:numTrials
        [w, error] = trainNN(k, learningRate, nIters, RXTraining, RYTraining);       
        if error(end) < minError
            minError = error(end);
            weights{k} = w;
        end
    end
    validationErrors(k) = applyNN(weights{k}, RXValidation, RYValidation);
end

[~, bestK] = min(validationErrors);
error = applyNN(weights{bestK}, RXTest, RYTest);
fprintf('Error %f\n', error);


%%
% h) Neural networks: Early stopping. Modify your software so that it computes the validation 
% error and stores the model parameters after each epoch of learning. For k=10, generate and 
% submit a single plot that includes both the training error and the validation error versus the 
% number of epochs. (You do not need to use the 20-random-restart method here.) Determine the 
% number of epochs that achieves the lowest validation error and report this value. Report the test 
% error for this model and discuss how this value compares to those from (1a) to (1g). 

k = 10;
learningRate = 0.001;
nIters = 1000000;
trainingError = zeros(nIters, 1);
validationError = zeros(nIters, 1);
N = length(RXTraining);
weights = zeros(k + 3, k + 3, nIters);

w = zeros(k + 3, k + 3);
w(1, 3:k+3) = rand(1, length(3:k+3));
w(2, 3:k+2) = rand(1, length(3:k+2));
w(3:k+2, k+3) = rand(length(3:k+2), 1);

xx = zeros(N, k + 3);
xx(:, 1) = 1;
xx(:, 2) = RXTraining;

g = zeros(N, k+3);
g(:, 1) = 1;
g(:, 2) = RXTraining;

dedx = zeros(N, k+3);

for i = 1:nIters
    for j = 3:k+2
        xx(:, j) = g * w(:, j);
        g(:, j) = 1./(1 + exp(-xx(:, j)));
    end
    xx(:, k+3) = g * w(:, k+3);
    g(:, k+3) = xx(:, k+3);
    trainingError(i) = meanSquaredError(g(:,k+3), RYTraining);
    validationError(i) = applyNN(w, RXValidation, RYValidation);
    weights(:, :, i) = w;
    dedx(:, k+3) = 2 * (g(:,k+3) - RYTraining);
    for m = k+2:-1:3
        dedx(:, m) = dedx(:, m+1:k+3) * w(m, m+1:k+3)' .* g(:, m) .* (1 - g(:,m));
    end
    del = g' * dedx;
    w = w - learningRate * del .* (w ~= 0);
end
plot(1:nIters, trainingError, 'r-', 1:nIters, validationError, 'b-');
title('Training/Validation Error vs Epochs');
xlabel('Epoch');
ylabel('Error');
legend('Training Error', 'Validation Error');

[~, minInd] = min(validationError);
fprintf('Using weights from epoch %d out of %d\n', minInd, nIters);
bestWeights = weights(:, :, minInd);
testError = applyNN(bestWeights, RXTest, RYTest);
fprintf('Test Error %f\n', testError);

%%



