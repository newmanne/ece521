%% Load the variables
RegressionX = load('RegressionX.txt');
RegressionY = load('RegressionY.txt');

%% Split into training, validation, and testing

training_ind = 1:50;
validation_ind = 51:100;
test_ind = 101:200;

RXTraining = RegressionX(training_ind);
RYTraining = RegressionY(training_ind);

RXValidation = RegressionX(validation_ind);
RYValidation = RegressionY(validation_ind);

RXTest = RegressionX(test_ind);
RYTest = RegressionY(test_ind);

%% Normalize
[RXTraining, mu, sigma] = normalize_features(RXTraining);
RXValidation = normalizeFromMuSigma(RXValidation, mu, sigma);
RXTest = normalizeFromMuSigma(RXTest, mu, sigma);

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
learningRates = ones(5, 1);
for i = 1:length(learningRates)
    learningRates(i) = 0.01./(10^i);
end
nIters = 10000;
error = zeros(nIters, length(learningRates));
N = length(RXTraining);

for q = 1:length(learningRates)
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
        error(i, q) = meanSquaredError(g(:,k+3), RYTraining);
        dedx(:, k+3) = 2 * (g(:,k+3) - RYTraining);
        for m = k+2:-1:3
            dedx(:, m) = dedx(:, m+1:k+3) * w(m, m+1:k+3)' .* g(:, m) .* (1 - g(:,m));
        end
        del = g' * dedx;
        w = w - learningRates(q) * del .* (w ~= 0);
    %     if (mod(it, 1000) == 0)
    %         plot(RXTraining, RYTraining, 'bo', RXTraining, g(:, k+3), 'g+');
    %         pause(0.5);
    %     end
    end
    % plot(RXTraining, g(:, k+3), 'y+');
end

% semilogx(1:nIters, error(:, 1), 'ro', 1:nIters, error(:, 2), 'yo', 1:nIters, error(:, 3), 'bo');
figure
cc=hsv(length(learningRates));
hold on
for i = 1:length(learningRates)
    semilogx(1:nIters, error(:, i), 'color', cc(i, :));
end
title('Training error vs Epoch for different LR');
xlabel('Epoch');
ylabel('Training error (MSE)');
legend(cellfun(@num2str,num2cell(learningRates),'uniformoutput',0));
hold off

% TODO: report test error

%%
% f) Neural networks: Random restarts. Sometimes, the training algorithm can get stuck in poor 
% local minima of the training error. To help avoid this, add a loop in your software so that training 
% is repeated 20 times using 20 different random parameter initializations and the model with 
% lowest training error is selected. For k=5, submit a histogram of the 20 resulting training error 
% values. How much variability is there? Compute and report the test error and compare it to the 
% test error obtained in (1e).

k = 5;
numTrials = 20;
learningRate = 0.001;
nIters = 10000;
error = zeros(nIters, numTrials);
N = length(RXTraining);

for q = 1:numTrials
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
        error(i, q) = meanSquaredError(g(:,k+3), RYTraining);
        dedx(:, k+3) = 2 * (g(:,k+3) - RYTraining);
        for m = k+2:-1:3
            dedx(:, m) = dedx(:, m+1:k+3) * w(m, m+1:k+3)' .* g(:, m) .* (1 - g(:,m));
        end
        del = g' * dedx;
        w = w - learningRates(q) * del .* (w ~= 0);
    %     if (mod(it, 1000) == 0)
    %         plot(RXTraining, RYTraining, 'bo', RXTraining, g(:, k+3), 'g+');
    %         pause(0.5);
    %     end
    end
    % plot(RXTraining, g(:, k+3), 'y+');
end
