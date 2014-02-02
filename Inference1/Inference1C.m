%% Load variables
[~, ~, ~, ~, ~, ~, CXTraining, CYTraining, CXValidation, CYValidation, CXTest, CYTest] = loadVars( );
%% Define helpers
pseudocount = @(x) 0.1 + length(find(x == 1));
knnWrapper = @(X, Y, k, f, e) knn(X, Y, CXTraining, CYTraining, k, f, e);
%% 2a 
% a) Nearest neighbors. Use the nearest neighbor method to make label predictions for the 100 test 
% cases using the 50 training cases. Compute and report the classification error rate on the test set. 
% Also, compute and report the 2 x 2 confusion matrix for the test data.

[ce, YPred] = knnWrapper(CXTest, CYTest, 1, @mode, @classificationErrorRate); 
fprintf('Classification error rate for k=1 is %f\n', ce);

% Confusion Matrix
confusionMatrix(YPred, CYTest);
%% 2b

% b) k-nearest neighbors: Voting. For k=1,3,5,…,11, use the k-nearest neighbor method to make 
% predictions for the 50 validation cases, using voting to decide the most likely label. Compute and 
% report the 10 corresponding values of the validation error rate. Which value of k achieves the 
% lowest validation error? Use that value of k to make predictions for the 100 test cases and report 
% the error rate on the test set. Discuss how this value compares to the error rate reported in (2a). 
% Report the 2 x 2 confusion matrix for the test data

kvals = 1:2:11;
classificationErrorRates = zeros(length(kvals), 1);

for k_ind = 1:length(kvals)
    k = kvals(k_ind);
    classificationErrorRates(k_ind) = knnWrapper(CXValidation, CYValidation, k, @mode, @classificationErrorRate);
    fprintf('classification error for k=%d is %f\n', k, classificationErrorRates(k_ind));
end

[min_error_rate, bestKInd] = min(classificationErrorRates);
bestK = kvals(bestKInd);
fprintf('Lowest classification error for k=%d with value of %f\n', bestK, min_error_rate);

% Using the best k, try on the test set
[ce, YPred] = knnWrapper(CXTest, CYTest, bestK, @mode, @classificationErrorRate);
fprintf('On the test data, classification errror for k=%d is %f\n' , bestK, ce);

% Confusion Matrix
confusionMatrix(YPred, CYTest);
%% 2c
% c) k-nearest neighbors: Probabilities. Modify your software so that instead of using the k-nearest 
% neighbors to vote on the label, they are used to estimate the probabilities that the label is 0 or 1, 
% using the proportion of labels with the value 0 and 1. Include a pseudocount of 0.1 in each class 
% to prevent zero-probabilities; for example, for k=3 if there are two 0’s and one 1, the 
% probabilities of 0 and 1 would be 2.1/3.2 and 1.1/3.2 (note that these sum to one!) For the above 
% values of k, compute and report the log-likelihood of the validation set. Which value of k 
% achieves the lowest validation error, and is it the same as the value of k found using voting? Use that value of k to make predictions for the 100 test cases and report the test log-likelihood and 
% also the test classification error rate.

kvals = 1:2:11;
logLikelikehoods = zeros(length(kvals), 1);

for k_ind = 1:length(kvals)
    k = kvals(k_ind);
    logLikelikehoods(k_ind) = knnWrapper(CXValidation, CYValidation, k, pseudocount, @logLikelihood);
    fprintf('log likelihood for k=%d is %f\n', k, logLikelikehoods(k_ind));
end

[max_likelihood, bestKInd] = max(logLikelikehoods);
bestK = kvals(bestKInd);

fprintf('Lowest validation error for k=%d with log likelihood of  %f\n', bestK, max_likelihood);

% Using the best k, try on the test set
ce = knnWrapper(CXTest, CYTest, bestK, @mode, @classificationErrorRate);
ll = knnWrapper(CXTest, CYTest, bestK, pseudocount, @logLikelihood);
fprintf('On the test data for k=%d log likelihood is %f, classifcation error is %f\n' , bestK, ll, ce);

%% 2d

% Linear regression. We can cheat and pretend that the labels 0 and 1 are real numbers. Use the 
% steepest descent method to fit a linear regression model (two parameters) to the training data. 
% Apply this model to the test data. The resulting predictions will be real numbers. Apply a 
% threshold of 0.5 to obtain binary predictions and report the classification error rate. Compare to 
% the rates obtained in (2a) to (2c). To test the effect of the threshold, repeat the above procedure 
% for a range of thresholds and plot the test error rate versus the threshold.

learningRate = 0.0001;
numIters = 100000;
thresholds = 0.1:0.1:0.9;
classificationErrorRates = zeros(length(thresholds), 1);
for i = 1:length(thresholds);
    thresholdCE = @(YPred, YActual) classificationErrorRate(YPred > thresholds(i), YActual);
    classificationErrorRates(i) = linearRegression(CXTest, CYTest, learningRate,  numIters, CXTraining, CYTraining, thresholdCE);
end

% TODO: Is this the error rate, or does it mean MSE?
plot(thresholds, classificationErrorRates, 'ob');
title('Test Error Rate vs Threshold');
xlabel('Threshold');
ylabel('Classification Error Rate');

%% 2e

% e) Linear regression using polynomial inputs. Expand the input x to form a vector of inputs (x, ), for a positive integer k. Use steepest descent to fit linear regression models to the 
% training data, for k ranging from 1 to 10. For each model, threshold the predictions for the 
% validation cases and compute and report the 10 corresponding values of the validation error 
% rate. Which value of k achieves the lowest validation error? Use that value of k to make 
% predictions for the 100 test cases and report the test error rate. Discuss how this value compares 
% to the error rates reported in (2a) to (2d). 


weights = cell(10, 1);
classificationErrors = zeros(10, 1);
thresh = 0.5;
[XTrain, XValid, XTest] = polyExpand(load('ClassificationX.txt'));
fixedThresholdCE = @(YPred, YActual) classificationErrorRate(YPred > thresh, YActual);
for k = 1:10
    [classificationErrors(k), ~, weights{k}] = linearRegression(XValid(:, 1:k), CYValidation, 0.01, 100000, XTrain(:, 1:k), CYTraining, fixedThresholdCE);
end

display(classificationErrors);
[minError, bestK] = min(classificationErrors);
fprintf('Minimum error for k=%d is %f\n', bestK, minError);

ce = classificationErrorRate(applyWeights(XTest(:, 1:bestK), weights{bestK}) > thresh, CYTest);
fprintf('Error for test set is %f\n', ce);

%% 2f
% ) Logistic regression using polynomial inputs. Use steepest descent to fit a logistic regression 
% model to the training data, using polynomial inputs with k ranging from 1 to 10. Submit a figure 
% that plots the log-likelihood of the training set versus the number of epochs, for each value of k

% TODO: complex number problems...

weights = cell(10, 1);
logLikelihoods = zeros(10, 1);
numIters = 100000;
thresh = 0.5;
[XTrain, XValid, XTest] = polyExpand(load('ClassificationX.txt'));
hold on
cc = hsv(length(1:10));
for k = 1:10
    [logLikelihoods(k), lls, weights{k}] = logisticRegression(XValid(:, 1:k), CYValidation, 0.01, numIters, XTrain(:, 1:k), CYTraining, @logLikelihood);
    plot(1:numIters, lls, 'color', cc(k, :));
end
legend(cellfun(@num2str,num2cell(1:10),'uniformoutput',0));
title('Log-likelihood vs Epochs');
xlabel('Epoch');
ylabel('Log-likelihood');

display(logLikelihoods);
[maxLikelihood, bestK] = max(logLikelihoods);
fprintf('Highest likelihood for k=%d is %f\n', bestK, maxLikelihood);

% Test data
loglikelihood = logLikelihood(applyWeights(XTest(:, 1:bestK), weights{bestK}), CYTest);
ce = classificationErrorRate(applyWeights(XTest(:, 1:bestK), weights{bestK}) > thresh, CYTest);
fprintf('Error for test set is %f, CE is %f\n', loglikelihood, ce);
