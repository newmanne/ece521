%% Load the variables

ClassificationX = load('ClassificationX.txt');
ClassificationY = load('ClassificationY.txt');
RegressionX = load('RegressionX.txt');
RegressionY = load('RegressionY.txt');

%% Split into training, validation, and testing

training_ind = 1:50;
validation_ind = 51:100;
test_ind = 101:200;

CXTraining = ClassificationX(training_ind);
CYTraining = ClassificationY(training_ind);
RXTraining = RegressionX(training_ind);
RYTraining = RegressionY(training_ind);

CXValidation = ClassificationX(validation_ind);
CYValidation = ClassificationY(validation_ind);
RXValidation = RegressionX(validation_ind);
RYValidation = RegressionY(validation_ind);

CXTest = ClassificationX(test_ind);
CYTest = ClassificationY(test_ind);
RXTest = RegressionX(test_ind);
RYTest = RegressionY(test_ind);

%% 1a - 1 Nearest Neighbour

RYPredictions = knn(RXTest, RXTraining, RYTraining, 1, 1);
fprintf('MSE for k=1 is %f\n', meanSquaredError(RYPredictions, RYTest));

% Plot
plotKnn(RXTest, RYPredictions, RXTraining, RYTraining, 1);

%% 1b - K Nearest Neighbour

MSEs = zeros(10, 1);
for k = 1:10
    RYPredictions = knn(RXValidation, RXTraining, RYTraining, k, 1);
    MSEs(k) = meanSquaredError(RYPredictions, RYValidation);
    fprintf('MSE for k=%d is %f\n', k, MSEs(k));
%     plotKnn(RXValidation, RYPredictions, RXTraining, RYTraining, k);
end
[min_MSE, best_k] = min(MSEs);
fprintf('Lowest MSE for k=%d with MSE value of %f\n', best_k, min_MSE);

% Using the best k, try on the test set
RYPredictions = knn(RXTest, RXTraining, RYTraining, best_k, 1);
fprintf('On the test data, MSE for k=%d is %f\n' , best_k, meanSquaredError(RYPredictions, RYTest));

plotKnn(RXTest, RYPredictions, RXTraining, RYTraining, best_k);

%% 1c - Linear regression. 
% Use the steepest descent method to fit a linear regression model to the training data. Make predictions for the test cases and report the MSE

learningRate = 0.0001;
numIters = 50000;

[RXTrainingNormalized, mu, sigma] = normalize_features(RXTraining);
[weights, MSEs] = linearRegression(learningRate, numIters, RXTrainingNormalized, RYTraining);

XTestNormalized = normalize(RXTest, mu, sigma);
YPred = applyWeights(XTestNormalized, weights);
mse = meanSquaredError(YPred, RYTest);

display(weights);
fprintf('MSE on test data is %f\n', mse);

% hold on
% domain = linspace(min(XTestNormalized), max(XTestNormalized))';
% plot(domain, applyWeights(domain, weights));
% plot(XTestNormalized, RYTest, 'b.');
% hold off

%% 1d - Linear regression with polynomial inputs

% d) Linear regression using polynomial inputs. Expand the input x to form a vector of inputs (x, 
% ), for a positive integer k. Use steepest descent to fit linear regression models to the 
% training data, for k ranging from 1 to 10. Note that for a given value of k, there will be k+1 
% parameters. For each model, make predictions for the validation cases. Compute and report the 
% 10 corresponding values of the MSE. Which value of k achieves the lowest validation error? Use 
% that value of k to make predictions for the 100 test cases and report the MSE on the test set. 
% Discuss how this value compares to the MSE reported in (1c).

learningRate = 0.0001;
numIters = 50000;

RegressionXExpanded = zeros(size(RegressionX, 1), 10);
for k = 1:10
    RegressionXExpanded(:, k) = RegressionX .^ k;
end

RXTestExpanded = RegressionXExpanded(test_ind, :);
RXValidationExpanded = RegressionXExpanded(validation_ind, :);
RXTrainingExpanded = RegressionXExpanded(training_ind, :);

weights = zeros(10, 11);
MSEs = zeros(10, 1);

[RXTrainingNormalized, mu, sigma] = normalize_features(RXTrainingExpanded);

for k = 1:10
    % Fit a model
    XData = RXTrainingNormalized(:, 1:k);
    k_weights = linearRegression(learningRate, numIters, XData, RYTraining);
    weights(k, 1:k+1) = k_weights;
    
    % Make predictions for the validation cases
    XValidationData = normalize(RXValidationExpanded(:, 1:k), mu(1:k), sigma(1:k));
    YPred = applyWeights(XValidationData, k_weights);
    MSEs(k) = meanSquaredError(YPred, RYValidation);
end

display(MSEs);
[min_MSE, best_k] = min(MSEs);
fprintf('Minimum mse for k=%d is %f\n', best_k, min_MSE);

XTestData = normalize(RXTestExpanded(:, 1:best_k), mu(1:best_k), sigma(1:best_k));
YPred = applyWeights(XTestData, weights(best_k, 1:best_k+1));
mse = meanSquaredError(YPred, RYTest);

fprintf('MSE for test set is %f\n', mse);

%% 2a 

% a) Nearest neighbors. Use the nearest neighbor method to make label predictions for the 100 test 
% cases using the 50 training cases. Compute and report the classification error rate on the test set. 
% Also, compute and report the 2 x 2 confusion matrix for the test data.

CYPred = knn(CXTest, CXTraining, CYTraining, 1, 0); 

fprintf('Classification error rate for k=1 is %f\n', classificationErrorRate(CYPred, CYTest));

% Plot
% plotKnn(CXTest, CYPred, CXTraining, CYTraining, 1);

% Confusion Matrix
confusion = confusionMatrix(CYPred, CYTest);
display(confusion);

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
    CYPred = knn(CXValidation, CXTraining, CYTraining, k, 0);
    classificationErrorRates(k_ind) = classificationErrorRate(CYPred, CYValidation);
    fprintf('classification error for k=%d is %f\n', k, classificationErrorRates(k_ind));
%     plotKnn(RXValidation, RYPredictions(:, k), RXTraining, RYTraining, k);
end

[min_error_rate, best_k_ind] = min(classificationErrorRates);
best_k = kvals(best_k_ind);

fprintf('Lowest classification error for k=%d with value of %f\n', best_k, min_error_rate);

% Using the best k, try on the test set
CYPred = knn(CXTest, CXTraining, CYTraining, best_k, 0);
errorRate = classificationErrorRate(CYPred, CYTest);
fprintf('On the test data, classification errror for k=%d is %f\n' , best_k, errorRate);

% Confusion Matrix
confusion = confusionMatrix(CYPred, CYTest);
display(confusion);

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
    CYPredProb = knn(CXValidation, CXTraining, CYTraining, k, 2);
    logLikelikehoods(k_ind) = logLikelihood(CYPredProb, CYValidation);
    fprintf('log likelihood for k=%d is %f\n', k, logLikelikehoods(k_ind));
%     plotKnn(RXValidation, RYPredictions(:, k), RXTraining, RYTraining, k);
end

[max_likelihood, best_k_ind] = max(logLikelikehoods);
best_k = kvals(best_k_ind);

fprintf('Lowest validation error for k=%d with log likelihood of  %f\n', best_k, max_likelihood);

% Using the best k, try on the test set
CYPred = knn(CXTest, CXTraining, CYTraining, best_k, 0);
errorRate = classificationErrorRate(CYPred, CYTest);
CYPredProb = knn(CXTest, CXTraining, CYTraining, best_k, 2);
logLike = logLikelihood(CYPredProb, CYTest);
fprintf('On the test data for k=%d log likelihood is %f, classifcation error is %f\n' , best_k, logLike, errorRate);

%% 2d

% Linear regression. We can cheat and pretend that the labels 0 and 1 are real numbers. Use the 
% steepest descent method to fit a linear regression model (two parameters) to the training data. 
% Apply this model to the test data. The resulting predictions will be real numbers. Apply a 
% threshold of 0.5 to obtain binary predictions and report the classification error rate. Compare to 
% the rates obtained in (2a) to (2c). To test the effect of the threshold, repeat the above procedure 
% for a range of thresholds and plot the test error rate versus the threshold.

learningRate = 0.0001;
numIters = 50000;

[CXTrainingNormalized, mu, sigma] = normalize_features(CXTraining);
weights = linearRegression(learningRate, numIters, CXTrainingNormalized, CYTraining);

XTestNormalized = normalize(CXTest, mu, sigma);
YPred = applyWeights(XTestNormalized, weights);

thresholds = 0.1:0.1:0.9;
classificationErrorRates = zeros(length(thresholds), 1);

for i = 1:length(thresholds);
    predictions = YPred > thresholds(i);
    classificationErrorRates(i) = classificationErrorRate(predictions, CYTest);
end

plot(thresholds, classificationErrorRates, 'ob');

%% 2e

% e) Linear regression using polynomial inputs. Expand the input x to form a vector of inputs (x, ), for a positive integer k. Use steepest descent to fit linear regression models to the 
% training data, for k ranging from 1 to 10. For each model, threshold the predictions for the 
% validation cases and compute and report the 10 corresponding values of the validation error 
% rate. Which value of k achieves the lowest validation error? Use that value of k to make 
% predictions for the 100 test cases and report the test error rate. Discuss how this value compares 
% to the error rates reported in (2a) to (2d). 

learningRate = 0.0001;
numIters = 50000;
thresh = 0.5;

ClassificationXExpanded = zeros(size(ClassificationX, 1), 10);
for k = 1:10
    ClassificationXExpanded(:, k) = ClassificationX .^ k;
end

CXTestExpanded = ClassificationXExpanded(test_ind, :);
CXValidationExpanded = ClassificationXExpanded(validation_ind, :);
CXTrainingExpanded = ClassificationXExpanded(training_ind, :);

weights = zeros(10, 11);

[CXTrainingNormalized, mu, sigma] = normalize_features(CXTrainingExpanded);

classificationErrorRates = zeros(10, 1);

for k = 1:10
    % Fit a model
    XData = CXTrainingNormalized(:, 1:k);
    k_weights = linearRegression(learningRate, numIters, XData, CYTraining);
    weights(k, 1:k+1) = k_weights;
    
    % Make predictions for the validation cases
    XValidationData = normalize(CXValidationExpanded(:, 1:k), mu(1:k), sigma(1:k));
    YPred = applyWeights(XValidationData, k_weights) > thresh;
    classificationErrorRates(k) = classificationErrorRate(YPred, CYValidation);
end

display(classificationErrorRates);
[min_CE, best_k] = min(classificationErrorRates);
fprintf('Minimum ce for k=%d is %f\n', best_k, min_CE);

XTestData = normalize(CXTestExpanded(:, 1:best_k), mu(1:best_k), sigma(1:best_k));
YPred = applyWeights(XTestData, weights(best_k, 1:best_k+1)) > thresh;
ce = classificationErrorRate(YPred, CYTest);

fprintf('CE for test set is %f\n', ce);

%% 2f

% e) Linear regression using polynomial inputs. Expand the input x to form a vector of inputs (x, 
% ), for a positive integer k. Use steepest descent to fit linear regression models to the 
% training data, for k ranging from 1 to 10. For each model, threshold the predictions for the 
% validation cases and compute and report the 10 corresponding values of the validation error 
% rate. Which value of k achieves the lowest validation error? Use that value of k to make 
% predictions for the 100 test cases and report the test error rate. Discuss how this value compares 
% to the error rates reported in (2a) to (2d). 

