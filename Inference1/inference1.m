%% Load the variables

ClassificationX = load('ClassificationX.txt');
ClassificationY = load('ClassificationY.txt');
RegressionX = load('RegressionX.txt');
RegressionY = load('RegressionY.txt');

%% Split into training, validation, and testing

CXTraining = ClassificationX(1:50);
CYTraining = ClassificationY(1:50);
RXTraining = RegressionX(1:50);
RYTraining = RegressionY(1:50);

CXValidation = ClassificationX(51:100);
CYValidation = ClassificationY(51:100);
RXValidation = RegressionX(51:100);
RYValidation = RegressionY(51:100);

CXTest = ClassificationX(101:200);
CYTest = ClassificationY(101:200);
RXTest = RegressionX(101:200);
RYTest = RegressionY(101:200);

%% 1a - 1 Nearest Neighbour

RYPredictions = zeros(length(RYTest), 1);
for i = 1:length(RXTest)
    RYPredictions(i) = knn(RXTest(i), RXTraining, RYTraining, 1, false);
end 

fprintf('MSE for k=1 is %f\n', meanSquaredError(RYPredictions, RYTest));

% Plot
plotKnn(RXTest, RYPredictions, RXTraining, RYTraining, 1);

%% 1b - K Nearest Neighbour

RYPredictions = zeros(length(RYValidation), 10);
MSEs = zeros(10, 1);
for k = 1:10
    for i = 1:length(RXValidation)
        RYPredictions(i, k) = knn(RXValidation(i), RXTraining, RYTraining, k, false);
    end
    MSEs(k) = meanSquaredError(RYPredictions(:, k), RYValidation);
    fprintf('MSE for k=%d is %f\n', k, MSEs(k));
%     plotKnn(RXValidation, RYPredictions(:, k), RXTraining, RYTraining, k);
end
[max_val, max_ind ] = min(MSEs);
fprintf('Lowest MSE for k=%d with MSE value of %f\n', max_ind, max_val);

% Using the best k, try on the test set
RYPredictions = zeros(length(RYTest), 1);
for i = 1:length(RXTest)
    RYPredictions(i) = knn(RXTest(i), RXTraining, RYTraining, max_ind, false);
end 

fprintf('On the test data, MSE for k=%d is %f\n' , max_ind, meanSquaredError(RYPredictions, RYTest));



%% 1c - Linear regression. 
% Use the steepest descent method to fit a linear regression model to the training data. Make predictions for the test cases and report the MSE
learningRate = 0.001;
numIters = 5000;
weights = linearRegression(learningRate, numIters, RXTraining, RYTraining);
weights
%TODO

%% 1d - Liear regression with polynomial inputs

% d) Linear regression using polynomial inputs. Expand the input x to form a vector of inputs (x, 
% ), for a positive integer k. Use steepest descent to fit linear regression models to the 
% training data, for k ranging from 1 to 10. Note that for a given value of k, there will be k+1 
% parameters. For each model, make predictions for the validation cases. Compute and report the 
% 10 corresponding values of the MSE. Which value of k achieves the lowest validation error? Use 
% that value of k to make predictions for the 100 test cases and report the MSE on the test set. 
% Discuss how this value compares to the MSE reported in (1c).

X = zeros(size(RXTraining, 1), 10);
func = '%f + %f*x';
for k = 1 : 2
    X(:, k) = RXTraining .^ k; 
    weights = linearRegression(0.000001, 10000, X(:, 1:k), RYTraining);
    YPred = applyWeights(X(:, 1:k), weights);
    figure
    hold on
    plot(RXTraining, RYTraining, 'ob');
    domain = [min(RXTraining) max(RXTraining)];
    ezplot(sprintf(func, weights), domain);
    hold off
    func = strcat(func, sprintf(' + %%f*x.^%d', k + 1))
    fprintf(func);

end
    

%% 2a 

% a) Nearest neighbors. Use the nearest neighbor method to make label predictions for the 100 test 
% cases using the 50 training cases. Compute and report the classification error rate on the test set. 
% Also, compute and report the 2 x 2 confusion matrix for the test data.

CYProb = zeros(length(CYTest), 1);
for i = 1:length(RXTest)
    CYProb(i) = knn(CXTest(i), CXTraining, CYTraining, 1, true);
end 

fprintf('Classification error rate for k=1 is %f\n', classificationErrorRate(CYProb, CYTest));
% Plot
plotKnn(CXTest, CYProb, CXTraining, CYTraining, 1);

% Confusion Matrix
%TODO

%% 2b

% b) k-nearest neighbors: Voting. For k=1,3,5,…,11, use the k-nearest neighbor method to make 
% predictions for the 50 validation cases, using voting to decide the most likely label. Compute and 
% report the 10 corresponding values of the validation error rate. Which value of k achieves the 
% lowest validation error? Use that value of k to make predictions for the 100 test cases and report 
% the error rate on the test set. Discuss how this value compares to the error rate reported in (2a). 
% Report the 2 x 2 confusion matrix for the test data

kvals = 1:2:11;
CYProb = zeros(length(CYValidation), length(kvals));

logLikelikehoods = zeros(length(kvals), 1);
for k_ind = 1 : length(kvals)
    k = kvals(k_ind);
    for i = 1:length(CXValidation)
        CYProb(i, k) = knn(CXValidation(i), CXTraining, CYTraining, k, true);
    end
    logLikelikehoods(k_ind) = classificationErrorRate(CYProb(:, k_ind), CYValidation);
    fprintf('classification error for k=%d is %f\n', k, logLikelikehoods(k_ind));
%     plotKnn(RXValidation, RYPredictions(:, k), RXTraining, RYTraining, k);
end
[max_val, max_ind] = min(logLikelikehoods);
fprintf('Lowest classification error for k=%d with value of %f\n', kvals(max_ind), max_val);

% Using the best k, try on the test set
CYProb = zeros(length(CYTest), 1);
for i = 1:length(CXTest)
    CYProb(i) = knn(CXTest(i), CXTraining, CYTraining, kvals(max_ind), true);
end 

fprintf('On the test data, classification errror for k=%d is %f\n' , kvals(max_ind), classificationErrorRate(CYProb, CYTest));

% Confusion Matrix
%TODO

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
CYProb = zeros(length(CYValidation), length(kvals));
logLikelikehoods = zeros(length(kvals), 1);

for k_ind = 1 : length(kvals)
    k = kvals(k_ind);
    for i = 1:length(CXValidation)
        CYProb(i, k) = knn_prob(CXValidation(i), CXTraining, CYTraining, k);
    end
    logLikelikehoods(k_ind) = logLikelihood(CYProb(:, k_ind), CYValidation);
    fprintf('log likelihood for k=%d is %f\n', k, logLikelikehoods(k_ind));
%     plotKnn(RXValidation, RYPredictions(:, k), RXTraining, RYTraining, k);
end
[max_val, max_ind] = max(logLikelikehoods);
fprintf('Highest log likelihood k=%d with value of %f\n', kvals(max_ind), max_val);

% % Using the best k, try on the test set
% CYProb = zeros(length(CYTest), 1);
% for i = 1:length(CXTest)
%     CYProb(i) = knn(CXTest(i), CXTraining, CYTraining, kvals(max_ind), true);
% end 
% 
% fprintf('On the test data, classification errror for k=%d is %f\n' , kvals(max_ind), classificationErrorRate(CYProb, CYTest));


