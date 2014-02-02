[ ~, ~, ~, ~, ~, ~, CXTraining, CYTraining, CXValidation, CYValidation, CXTest, CYTest  ] = loadVars( );

%%
% g) Neural network classifiers: Real-valued outputs. As with linear regression, we can pretend that 
% the labels 0 and 1 are real numbers and train a neural network that has a real-valued output. 
% Repeat parts (1e) to (1h) from Question 1, using the classification data instead of the regression 
% data. Wherever it is requested that you report the MSE, report the MSE and also report the 
% classification error rate, by applying a threshold of 0.5 to output of the neural network. Compare 
% the error rates to those reported in (2a) to (2f). 

% 1e
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
    [w, error] = trainNN(k, learningRates(i), nIters, CXTraining, CYTraining);       
    semilogx(1:nIters, error, 'color', cc(i, :));
end
legend(cellfun(@num2str,num2cell(learningRates),'uniformoutput',0));
hold off

% select lr = 0.001
learningRate = 0.001;
[w, ~] = trainNN(k, learningRate, nIters, CXTraining, CYTraining);       
[error, YPred] = applyNN(w, CXTest, CYTest);
YPred = YPred > 0.5;
classificationError = classificationErrorRate(YPred, CYTest);
fprintf('MSE %f Classification Error %f\n', error(end), classificationError);

%%
% h) Neural network classifiers: Probabilistic outputs. Instead of a linear output unit, we can use a 
% logistic output unit so that the output is a real number between 0 and 1. This results in a neural 
% network version of logistic regression (see 2f). Implement a steepest descent method that 
% maximizes the log-likelihood of the training data. Repeat parts (1e) to (1h) from Question 1, 
% reporting log-likelihood values instead of MSEs and applying a threshold of 0.5 to the predicted 
% probabilities so that you can report classification error rates. Compare the error rates to those 
% reported in (2a) to (2g).