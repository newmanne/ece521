function [ weights, MSEs ] = linearRegression( learningRate,  numIters, XTrain, YTrain )

% Prepend a column of ones
X = [ones(length(XTrain), 1) XTrain];

% Init the weights
weightInitRange = [-0.1; 0.1];
weights = weightInitRange(1) + (weightInitRange(2) - weightInitRange(1)) .* rand(1, size(X, 2));

T = size(XTrain, 1);

MSEs = zeros(numIters, 1);
for i = 1 : numIters
    % Compute the predictions
    YPred = X * weights';
    
    % Compute negative derivatives
    derivatives = (2/T) .* sum(bsxfun(@times, X, YTrain - YPred));
        
    % Update the weights
    weights = weights + learningRate .* derivatives;
    
    MSEs(i) = meanSquaredError(YPred, YTrain);
end

end

