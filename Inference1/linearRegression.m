function [ error, MSEs, weights, YPred ] = linearRegression(Xin, YActual, learningRate,  numIters, XTrain, YTrain, errorFn )

    % Prepend a column of ones
    X = [ones(size(XTrain, 1), 1) XTrain];

    % Init the weights
    weights = randomBetween([1, size(X, 2)], [-0.1; 0.1]);

    T = size(XTrain, 1);
    MSEs = zeros(numIters, 1);
    
    for i = 1 : numIters
        % Compute the predictions
        YPred = X * weights';

        % Compute negative derivatives
        derivatives = (2/T) .* sum(bsxfun(@times, X, YTrain - YPred));

        % Update the weights
        weights = weights + learningRate .* derivatives;

        % TODO: momentum?

        MSEs(i) = meanSquaredError(YPred, YTrain);
    end

    YPred = applyWeights(Xin, weights);
    error = errorFn(YPred, YActual);
end

