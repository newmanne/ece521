function [ error, logLikelihoods, weights, YPred ] = logisticRegression(Xin, YActual, learningRate,  numIters, XTrain, YTrain, errorFn )

    % Prepend a column of ones
    X = [ones(size(XTrain, 1), 1) XTrain];

    % Init the weights
    weights = randomBetween([1, size(X, 2)], [-0.1; 0.1]);

    T = size(XTrain, 1);
    logLikelihoods = zeros(numIters, 1);
    
    for i = 1 : numIters
        % Compute the predictions
        YPred = 1./(1 + exp(-(X * weights')));

        % Compute derivatives
        derivatives = (1/T) .* sum(bsxfun(@times, X, YTrain - YPred));

        % Update the weights
        weights = weights + learningRate .* derivatives;

        % TODO: momentum?

        logLikelihoods(i) = logLikelihood(YPred, YTrain);
    end

    YPred = applyWeights(Xin, weights);
    error = errorFn(YPred, YActual);
end

