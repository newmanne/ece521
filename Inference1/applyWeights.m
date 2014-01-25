function [ YPred ] = applyWeights( X, weights )

  X = [ones(size(X, 1), 1) X];
  YPred = X * weights';

end

