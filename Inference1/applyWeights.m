function [ YPred ] = applyWeights( X, weights )

  X = [ones(size(X, 1), 1) X];
  YPred = sum(bsxfun(@times, X, weights), 2);

end

