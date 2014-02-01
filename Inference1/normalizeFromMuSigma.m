function [ normalized_features ] = normalizeFromMuSigma( features, mu, sigma )

    normalized_features = bsxfun(@rdivide, bsxfun(@minus, features, mu), sigma);

end

