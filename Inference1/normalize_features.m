function [ normalized_features, mu, sigma ] = normalize_features( features )

    mu = mean(features);
    sigma = std(features);
    normalized_features = normalizeFromMuSigma(features, mu, sigma);

end

