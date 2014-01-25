function [ normalized_features ] = normalize( features, mu, sigma )

    normalized_features = bsxfun(@rdivide, bsxfun(@minus, features, mu), sigma);

end

