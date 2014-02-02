function [ XTrain, XValid, XTest ] = polyExpand( Xin )

    X = zeros(size(Xin, 1), 10);
    for k = 1:10
        X(:, k) = Xin .^ k;
    end

    XTrain = X(1:50, :);
    XValid = X(51:100, :);
    XTest = X(101:200, :);
    
    [XTrain, mu, sigma] = normalize_features(XTrain);
    XValid = normalizeFromMuSigma(XValid, mu, sigma);
    XTest = normalizeFromMuSigma(XTest, mu, sigma);

end

