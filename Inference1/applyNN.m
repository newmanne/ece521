function [ mse, YPred ] = applyNN( w, X, YActual )

    N = length(X);
    k = length(w) - 3;
    
    xx = zeros(N, k + 3);
    xx(:, 1) = 1;
    xx(:, 2) = X;

    g = zeros(N, k + 3);
    g(:, 1) = 1;
    g(:, 2) = X;
    
    for j = 3:k+2
        xx(:, j) = g * w(:, j);
        g(:, j) = 1./(1 + exp(-xx(:, j)));
    end
    xx(:, k+3) = g * w(:, k+3);
    YPred = xx(:, k+3);
    mse = meanSquaredError(YPred, YActual);
    
end
