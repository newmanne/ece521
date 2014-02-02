function [ w, error ] = trainNN( k, learningRate, nIters, XTraining, YTraining )

    error = zeros(nIters, 1);
    N = length(XTraining);

    w = zeros(k + 3, k + 3);
    w(1, 3:k+3) = rand(1, length(3:k+3));
    w(2, 3:k+2) = rand(1, length(3:k+2));
    w(3:k+2, k+3) = rand(length(3:k+2), 1);

    xx = zeros(N, k + 3);
    xx(:, 1) = 1;
    xx(:, 2) = XTraining;

    g = zeros(N, k+3);
    g(:, 1) = 1;
    g(:, 2) = XTraining;

    dedx = zeros(N, k+3);

    for i = 1:nIters
        for j = 3:k+2
            xx(:, j) = g * w(:, j);
            g(:, j) = 1./(1 + exp(-xx(:, j)));
        end
        xx(:, k+3) = g * w(:, k+3);
        g(:, k+3) = xx(:, k+3);
        error(i) = meanSquaredError(g(:,k+3), YTraining);
        dedx(:, k+3) = 2 * (g(:,k+3) - YTraining);
        for m = k+2:-1:3
            dedx(:, m) = dedx(:, m+1:k+3) * w(m, m+1:k+3)' .* g(:, m) .* (1 - g(:,m));
        end
        del = g' * dedx;
        w = w - learningRate * del .* (w ~= 0);
    %     if (mod(it, 1000) == 0)
    %         plot(XTraining, YTraining, 'bo', XTraining, g(:, k+3), 'g+');
    %         pause(0.5);
    %     end
    end

end

