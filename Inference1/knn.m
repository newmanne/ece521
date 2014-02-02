function [ error, YPredictions ] = knn( X, Y, XTraining, YTraining, k, selector, errorFn)

T = size(X, 1);
YPredictions = zeros(T, 1);

for i = 1:T
    x_val = X(i);
    for j = 1:size(XTraining, 1)
        [~, min_indices] = sort((abs(XTraining - x_val)));
        nearestNeighbours = YTraining(min_indices(1:k));
        YPredictions(i) = selector(nearestNeighbours);
    end
end
error = errorFn(YPredictions, Y);

end