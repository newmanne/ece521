function [ YPredictions ] = knn( XTest, XTraining, YTraining, N, isLogistic )

YPredictions = zeros(length(XTest), 1);
for i = 1:length(YPredictions)
    x_val = XTest(i);
    [~, min_indices] = sort((abs(XTraining - x_val)));
    nearestNeighbours = YTraining(min_indices(1:N));
    if isLogistic
        YPredictions(i) = mode(nearestNeighbours);
    else
        YPredictions(i) = mean(nearestNeighbours);
    end
end

end