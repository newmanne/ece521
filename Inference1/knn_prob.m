function [ YProb ] = knn_prob( XTest, XTraining, YTraining, N )

YProb = zeros(length(XTest), 1);
YProb(1:length(YProb)) = 0.1;
for i = 1:length(YProb)
    x_val = XTest(i);
    [~, min_indices] = sort((abs(XTraining - x_val)));
    nearestNeighbours = YTraining(min_indices(1:N));
    YProb(i) = YProb(i) + sum(nearestNeighbours == ones(length(nearestNeighbours), 1));
end

end

