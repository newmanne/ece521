function [ YPredictions ] = knn( XTest, XTraining, YTraining, k, type)

T = size(XTest, 1);
YPredictions = zeros(T, 1);

for i = 1:T
    x_val = XTest(i);
    for j = 1:size(XTraining, 1)
        [~, min_indices] = sort((abs(XTraining - x_val)));
        nearestNeighbours = YTraining(min_indices(1:k));
        if type == 0
            YPredictions(i) = mode(nearestNeighbours);
        elseif type == 1
            YPredictions(i) = mean(nearestNeighbours);
        end
    end
end

end