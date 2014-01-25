function [ confusionMatrix ] = confusionMatrix( YPred, YActual )

confusionMatrix = zeros(2);

zero_indices = find(YActual == 0);
one_indices = find(YActual == 1);

predicted_zero_indices = find(YPred == 0);
predicted_one_indices = find(YPred == 1);

confusionMatrix(1, 1) = sum(YActual(zero_indices) == YPred(zero_indices));
confusionMatrix(2, 2) = sum(YActual(one_indices) == YPred(one_indices));

confusionMatrix(1, 2) = sum(YPred(predicted_zero_indices) ~= YActual(predicted_zero_indices));
confusionMatrix(2, 1) = sum(YPred(predicted_one_indices) ~= YActual(predicted_one_indices));

end

