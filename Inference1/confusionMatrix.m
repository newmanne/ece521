function [ confusionMatrix ] = confusionMatrix( YPred, YActual )

confusionMatrix = zeros(2);

zero_indices = find(YActual == 0);
one_indices = find(YActual == 1);

confusionMatrix(1, 1) = sum(YActual(zero_indices) == YPred(zero_indices));
confusionMatrix(1, 2) = length(YActual) - confusionMatrix(1, 1);

confusionMatrix(2, 1) = sum(YActual(one_indices) == YPred(one_indices));
confusionMatrix(2, 1) = length(YActual) - confusionMatrix(2, 1);

end

