function [ meanSquaredError ] = meanSquaredError(YPrediction, YMeasured)
    meanSquaredError = sum((YPrediction - YMeasured).^2) ./ length(YPrediction);
end

