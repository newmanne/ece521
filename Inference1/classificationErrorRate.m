function [ errorRate ] = classificationErrorRate( YPred, YActual )

    errorRate = sum(YPred ~= YActual) / length(YPred);
    
end

