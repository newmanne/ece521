function [ logLikelihood ] = logLikelihood( YProb, YActual )

T = length(YProb);

logLikelihood = (1 / T) .* sum(YActual .* log(YProb) + (1 - YActual) .* log(1 - YProb)); 


end

