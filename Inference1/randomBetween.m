function [ matrix ] = randomBetween( rand_size, range )
    matrix = range(1) + (range(2) - range(1)) .* rand(rand_size(1), rand_size(2));
end

