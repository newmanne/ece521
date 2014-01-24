function [ ] = plotKnn( XTest, YPred, XTraining, YTraining, k )
figure
plot(XTest, YPred, '.g', XTraining, YTraining, 'xb');
title(sprintf('%d Nearest Neighbour', k));
legend('Test', 'Training');
xlabel('x');
ylabel('y');