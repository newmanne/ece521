function [ RXTraining, RYTraining, RXValidation, RYValidation, RXTest, RYTest, CXTraining, CYTraining, CXValidation, CYValidation, CXTest, CYTest  ] = loadVars( )

    % Load the variables
    ClassificationX = load('ClassificationX.txt');
    ClassificationY = load('ClassificationY.txt');
    RegressionX = load('RegressionX.txt');
    RegressionY = load('RegressionY.txt');

    % Split into training, validation, and testing
    training_ind = 1:50;
    validation_ind = 51:100;
    test_ind = 101:200;

    CXTraining = ClassificationX(training_ind);
    CYTraining = ClassificationY(training_ind);
    RXTraining = RegressionX(training_ind);
    RYTraining = RegressionY(training_ind);

    CXValidation = ClassificationX(validation_ind);
    CYValidation = ClassificationY(validation_ind);
    RXValidation = RegressionX(validation_ind);
    RYValidation = RegressionY(validation_ind);

    CXTest = ClassificationX(test_ind);
    CYTest = ClassificationY(test_ind);
    RXTest = RegressionX(test_ind);
    RYTest = RegressionY(test_ind);

    % Normalize
    [RXTraining, mu, sigma] = normalize_features(RXTraining);
    RXValidation = normalizeFromMuSigma(RXValidation, mu, sigma);
    RXTest = normalizeFromMuSigma(RXTest, mu, sigma);
    
    [CXTraining, mu, sigma] = normalize_features(CXTraining);
    CXValidation = normalizeFromMuSigma(CXValidation, mu, sigma);
    CXTest = normalizeFromMuSigma(CXTest, mu, sigma);

end

