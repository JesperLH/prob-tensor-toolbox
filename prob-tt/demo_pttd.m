%% Demostration of Probabilistic Tensor Train (PTT) Decomposition
clear;
addpath('./requiredfunctions/')
rng(3425331)

N = [12,11,10,9];  % Observations in each mode
D = [1,4,5,3,1];  % Size of the tensor train

% Generate tensor train
[data, ~, S_truth, V_truth] = generateTensorTrain(N,D);
% data = data/sqrt(var(data(:)));

% Add homoscedastic white noise, such that SNR=10dB
X = addTensorNoise(data, 10);

%% Estimate probabilistic tensor train
D_est = D;
%D_est(2:end-1) = D_est(2:end-1)*2;


[G_est, ES, EV, Etau, elbo] = tt_prob_tensor(X, [], D_est,...
    'conv_crit', 1e-6, ...  % Continue until change in ELBO is below 1e-6 .
    'maxiter', 50, ...  % Stop at convergence or after 100 iterations.
    'verbose', 'yes', ...  % Display output ('yes','no')
    'model_tau', true, ... % Should the noise precision (tau) be estimated?
    'fixed_tau', 0, ... % At what iteration should noise estimation begin? default: at the start (0).
    'tau_a0', 1e-6, 'tau_b0', 1e-6, ... % Hyperparameters for tau, default is broad priors.
    'model_lambda', true, ... Model the scale on S
    'fixed_lambda', 0, ...  
    'lambda_a0', 1e-6, 'lambda_b0', 1e-6, ... % Hyperparameters for lambda
    'model_S', true, 'fixed_S', 0, ...  % On/off modeling of S (for debugging)
    'model_V', true, 'fixed_V', 0, ...  % On/off modeling of V (for debugging)
    'model_G', true, 'fixed_G', 0 ...  % On/off modeling of G (for debugging)
    );
