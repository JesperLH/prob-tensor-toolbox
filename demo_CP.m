% Demo of CP
addpath(genpath('./myclasses'))
addpath(genpath('./tools'))
addpath('./myMatlab/')
addpath('./tools/trandn/')
addpath('../thirdparty-matlab/nway331/') % Download from http://www.models.life.ku.dk/nwaytoolbox

Ndims = [50,49,48]; D = 5;
factor_distr = {'normal ard','nonneg sparse','normal constr'}; 
% factor_distr = {'nonneg expo sparse','nonneg ard','normal scale'}; 
% type 'help VB_CP_ALS' to see additional constraints

[X,A_true] = generateTensorData(Ndims,D,[6,3,6]);
X = addTensorNoise(X,10); % SNR = 10 dB

% Create missing values
% X(randi(numel(X),200,1)) = nan;
% X=X/sqrt(var(X(:)));

% Run the CP model using variational inference
[EA,EAtA,lambda, elbo, model, all_samples] = VB_CP_ALS(X,... % The data an N-way array
            D,... % Maximum number of components
            factor_distr,... % Factor distributions/constraints,
            'maxiter',50,...
            'Inference','variational',... %'variational' inference or Gibbs 'sampling'
            'conv_crit', 1e-8... % Stopping criteria, relative change in lowerbound
            );

% EA:           Contrains the expected first moment of the distributions.
% EAtA:         Contrains the expected second moment of the distributions.
% lambda:       Contains the expected first moment of the hyper-prior distributions (ard, sparsity, etc.).
% elbo:         Contains the value of the evidience lowerbound at each iteration (higher is better)
% model:        Distributions and lots of model info (currently mostly for debugging)
% all_samples:  Returns all samples when using gibbs sampling


%% Example visualization
figure; plot(EA{1}); 
xlabel('Observations'); ylabel('Expected value')
title('Expected value of fitted first mode')

% figure
% visualizeDecomposition(EA);
% suptitle('Expected value of each factor matrix')