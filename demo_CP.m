% Demo of CP
addpath(genpath('./myclasses'))
addpath(genpath('./tools'))
addpath('./myMatlab/')
addpath('../thirdparty-matlab/trandn/') % Download from http://www.models.life.ku.dk/nwaytoolbox

Ndims = [50,49]; D = 5;
constr = {'normal ard','nonneg sparse'}; 
% type 'help VB_CP_ALS' to see additional constraints

X = generateTensorData(Ndims,D,[6,3]);
X = addTensorNoise(X,10); % SNR = 10 dB

% Run the CP model using variational inference
[A,A2,lambda, elbo, model] = VB_CP_ALS(X,D,constr,'maxiter',50,...
    'inference','variational');
% A:         Contrains the expected first moment of the distributions.
% A2:        Contrains the expected second moment of the distributions.
% lambda:    Contains the expected first moment of the hyper-prior distributions (ard, sparsity, etc.).
% elbo:      Contains the value of the evidience lowerbound at each iteration (higher is better)
% model:     Distributions and lots of stuff... (currently mostly
%                   for debugging)

figure; plot(A{1}); 
xlabel('Observations'); ylabel('Expected value')
title('Expected value of fitted first mode')
