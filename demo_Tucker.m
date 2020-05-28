%% Demo Tucker and Block-Term Decomposition
clear
%% Tucker
N = [20,19,18];
D = [3,3,3];
factor_distr = {'normal const', 'nonneg const',  'normal const'};
factor_distr = {'infinity', 'orthogonal', 'nonneg expo constr'};
core_constraint = 'sparse'; % ('scale, 'ard', 'sparse')
% Note. While it is possible to model core array ARD (core_constraint='ard')
%       and ARD on one or more factors (e.g. 'normal ard', 'nonneg ard'). Such
%		an ARD is not shared between the core and factors. Instead, the factors
%       should be set to "constant" and the ARD be modelled only on the core array.

%% Simulate data
sim_constr = zeros(1,length(N));
sim_constr(contains(factor_distr,'nonneg')) = 3;
sim_constr(contains(factor_distr,'normal')) = 7;
sim_constr(contains(factor_distr,'ortho')) = 6;
snr_db = 10;
[X, U, G] = generateTuckerData(N,D,sim_constr); 
X = addTensorNoise(X,snr_db);
%X=X/sqrt(var(X(:)));
X=X/(var(X(:))^(1/ndims(X)));
%% Fit model
D_est = D;
[EG, EA, EAtA, ELambda, lowerbound, model, all_samples]=...
            pt_Tucker(X,... % The data an N-way array
            D_est,... % Maximum number of components in each mode
            factor_distr,... % Factor distributions/constraints,
            core_constraint,... % Always a normal core, but with a 'scale', 'ard', or 'sparse' prior
            'maxiter',50,...
            'Inference','variational',... %'variational' inference or Gibbs 'sampling 
            'conv_crit', 1e-8,... % Stopping criteria, relative change in lowerbound
            'init_method','sampling'... % Initial model parameters ('sampling', 'nway', 'ml-step','bayes-step) 
            );
% EA:         Contrains the expected first moment of the distributions.
% EAtA:       Contrains the expected second moment of the distributions.
% ELambda:    Contains the expected first moment of the hyper-prior distributions (ard, sparsity, etc.).
% lowerbound: Contains the value of the evidience lowerbound at each iteration (higher is better)
% model:      Distributions and lots of model info... (currently mostly for debugging)
% all_samples: Returns all samples if using gibbs sampling
        
X_recon = nmodel(EA,EG);
corr(X(:),X_recon(:))
% [coeffRV(U{1},EA{1}),coeffRV(U{2},EA{2}),coeffRV(U{3},EA{3})]

%% Visualize
figure;
visualizeDecomposition(EA,EG);