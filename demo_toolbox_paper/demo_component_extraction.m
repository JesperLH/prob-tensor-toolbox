% Demo on how to called CPD, constrained CPD,
clear
rng(0,'twister')
load ../Data/Amino_Acid_fluo/amino.mat;
X = reshape(X,DimX); D = 3; 
labels_mode = {'Samples','Emission','Excitation'};
xaxis_modes = {1:5, linspace(250,450,size(X,2)), linspace(250,300,size(X,3))};

X = X/sqrt(var(X(:)));
% 
%% Fit and visualize Candecomp/PARAFAC
D=3; 

% Normal factors
constr = {'normal ard', 'normal ard', 'normal ard'};
% constr = {'normal ard', 'normal scale', 'normal constr'};
% Truncated normal factors (non-negative)
% constr = {'nonneg ard', 'nonneg ard', 'nonneg constr'};
% Exponential factors (non-negative)
% constr = {'nonneg expo ard', 'nonneg expo ard', 'nonneg expo constr'};
% Mixed of factors (for illustration)
%constr = {'nonneg expo sparse', 'normal ard', 'orthogonal'};
A_best = []; max_elbo = -inf;
for i=1:5
    [A,A2,lambda, elbo, model] = pt_CP_ALS(X,D,constr,'maxiter',100,...
    'inference','variational');
    if elbo(end)>max_elbo
            A_best=A; max_elbo = elbo(end);
    end
end

figure('Position',[200,100,1000,350])
visualizeDecomposition(A_best,[],labels_mode)
suptitle('Visualization of Candecomp/PARAFAC')

%% Fit and visualize Tensor Train
addpath(genpath('./prob-tt/'));
D_est = [1,3,2,1]; % Low model order for visualization purposes
[G_est, ES, EV, Etau, elbo] = tt_prob_tensor(X, [], D_est,...
    'conv_crit', 1e-6, ...  % Continue until change in ELBO is below 1e-6 .
    'maxiter', 100, ...  % Stop at convergence or after 100 iterations.
    'verbose', 'yes');

figure('Position',[200,100,1100,500])
visualizeTensorTrain(G_est,labels_mode)
save_currentfig('./demo_toolbox_paper/img/','extract_tensortrain')
suptitle('Visualization of Tensor Train')

%% Fit and visualize Tucker decomposition
% Here with orthogonal factors, normal core, and elementwise sparsity on
% core elements.
D_est = [3,3,3];
[EG, EU, elbo] = pt_Tucker_BTD(X,D_est);
X_recon_tucker = nmodel(EU,EG);

figure('Position',[200,100,1000,400])
visualizeDecomposition(EU,EG,labels_mode)
save_currentfig('./demo_toolbox_paper/img/','extract_tucker')
suptitle('Visualization of Tucker')

%% Tucker-BTD (preliminary work)
D_est = 1*ones(3,3);
[EG, EU, elbo] = pt_Tucker_BTD(X,D_est);
X_recon_BTD = nmodel(EU,EG);

figure('Position',[200,100,1000,400])
visualizeDecomposition(EU,EG,{'Samples','Emission','Excitation'})
suptitle('Visualization of Tucker with Block-term structure')