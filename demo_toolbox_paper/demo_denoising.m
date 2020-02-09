clear; close all;
% Use Amino-Acid and Sugar data?
load ../Data/Amino_Acid_fluo/amino.mat;
X = reshape(X,DimX); D = 3; idx_samples=1:5;

 load ../Data/sugar_Process/data.mat
 X = reshape(X,DimX); D = 3; idx_samples=1:60:size(X,1);
 X=X(idx_samples,:,:);

%  X=X/sqrt(var(X(:)));

% - Show raw data, estimated factors?, and reconstructed (as in TT paper)
constr = {'normal ard','normal ard','normal ard'}; 
% type 'help VB_CP_ALS' to see additional constraints

% Run the CP model using variational inference
[A,A2,lambda, elbo, model] = VB_CP_ALS(X,D,constr,'maxiter',50,...
    'inference','variational');

X_recon_cp = nmodel(A);

%% TT 
% The mode order is (2,3,1) and model order D = (1,6,5,1) which where
% determined in the original probabilistic TT paper.
addpath(genpath('../prob-tt/'));
D_est = [1,6,5,1];
mode_order = [2,3,1]; rev_mode_order = [3,1,2];
[G_est, ES, EV, Etau, elbo] = tt_prob_tensor(permute(X,mode_order),...
    [], D_est,...
    'conv_crit', 1e-6, ...  % Continue until change in ELBO is below 1e-6 .
    'maxiter', 50, ...  % Stop at convergence or after 100 iterations.
    'verbose', 'yes');

X_recon_tt = permute(constructTensorTrain(G_est),rev_mode_order);

%% Tucker
D_est = [5,5,5];
[EG, EU, elbo] = VB_Tucker_BTD(X,D_est);
X_recon_tucker = nmodel(EU,EG);


%%
figure('Position',[50,100,1400,600])
visualizeDenoising(X, {X_recon_cp, X_recon_tucker, X_recon_tt},...
    {'Data','CP','Tucker','TT'},true)
save_currentfig('./demo_toolbox_paper/img/','denoising_cptuckertt')

