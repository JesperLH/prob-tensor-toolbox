%% Demonstration of Tensor Denoising
% - Requires the Amino Acid dataset (available from http://www.models.life.ku.dk/Amino_Acid_fluo)
setup_paths;

clear; close all;
num_random_start = 50;
rng(0,'twister')

% Amino-Acid dataset
load ../Data/Amino_Acid_fluo/amino.mat;
X = reshape(X,DimX); D = 3; idx_samples=1:5;
axis_labels = {'Intensity', 'Emission (nm)', 'Excitation (nm)'};
axis_ticks = {[], 250:450, 250:300};
 
scale_x = sqrt(var(X(:)));%^(1/ndims(X));
X=X/scale_x;
%% Candecomp/PARAFAC
constr = {'normal ard', 'normal ard', 'normal ard'};
% type 'help VB_CP_ALS' to see additional constraints

% Run the CP model using variational inference
A_best = []; elbo_best = -inf;
for i_rep = 1:num_random_start
    [A,A2,lambda, elbo, model] = VB_CP_ALS(X,D,constr,'maxiter',100,...
        'inference','variational');
    if elbo(end)>elbo_best
        A_best = A; elbo_best=elbo(end);
    end
end

X_recon_cp = nmodel(A_best);

%% TT 
% The mode order is (2,3,1) and model order D = (1,6,5,1) which where
% determined in the original probabilistic TT paper.
addpath(genpath('./prob-tt/'));
D_est = [1,6,5,1];
mode_order = [2,3,1]; rev_mode_order = [3,1,2];

G_best = []; elbo_best = -inf;
for i_rep = 1:num_random_start
    try
    [G_est, ES, EV, Etau, elbo] = tt_prob_tensor(permute(X,mode_order),...
        [], D_est,...
        'conv_crit', 1e-6, ...  % Continue until change in ELBO is below 1e-6 .
        'maxiter', 100, ...  % Stop at convergence or after 100 iterations.
        'verbose', 'no');
        if elbo(end)>elbo_best
            G_best = G_est; elbo_best=elbo(end);
        end
    catch
    end
end
X_recon_tt = permute(constructTensorTrain(G_best),rev_mode_order);

%% Tucker
% Estimate model order 
% D_max = min(10,size(X));
% noc_elbo = nan([D_max,num_random_start]); 
% t0 = tic;
% for d1 = 2:D_max(1)
%     for d2 = 2:D_max(2)
%         for d3 = 2:D_max(3)
%             for i_rep = 1:num_random_start
%                 try
%                 [~,~,~, elbo] = evalc('pt_Tucker_BTD(X,[d1,d2,d3]);');
%                 noc_elbo(d1,d2,d3,i_rep) = elbo(end);
%                 catch
%                    warning('Error for (%i,%i,%i)',d1,d2,d3)
%                 end
%             end
%             fprintf('Ran (%i,%i,%i)\n',d1,d2,d3)
%         end
%     end
% end
% toc(t0);
% % Determine model order with highest ELBO
% ll = nanmax(noc_elbo,[],4);
% [max_val, idx_max] = max(ll(:)); 
% [D1,D2,D3] = ind2sub(size(ll),idx_max);
% D_est = [D1,D2,D3]
D_est = [5,9,9];

% Fit model
G_best = []; U_best = []; elbo_best = -inf;
for i_rep = 1:num_random_start
    [EG, EU, elbo] = pt_Tucker_BTD(X,D_est);
    if elbo(end)>elbo_best
        G_best = EG; U_best = EU; elbo_best=elbo(end);
    end
end

X_recon_tucker = nmodel(U_best,G_best);


%% Visualize the reconstructed data and residual error.
figure('Position',[0,0,1000,1200])
visualizeDenoising(X*scale_x,...
    {X_recon_cp*scale_x, X_recon_tucker*scale_x, X_recon_tt*scale_x},...
    {'Data','CP','Tucker','TT'},true)
save_currentfig('./demo_toolbox_paper/img/','denoising_cptuckertt')

