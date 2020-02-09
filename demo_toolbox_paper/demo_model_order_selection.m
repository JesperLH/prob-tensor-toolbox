% Demo: Model Order Selection

model_constraints = {'normal ard', 'normal ard', 'normal constr'};
algo_inputs = {'maxiter',70,'conv_crit',1e-8,'fixed_tau',0,'fixed_lambda',0};
%%

component_range = 1:2:10;
test_fraction = 0.2;
mos_scenarios = {...'ARD',...
    ...'Missing Elements', ...
    ...'Missing Fibers'}%,...
     'Heldout Samples'};

% Load data
% - Facescramble, average ERPs
X = generateTensorData([30,29,28],5,[7,7,7]); X = addTensorNoise(X,10);
[n_time, n_channel, n_subject] = size(X);

X=X/sqrt(var(X(:)));
%% Perform Analysis
train_error_mu = zeros(length(component_range), length(mos_scenarios),2);
train_error_sd = zeros(size(train_error_mu));
test_error_mu = zeros(size(train_error_mu));
test_error_sd = zeros(size(train_error_mu));

for sc = 1:length(mos_scenarios)
    
    %% Determine splitting into training and test set
    exists_testset = false;
    if contains(mos_scenarios(sc),'ARD')
        X_train = X;
        X_test = [];
    
    elseif contains(mos_scenarios(sc),'elements','IgnoreCase',true)
        train_idx = randperm(numel(X));
        test_idx = train_idx(1:floor(numel(X)*test_fraction));
        train_idx(1:floor(numel(X)*test_fraction)) = [];
        X_train = X; X_train(test_idx)=nan;
        X_test = X; X_test(train_idx)=nan;
        exists_testset = true;
        
    elseif contains(mos_scenarios(sc),'fibers','IgnoreCase',true)
        [X_train, X_test] = randomlyMissingFibers(X,test_fraction);
        exists_testset = true;
        
    elseif contains(mos_scenarios(sc),'samples','IgnoreCase',true)
        X_train = X(:,:,1:end-2);
        X_test = X(:,:,end-1:end); % TODO make this split sensible...
        
    end
    
    %%
    train_idx = find(~isnan(X_train(:)));
    test_idx = find(~isnan(X_test(:)));
    
    for d = 1:length(component_range)
        D_est = component_range(d);
        
        [A_vb,~, lambda_vb, lb_vb, model_vb ,~]=VB_CP_ALS(...
            X_train,D_est,model_constraints,'inference','variational',algo_inputs{:});
        [A_gibb,~, lambda_gibb, lb_gibb, model_gibb ,samples_gibb]=...
            VB_CP_ALS(X_train,D_est,model_constraints,'inference','sampling',algo_inputs{:});
        
        % Error on VB
        vb_recon = nmodel(A_vb);
        train_error_mu(d,sc,1) = (norm(X_train(train_idx)-vb_recon(train_idx))...
                                /norm(X_train(train_idx)) )^2;
        train_error_sd(d,sc,1) = sqrt(1/model_vb.noise.getExpFirstMoment());
               
        if exists_testset
            test_error_mu(d,sc,1) = (norm(X_test(test_idx)-vb_recon(test_idx))...
                                /norm(X_test(test_idx)) )^2;
            test_error_sd(d,sc,1) = sqrt(1/model_vb.noise.getExpFirstMoment());       
        
        elseif contains(mos_scenarios(sc),'samples','IgnoreCase',true)
            use_trained_factors = [1,1,0];
            use_trained_noise = true;
            [A, AtA, lambda, elbo, model] = VB_CP_Predict(...
            X_test, model_vb, model_constraints, use_trained_factors, use_trained_noise);
            
            train_error_mu(d,sc,1) = lb_vb(end);
            test_error_mu(d,sc,1) = elbo(end);
            1+1;
        end
        
        % Error sampling
        gibbs_err = zeros(size(samples_gibb{1},3),2);
        for i_samp = 1:size(samples_gibb{1},3)
            gibb_recon = nmodel({samples_gibb{1}(:,:,i_samp), samples_gibb{2}(:,:,i_samp), samples_gibb{3}(:,:,i_samp)});
            gibbs_err(i_samp,1) = (norm(X_train(train_idx)-gibb_recon(train_idx))...
                                /norm(X_train(train_idx)) )^2;
            if exists_testset
                gibbs_err(i_samp,2) = (norm(X_test(test_idx)-gibb_recon(test_idx))...
                    /norm(X_test(test_idx)) )^2;
            end
        end
        train_error_mu(d,sc,2) = mean(gibbs_err(:,1));
        train_error_sd(d,sc,2) = sqrt(var(gibbs_err(:,1)));
               
        if exists_testset
            test_error_mu(d,sc,2) = mean(gibbs_err(:,2));
            test_error_sd(d,sc,2) = sqrt(var(gibbs_err(:,2)));
        end
        %[train_error_mu(d,sc)]
        
    end
    1+1;
end
figure; plot(component_range, squeeze(train_error_mu(:,1,:))); legend({'VB','Gibbs'})
title('Train')
figure; plot(component_range, squeeze(test_error_mu(:,1,:))); legend({'VB','Gibbs'})
title('Test')
%% Visualize Results