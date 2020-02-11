function [train_err, test_err] = demo_missing_analysis(...
    data,D_est,range_miss_fraction,data_scenario)

if nargin < 1
    % Generate synthetic data
    Ns = [50,49,48];
    D_est = 5;
    gen_constr = [6,6,6];
    [data,~] = generateTensorData(Ns,D_est,gen_constr);
    data = addTensorNoise(data,0); % SNR=10 dB
    range_miss_fraction = 0:0.3:0.9;
end

if nargin < 4
    data_scenario=[];
end

miss_schemes = {'Missing Elements', 'Missing Fibers'};
model_schemes = {'Variational', 'Sampling'};

algo_inputs = {'maxiter',500,'conv_crit',1e-7};
algo_constr = {'normal ard', 'normal ard', 'normal ard'};

% Res
train_err = nan(length(range_miss_fraction),length(miss_schemes),...
    length(model_schemes),2);
test_err = nan(size(train_err));

% Analyse
for i_sc = 1:length(miss_schemes)
    for i_miss = 1:length(range_miss_fraction)
        if contains(miss_schemes{i_sc},'element','IgnoreCase',true)
            idx_perm = randperm(numel(data));
            idx_test = idx_perm(1:round(numel(data)*range_miss_fraction(i_miss)));
            idx_train = idx_perm(round(numel(data)*range_miss_fraction(i_miss))+1:numel(data));
            X_train = data; X_train(idx_test)=nan;
            X_test = data; X_test(idx_train)=nan;
            
        elseif contains(miss_schemes{i_sc},'fibers','IgnoreCase',true)
            if ~isempty(data_scenario)
                Nx = size(data);
                if strcmpi(data_scenario,'allendata')
                    % Randomly missing geneset for some area and subject
                    [X_train, X_test] = randomlyMissingFibers(data,range_miss_fraction(i_miss),1);
                elseif strcmpi(data_scenario,'eegdata')
                    % Randomly missing time for some channel and trial
                    [X_train, X_test] = randomlyMissingFibers(data,range_miss_fraction(i_miss),2);
                end
                
            else
                % Fibers at random
                [X_train, X_test] = randomlyMissingFibers(data,range_miss_fraction(i_miss));
            end
        end
        
        if ~isempty(X_train)
            for i_mod = 1:length(model_schemes)
                [train_err(i_miss, i_sc, i_mod,:), test_err(i_miss, i_sc, i_mod,:)] = ...
                    fitAndEvaluate(model_schemes{i_mod},X_train, X_test, D_est,...
                    algo_constr, algo_inputs);
            end
        else
            % Default to nan value.
        end
    end
end

if nargin < 1
    keyboard
    %%
    for i_sc = 1:length(miss_schemes)
        
        figure;
        l_color = [1,0,0; 0,0,1];
        for i_mod = 1:length(model_schemes)
            hold on;
            plot(range_miss_fraction,train_err(:,i_sc,i_mod,1),'Color',l_color(i_mod,:))
            plot(range_miss_fraction,train_err(:,i_sc,i_mod,2),'--','Color',l_color(i_mod,:))
        end
        
        legend({'VB','VB test','Samp','Samp test'})
        title(miss_schemes{i_sc})
    end    
end

end

function [train_error, test_error] = fitAndEvaluate(model_scheme,...
    X_train, X_test, D, constraints, algo_inputs)

if contains(model_scheme,'sampling','IgnoreCase',true)
    algo_inputs(end+1:end+2) = {'inference', 'sampling'};
end

% Fit model
[A,~, lambda, lb, model ,samples]=VB_CP_ALS(X_train,D,constraints,algo_inputs{:});

idx_train = ~isnan(X_train(:));
idx_test = ~isnan(X_test(:));

if contains(model_scheme,'variational','IgnoreCase',true)
    X_recon = nmodel(A);
    sd_vb = sqrt(1./model.noise.getExpFirstMoment());
    
    train_error = [norm(X_train(idx_train)-X_recon(idx_train))...
        /norm(X_train(idx_train)), sd_vb];
    
    test_error = [norm(X_test(idx_test)-X_recon(idx_test))...
        /norm(X_test(idx_test)), sd_vb];
    
elseif contains(model_scheme,'sampling','IgnoreCase',true)
    error_gibbs = zeros(size(samples{1},3),1);
    M_burnin = size(samples{1},3)/2;
    for i = M_burnin+1:size(samples{1},3)
        X_recon = nmodel({samples{1}(:,:,i),samples{2}(:,:,i),...
            samples{3}(:,:,i)});
        
        error_gibbs(i,1) = norm(X_train(idx_train)-X_recon(idx_train))...
            /norm(X_train(idx_train));
        
        if any(idx_test)
            error_gibbs(i,2) = norm(X_test(idx_test)-X_recon(idx_test))...
                /norm(X_test(idx_test));
        end
    end
    error_gibbs(1:M_burnin,:) = [];
    train_error = [mean(error_gibbs(:,1)), sqrt(var(error_gibbs(:,1)))];
    if any(idx_test)
        test_error = [mean(error_gibbs(:,2)), sqrt(var(error_gibbs(:,2)))];
    else
        test_error = nan(1,2);
    end
end

end

