%% Functionality testing of PARAFAC2 decomposition
% Checks that the code runs, but not if it arrives at the correct result,
% i.e. can the functions be called without getting compilation or runtime
% errors.
%
classdef test_functionality_PARAFAC2 < matlab.unittest.TestCase
    properties (TestParameter)
        
        create_missing = {false};%, true};
        initialization_scheme = {'random','probabilistic'};
        inference_scheme = {'variational'}%, 'sampling'};
        n_fixedmodes = {1,2}%, 3};
        
        factor_constraint = {{'orthogonal'},... % 1
            {'normal constr'},... % 2
            {'nonneg constr'},... % 3
            {'nonneg expo constr'},... % 4
            {'normal constr', 'nonneg constr', 'nonneg expo constr'},... % 5
            {'normal scale', 'normal constr',  'normal constr'},... % 6
            {'normal ard', 'normal constr',  'normal constr'},... % 7
            {'normal wishart', 'normal constr',  'normal constr'},... % 8
            {'nonneg scale', 'nonneg constr','nonneg constr'},... % 9
            {'nonneg ard', 'nonneg constr','nonneg constr'},... % 10
            {'nonneg sparse', 'nonneg constr','nonneg constr'},... % 11
            {'nonneg constr', 'nonneg expo constr', 'normal constr'},... % 12
            {'orthogonal', 'infinity', 'nonneg expo scale'},... % 13
            {'infinity'}}; % 14
        instance_number = num2cell(1:1);
        % Possible settings for more extensive testing.
        snr_db={'10','-5'}%,'-10'};'inf'}%
        model_tau = {true, false}
        model_lambda = {true, false}
        
    end
    
    methods (Test)
        
        function parafac2_decomp(testCase,...
                initialization_scheme,factor_constraint, inference_scheme, ...
                snr_db, model_tau, model_lambda, instance_number, n_fixedmodes)
            
            if length(factor_constraint) == 1
                fact_constr = repmat(factor_constraint,1,n_fixedmodes+2);
            else
                fact_constr = factor_constraint;
                if length(factor_constraint) ~= n_fixedmodes+2
%                    fact_constr{end+1} = fact_constr{end};
                    fact_constr = [fact_constr,repmat(fact_constr(end), 1, n_fixedmodes+2 - length(fact_constr))];
                end
            end
            
            % Generate data
            K=5;
            n_components = 4;
            
            max_n_obs = 15;
            n_obs = [max_n_obs:-1:(max_n_obs-n_fixedmodes+1),K];
            
            max_m_obs = 10;
            m_obs = max_m_obs-K+1:max_m_obs;
                        
            [X,A,P] = generateDataParafac2(n_obs, m_obs,n_components, fact_constr);
            
%             load('../../Data/Amino_Acid_fluo/amino.mat','X','DimX');
%             Y = permute(reshape(X,DimX),[2,3,1]);
%             load('../../Data/life_ku/Sugar_Process/sugar_Process/data.mat')
%             Y = reshape(X,DimX);
%             K=size(Y,3);
%             clear X;
%             for k = K:-1:1
%                 X{k} = Y(:,:,k);
%             end
            
            
%             for k = 1:K
%                 X{k} = X{k}/(var(X{k}(:))^(1/ndims(X{k})));
%             end
            
            % add noise
            ns_lvl = str2num(snr_db);
            if ~isinf(ns_lvl)
                for k = 1:K
                    X{k} = addTensorNoise(X{k},ns_lvl);
                end
            end
            
            %% Run model
            convergence_threshold = 1e-6;
            Dest = n_components;
            [text,EA,EP,elbo] = evalc(['pt_PARAFAC2(X, Dest, fact_constr,', ... % Required input
                '''maxiter'', 100, ''conv_crit'', convergence_threshold, ', ...
                '''model_tau'',model_tau, ''model_lambda'', model_lambda, ', ...
                '''inference'', inference_scheme,''init_method'',initialization_scheme);']);
            
            %% Check Solution
%           % TODO
            
            % Check anything was actually learned
            X_recon = nmodel_pf2(EA(1:end-2),EA(end-1),EA{end},EP);
            
            if ns_lvl > 0
                for k = 1:K
%                     assert(max(abs(X_recon{k}(:)))>0, 'Reconstruction was zero!!') 
% 
%                     assert(corr(X{k}(:),X_recon{k}(:))>0.1,'Reconstruction did not correlated with the data! (SNR=10dB, \rho=%4.2f)',corr(X{k}(:),X_recon{k}(:)))
                end
            end
        end
    end
end
