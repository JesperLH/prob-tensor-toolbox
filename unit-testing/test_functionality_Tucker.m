%% Functionality testing of Tucker decomposition
% Checks that the code runs, but not if it arrives at the correct result,
% i.e. can the functions be called without getting compilation or runtime
% errors.
%
%
%
classdef test_functionality_Tucker < matlab.unittest.TestCase
    properties (TestParameter)
        
        create_missing = {false};%, true};
        initialization_scheme = {'sampling'}%, 'nway', 'ml-step','bayes-step'};
        inference_scheme = {'variational'}%, 'sampling'};
        
        factor_constraint = {{'orthogonal','orthogonal','orthogonal'},... % 1
            {'orthogonal','normal constr', 'orthogonal'},... % 2
            {'normal constr', 'normal constr', 'normal constr'},... % 3
            {'nonneg constr', 'nonneg constr','nonneg constr'},... % 4
            {'nonneg expo constr', 'nonneg expo constr','nonneg expo constr'},... % 5
            {'normal constr', 'nonneg constr', 'nonneg expo constr'},... % 6
            {'normal constr', 'normal scale',  'normal constr'},... % 7
            {'normal constr', 'normal ard',  'normal constr'},... % 8
            {'normal constr', 'normal wishart',  'normal constr'},... % 9
            {'nonneg constr', 'nonneg scale','nonneg constr'},... % 10
            {'nonneg constr', 'nonneg ard','nonneg constr'},... % 11
            {'nonneg constr', 'nonneg sparse','nonneg constr'},... % 12
            {'nonneg constr', 'nonneg expo constr', 'normal constr'},... % 13
            {'orthogonal', 'infinity', 'nonneg expo scale'}}; % 14
        core_constraint = {'scale', 'ard', 'sparse'}
        instance_number = num2cell(1:1);
        % Possible settings for more extensive testing.
        snr_db={'10'}%,'-5','-10'};
        model_tau = {true}%, false}
        %model_lambda = {true, false}
        model_psi = {true}%, false}
    end
    
    methods (Test)
        
        function tucker_decomp(testCase,...
                initialization_scheme,factor_constraint, core_constraint, inference_scheme, ...
                snr_db, model_tau, model_psi, instance_number)
            fact_constr = factor_constraint;
            core_constr = core_constraint;
            
            Nx = length(fact_constr);
             Ndims = 50:-1:50-Nx+1;
%            Ndims = 30:-1:30-Nx+1;
            D = [6,5,4];
            
            gen_distr = [6,6,6];
            gen_distr(my_contains(fact_constr,'orthogonal','IgnoreCase',true)) = 7;
            gen_distr(my_contains(fact_constr,'nonneg','IgnoreCase',true)) = 4;
            [X, U, G] = generateTuckerData(Ndims,D,gen_distr); % Ortho. factors
%             X=X/sqrt(var(X(:)));
            ns_lvl = str2num(snr_db);
            X = addTensorNoise(X,ns_lvl);
            X=X/(var(X(:))^(1/ndims(X)));      
            
%             if create_missing
%                 idx = randperm(numel(X));
%                 X(idx(1:round(numel(X)*0.1))) = nan;
%             end
            convergence_threshold = 1e-8;
            
            [text,E_CORE,A,A2,L,elbo] = evalc(['pt_Tucker(X, D, fact_constr, core_constr,', ... % Required input
                '''maxiter'', 100, ''conv_crit'', convergence_threshold, ', ...
                '''model_tau'',model_tau, ''model_lambda'', true, ', '''model_psi'', model_psi, ',...'''model_tau'',false, ''model_lambda'', false, ', '''model_psi'', false, ',...
                '''inference'', inference_scheme,''init_method'',initialization_scheme);']);
            
            % Check factors
            for i = 1:Nx
                assert(all(size(A{i}) == [Ndims(i), D(i)]),'%i. Factor has wrong size',i)
            end
            % Check anything was actually learned
            X_recon = nmodel(A,E_CORE);
            
            if ns_lvl > 0
                assert(max(abs(X_recon(:)))>0, 'Reconstruction was zero!!') 

                assert(corr(X(:),X_recon(:))>0.2,'Reconstruction did not correlated with the data! (SNR=10dB, \rho=%4.2f)',corr(X(:),X_recon(:)))
            end
        end
    end
end
