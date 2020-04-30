%% Functionality testing of Tucker and BTD
% Checks that the code runs, but not if it arrives at the correct result,
% i.e. can the functions be called without getting compilation or runtime
% errors.
%
%
%
classdef test_functionality_Tucker < matlab.unittest.TestCase
    properties (TestParameter)
        
        create_missing = {false};%, true};
        initialization_scheme = {'sampling', 'nway', 'ml'};
        inference_scheme = {'variational', 'sampling'};
        factor_constraint = {{'orthogonal','orthogonal','orthogonal'},... % 1
            {'normal constr', 'orthogonal', 'orthogonal'},... % 2
            {'orthogonal', 'normal constr', 'orthogonal'},... % 3
            {'normal scale', 'normal constr',  'orthogonal'},... % 4
            {'normal ard', 'normal constr', 'orthogonal'},... % 5
            {'normal wishart', 'normal constr',  'orthogonal'},... % 6
            {'nonneg scale', 'nonneg constr', 'nonneg constr'},... % 7
            {'nonneg expo scale', 'nonneg expo constr', 'nonneg expo constr'},... % 8
             {'orthogonal', 'infinity', 'normal scale'}}; % 9
%         
        core_constraint = { 'scale', 'ard', 'sparse'}
        instance_number = num2cell(1:5);
        
    end
    
    methods (Test)
        
        function tucker_decomp(testCase,...
                initialization_scheme,factor_constraint, core_constraint, inference_scheme, ...
                create_missing, instance_number)
            fact_constr = factor_constraint;
            core_constr = core_constraint;
            
            Nx = length(fact_constr);
%             Ndims = 50:-1:50-Nx+1;
            Ndims = 30:-1:30-Nx+1;
            D = [6,5,4];
            
            gen_distr = [6,6,6];
            gen_distr(my_contains(fact_constr,'orthogonal','IgnoreCase',true)) = 7;
            gen_distr(my_contains(fact_constr,'nonneg','IgnoreCase',true)) = 4;
            [X, U, G] = generateTuckerData(Ndims,D,gen_distr); % Ortho. factors
            X = addTensorNoise(X,10);
            X=X/sqrt(var(X(:)));            
            
            if create_missing
                idx = randperm(numel(X));
                X(idx(1:round(numel(X)*0.1))) = nan;
            end
            
            [text,E_CORE,A,A2,L,elbo] = evalc(['pt_Tucker(X, D, fact_constr, core_constr, ', ... % Required input
                '''maxiter'', 50, ''conv_crit'', 1e-8, ', ...
                '''model_tau'', true, ''model_lambda'', true, ', ...
                '''inference'', inference_scheme,''init_method'',initialization_scheme);']);
            
            % Check factors
            for i = 1:Nx
                assert(all(size(A{i}) == [Ndims(i), D(i)]),'%i. Factor has wrong size',i)
            end
            % Check anything was actually learned
            X_recon = nmodel(A,E_CORE);
            
            assert(max(abs(X_recon(:)))>0, 'Reconstruction was zero!!') 
            
            assert(corr(X(:),X_recon(:))>0.2,'Reconstruction did not correlated with the data! (SNR=10dB, \rho=%4.2f)',corr(X(:),X_recon(:)))
            
        end
    end
end
