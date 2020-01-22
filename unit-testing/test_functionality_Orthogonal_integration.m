%%
%
%
classdef test_functionality_Orthogonal_integration < matlab.unittest.TestCase
    properties (TestParameter)
        
        create_missing = {false};%, true};
        inference_scheme = {'variational'};%, 'sampling'};
        
        factor_distribution = {{'nonneg ', 'nonneg expo '},...
            {'nonneg exponential ', 'normal '},...
            {'normal ', 'normal'}};
        constraints = {{'constant','scale'},... 1
            {'constant','ard'},... 2
            {'constant','factor_spec'},... 3
            {'infinity','scale'},... 4
            {'infinity','ard'},... 5
            {'infinity','factor_spec'},... 6
            {'scale','scale'},... 7
            {'scale','ard'},... 8
            {'scale','factor_spec'},... 9
            {'ard','ard'},... 10
            {'ard','factor_spec'},... 11
            {'factor_spec','factor_spec'}}; % 12
        noise_scheme = {[0,0,0]};%, [0,0,1], [0,1,0], [0,1,1]};
        instance_number = num2cell(1:1);
        
        
    end
    
    methods (Test)
        
        function modes2_integration(testCase,...
                factor_distribution, inference_scheme,...
                constraints, create_missing, noise_scheme, instance_number)
            constr = {'',constraints{:}};
            distr = {'orthogonal' , factor_distribution{:}};
            % Check constraints, if infty, then remove distribution info
            if my_contains(lower(constr{2}),'infi')
                distr{2} = '';
            end
            
            for c = find(strcmpi(constr, 'factor_spec'))
                if my_contains(distr{c}, 'nonneg')
                    constr{c} = 'sparse';
                else
                    constr{c} = 'wishart';
                end
            end
            valid_constr = strcat(distr, {' ',' ',' '}, constr);
            
            if sum(strcmpi(constr, 'sparse')) == 2
                Ndims = [25,24,24];
            else
                Ndims = [25,24,23];
            end
            D = 5;
            
            % Generate data
            data_gen_constr = [7,6,6];
            for i = 2:3
                if my_contains(distr{i}, 'nonneg')
                    data_gen_constr(i) = 3;
                end
            end
            
            
            X = generateTensorData(Ndims,D,data_gen_constr);
            X = addTensorNoise(X,10); % SNR = 10 dB
            if create_missing
                idx = randperm(numel(X));
                X(idx(1:round(numel(X)*0.1))) = nan;
            end
            X = X/(var(X(:))^(1/ndims(X))); % Remove scale of the data.
            
            [text,A,A2,L,elbo] = evalc(['VB_CP_ALS(X, D, valid_constr, ', ... % Required input
                '''maxiter'', 50, ''conv_crit'', 1e-7, ', ...
                '''model_tau'', true, ''model_lambda'', true, ', ...
                '''inference'', inference_scheme, ''noise'', noise_scheme);']);
            %             [A,A2,L,elbo] = VB_CP_ALS(X, D, valid_constr, ... % Required input
            %                            'maxiter', 100, 'conv_crit', 1e-8 , ...
            %                            'model_tau', true, 'model_lambda', true, ...
            %                            'inference', inference_scheme);
            
            % Check factors
            for i = 1:ndims(X)
                assert(all(size(A{i}) == [Ndims(i), D]),'%i. Factor has wrong size',i)
            end
            
            
        end
        
    end
end
