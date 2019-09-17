%% Functionality testing of NCP
% Checks that the code runs, but not if it arrives at the correct result,
% i.e. can the functions be called without getting compilation or runtime
% errors.
%
%
%
classdef test_functionality_CP_and_NCP_integration < matlab.unittest.TestCase
    properties (TestParameter)
        
        create_missing = {false, true};
        inference_scheme = {'variational'};%, 'sampling'};
        factor_distribution = {{'nonneg ','normal '},...
            {'nonneg exponential ','normal '}};
        constraints_2way = {{'constant','scale'},... 1
            {'constant','ard'},... 2
            {'constant','wishart'},... 3
            {'infinity','scale'},... 4
            {'infinity','ard'},... 5
            {'infinity','wishart'},... 6
            {'scale','scale'},... 7
            {'scale','ard'},... 8
            {'scale','wishart'},... 9
            {'ard','ard'},... 10
            {'ard','wishart'},... 11
            {'sparse','constant'},... 12
            {'sparse','scale'},... 13
            {'sparse','ard'},... 14
            {'sparse','wishart'}}; % 15
        noise_scheme = {[0,0], [1,1], [1,0], [0,1]};
        instance_number = num2cell(1:1);
        
        
    end
    
    methods (Test)
        
        function modes2_integration(testCase,...
                factor_distribution, inference_scheme,...
                constraints_2way, create_missing, noise_scheme, instance_number)
            constr = constraints_2way;
            distr = factor_distribution;
            % Check constraints, if infty, then remove distribution info
            if my_contains(lower(constr{1}),'infi')
                distr{1} = '';
            elseif my_contains(lower(constr{2}),'infi')
                distr{2} = '';
            end
            
            valid_constr = strcat(distr, {' ',' '}, constr);
            Ndims = [25,24];
            D = 5;
            
            % Generate data
            X = generateTensorData(Ndims,D,[3,6]);
            X = addTensorNoise(X,10); % SNR = 10 dB
            if create_missing
                idx = randperm(numel(X));
                X(idx(1:round(numel(X)*0.1))) = nan;
            end
            
            [text,A,A2,L,elbo] = evalc(['VB_CP_ALS(X, D, valid_constr, ', ... % Required input
                '''maxiter'', 50, ''conv_crit'', 1e-8, ', ...
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
