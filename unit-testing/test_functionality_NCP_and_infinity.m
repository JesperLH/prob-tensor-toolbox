%% Functionality testing of NCP
% Checks that the code runs, but not if it arrives at the correct result,
% i.e. can the functions be called without getting compilation or runtime
% errors.
%
%
%
classdef test_functionality_NCP_and_infinity < matlab.unittest.TestCase
    properties (TestParameter)
        
        %add_noise = {true}%,false}
        create_missing = {false, true};
        inference_scheme = {'variational', 'sampling'};
        factor_distribution = {{'nonneg ','nonneg '},...
            {'nonneg exponential ','nonneg exponential '},...
            {'nonneg ','nonneg exponential '}};
        constraints_2way = {{'constant','scale'},... 1
            {'constant','ard'},... 2
            {'constant','sparse'},... 3
            {'infinity','scale'},... 4
            {'infinity','ard'},... 5
            {'infinity','sparse'},... 6
            {'scale','scale'},... 7
            {'scale','ard'},... 8
            {'scale','sparse'},... 9
            {'ard','ard'},... 10
            {'ard','sparse'},... 11
            {'sparse','sparse'}}; % 12
        %noise_scheme = {'homoscedastic', 'heteroscedastic'};
        noise_scheme = {[0,0], [1,1], [1,0], [0,1]};
        instance_number = num2cell(1:1);
        
    end
    
    methods (Test)
        
        function modes2_same_distribution(testCase,...
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
            
            if all(strcmpi(constr, 'sparse'))
                Ndims = [25,25];
            else
                Ndims = [25,24];
            end
            D = 5;
%             fprintf('Settings are %s\n',strjoin(valid_constr,'\n'))
%             disp(noise_scheme)
%             disp(create_missing)
%             fprintf('\n')
            
            % Generate data
            X = generateTensorData(Ndims,D,[3,3]);
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
            assert(all(size(A{1}) == [Ndims(1), D]),'1. Factor has wrong size')
            assert(all(size(A{2}) == [Ndims(2), D]),'2. Factor has wrong size')
            
            for i = 1:ndims(X)
                if ~isempty(strfind(lower(valid_constr{i}), 'constant')) || ...
                        ~isempty(strfind(lower(valid_constr{i}), 'scale'))
                    
                    assert(numel(L{i}) == 1, 'Prior had wrong dimension')
                    
                elseif ~isempty(strfind(lower(valid_constr{i}), 'ard'))
                    assert(numel(L{i}) == D, 'Prior had wrong dimension')
                    
                elseif ~isempty(strfind(lower(valid_constr{i}), 'sparse'))
                    assert(numel(L{i}) == Ndims(i)*D, 'Prior had wrong dimension')
                    
                elseif ~isempty(strfind(lower(valid_constr{i}), 'infty'))
                    assert(L{i} == 0, 'Prior was incorrect.')
                    
                end
            end
            
            
        end
    end
end
