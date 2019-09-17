%% Functionality testing of NCP
% Checks that the code runs, but not if it arrives at the correct result,
% i.e. can the functions be called without getting compilation or runtime
% errors.
%
%
%
classdef test_functionality_CP < matlab.unittest.TestCase
    properties (TestParameter)
        
        %add_noise = {true}%,false}
        create_missing = {false, true};
        inference_scheme = {'variational'};%, 'sampling'};
        factor_distribution = {{'normal','normal'},{'normal', 'normal', 'normal'}};
        
        constraints_2way = {{'constant','scale'},...
            {'constant','ard'},...
            {'constant','wishart'},...
            {'scale','scale'},...
            {'scale','ard'},...
            {'scale','wishart'},...
            {'ard','wishart'},...
            {'ard','ard'},...
            {'wishart','wishart'}};
        %noise_scheme = {'homoscedastic', 'heteroscedastic'};
        noise_scheme = {[0,0], [1,1], [1,0], [0,1]};
        instance_number = num2cell(1:1);
        
    end
    
    methods (Test)
        
        function combinations_CP(testCase,...
                factor_distribution, inference_scheme,...
                constraints_2way, create_missing, noise_scheme, instance_number)
            constr = constraints_2way;
            distr = factor_distribution;
            
            Nx = length(factor_distribution);
            if length(distr) == 3
                constr{3} = 'ard';
                noise_scheme(3) = 0;
            end
            valid_constr = strcat(distr, repmat({' '},1,Nx), constr);
            
            Ndims = 50:-1:50-Nx+1;
            D = 5;
            
            % Generate data
            X = generateTensorData(Ndims,D,repmat([6],1,Nx));
            X = addTensorNoise(X,10); % SNR = 10 dB
            if create_missing
                idx = randperm(numel(X));
                X(idx(1:round(numel(X)*0.1))) = nan;
            end
            %X(1) = nan;
%             X=X/(nanvar(X(:)))^(1/Nx);
            
            [text,A,A2,L,elbo] = evalc(['VB_CP_ALS(X, D, valid_constr, ', ... % Required input
                '''maxiter'', 50, ''conv_crit'', 1e-8, ', ...
                '''model_tau'', true, ''model_lambda'', true, ', ...
                '''inference'', inference_scheme, ''noise'', noise_scheme);']);
%             [A,A2,L,elbo] = VB_CP_ALS(X, D, valid_constr, ... % Required input
%                            'maxiter', 100, 'conv_crit', 1e-8 , ...
%                            'model_tau', true, 'model_lambda', true, ...
%                            'inference', inference_scheme);
            
            % Check factors
            for i = 1:Nx
                assert(all(size(A{i}) == [Ndims(i), D]),'%i. Factor has wrong size',i)
            end
            
            
        end
    end
end
