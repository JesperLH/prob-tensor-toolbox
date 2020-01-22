
classdef UniformFactorMatrix < TruncatedNormalFactorMatrix
    % UniformFactorMatrix
    %   Detailed explanation goes here
    
    %% Required functions
    methods
        function obj = UniformFactorMatrix(...
                shape, lowerbound, upperbound, has_missing,...
                inference_method)
            if ~exist('inference_method','var')
                inference_method = 'variational';
            end
            
            % Call base constructer
            obj = obj@TruncatedNormalFactorMatrix(...
                shape, HyperParameterNone(shape), ...
                has_missing,inference_method);
            
            if ~isempty(lowerbound) && ~isempty(upperbound)
                obj.lowerbound = lowerbound;
                obj.upperbound = upperbound;
            else
                % Use default [-1,1], aka infinity norm constraint.
                obj.lowerbound = -1;
                obj.upperbound = 1;
            end
            
            obj.distribution = sprintf('Elementwise Uniform(%f,%f) Distribution',...
                obj.lowerbound, obj.upperbound);
            
            obj.prior_mean = [];
        end
        
        function updateFactorPrior(self, ~)
            % The Uniform Distribution has no optimizable parameters. This
            % function must be defined, otherwise the update from
            % TruncatedNormalFactorMatrix would be used.
        end
        
        function logp = getLogPrior(self)
            % Gets the prior contribution to the cost function.
            logp = -log(self.upperbound-self.lowerbound)*prod(self.factorsize);
        end
    end
    
    %% Class specific functions
end