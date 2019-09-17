classdef HyperParameterConstant < HyperParameterInterface
    % Public properties
    properties (Constant)
        valid_prior_properties='Constant';
        supported_inference_methods=[];
        inference_default=[];
    end
    
    methods
        function obj = HyperParameterConstant(factorsize, value)
            obj.factorsize = factorsize;
            obj.prior_value= value;
            obj.prior_log_value= log(value);
            obj.prior_property='Constant';
            obj.prior_size = [1,1];
            
        end
        
        function val = getExpFirstMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if nargin == 1
                val = self.prior_value;
            else
                if all(self.prior_size == factorsize)
                    val = self.prior_value;
                elseif length(self.prior_size) == 2 && self.prior_size(2) == factorsize(2)
                    val = repmat(self.prior_value, factorsize(1), 1);
                elseif self.prior_size == 1
                    val = repmat(self.prior_value, factorsize);
                else
                    error('Unknown error')
                end
            end
        end
        
        function val = getExpLogMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if nargin == 1
                val = self.prior_log_value;
            else
                if all(self.prior_size == factorsize)
                    val = self.prior_log_value;
                elseif length(self.prior_size) == 2 && self.prior_size(2) == factorsize(2)
                    val = repmat(self.prior_log_value, factorsize(1), 1);
                elseif self.prior_size == 1
                    val = repmat(self.prior_log_value, factorsize);
                else
                    error('Unknown error')
                end
            end
        end
        
        function updatePrior(self, varargin)
            %Pass
        end
        
        function cost = calcCost(self)
            cost = 0; % TODO: Cost is constant, but not zero?
        end
    end
    
end