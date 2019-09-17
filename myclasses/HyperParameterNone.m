classdef HyperParameterNone < HyperParameterInterface
    % Public properties
    properties (Constant)
        valid_prior_properties='None';
        supported_inference_methods=[];
        inference_default=[];
    end
    
    methods
        function obj = HyperParameterNone(factorsize)
            obj.factorsize = factorsize;
            obj.prior_value=0;
            obj.prior_log_value=0;
            obj.prior_property='None';
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
            cost = 0;
        end
    end
    
end