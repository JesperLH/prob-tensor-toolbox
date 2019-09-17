classdef (Abstract) HyperParameterInterface < handle
    % Public properties
    properties
        prior_value;
        prior_log_value;
        prior_property;
        prior_size;
        factorsize;
        
        optimization_method;
        inference_method;
    end
    
    properties (Abstract, Constant)
        valid_prior_properties;
        supported_inference_methods;
        inference_default;
    end
    
    properties (Access = private)
        cost_contr = nan;
    end
    
    methods (Abstract)
        %%
        getExpFirstMoment(self, factorsize)
        getExpLogMoment(self, factorsize)
        updatePrior(self, eFact2)
        calcCost(self)
        
    end
    
    methods
        function set.prior_property(self, value)
            % Ensuring that only valid/supported prior properties can be
            % specified
            if any(strcmpi(value, self.valid_prior_properties))
                self.prior_property = value;
            else
                error('"%s" is not a valid/supported property.', value)
            end
        end
        
        function set.inference_method(self, s)
            if any(strcmpi(s, self.supported_inference_methods))
                self.inference_method = lower(s);
            else
                warning(['Inference method "%s" is not supported.',...
                    'Using default "%s" inference instead.'],...
                    s, self.inference_default)
                self.inference_method = self.inference_default;
            end
        end
    end
    
end