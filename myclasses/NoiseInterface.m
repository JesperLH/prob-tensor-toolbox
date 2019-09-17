classdef NoiseInterface < handle
    
    
    properties
        number_of_elements;
        display = true;
        SST;
        inference_method;
        
    end
    
    properties (Abstract, Constant)
        supported_inference_methods;
        inference_default;
    end
    
    
    methods (Abstract)
        
        updateNoise(self)
        calcCost(self)
        
        getExpFirstMoment(self)
        getExpLogMoment(self)
    end
    
    methods
        function sst = getSST(self)
            sst = self.SST;
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