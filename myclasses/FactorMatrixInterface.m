% classdef (Attributes) ClassName
%    properties (Attributes)
%       PropertyName
%    end
%    methods (Attributes)
%       function obj = methodName(obj,arg2,...)
%          ...
%       end
%    end
%    events (Attributes)
%       EventName
%    end
%    enumeration
%       EnumName
%    end
% end
classdef (Abstract) FactorMatrixInterface < handle
    % Public properties
    
    properties (Abstract, Constant)
        supported_inference_methods;
        inference_default;
    end
        
    properties
        hyperparameter; % Link to prior on the parameters
        factor;         % 
        factorsize;     % [#Observations, #Components]
        data_has_missing = false;
        
        inference_method;   % What type of inference was performed
        initialization;     % How was the factor initialized
        distribution;       % What type of distribution is on this factor
    end
    
    methods (Abstract)
        % Fitting methods
        updateFactor(self, update_mode, Xm, Rm, eFact, eFact2, eFact2elementwise, eNoise)
        updateFactorPrior(self, eContribution, distribution_constant)
        
        % Get expected moments
        getExpFirstMoment(self)
        getExpSecondMoment(self)
        getExpSecondMomentElem(self)        % Only for sparse and missing elements
        getExpElementwiseSecondMoment(self) % Only for missing elements and heteroscedastic noise
        
        % Get stuff related to cost contribution
        getEntropy(self)
        getLogPrior(self)
        calcCost(self)
        
        % 
        getSamples(self,n)
        
    end
    
    methods
        
        function ci = getCredibilityInterval(self, usr_quantiles)
            % Get samples form the factor distribution
            sample_data = self.getSamples(1000); % TODO: Determine a sutiable number..
            
            % Calculate the emperical quantiles of the distribution.
            ci = zeros([self.factorsize, length(usr_quantiles)]);
            for d = 1:self.factorsize(2)
                ci(:,d,:) = quantile(squeeze(sample_data(:,d,:))', usr_quantiles)';
            end
        end
        
        function initialize(self, init)
            if isa(init,'function_handle')
                self.factor = init(self.factorsize);
                self.initialization = func2str(init);
            elseif all(size(init) == self.factorsize)
                self.factor = init;
                self.initialization = 'Provided by user';
            else
                error('Not a valid initialization.')
            end
        end
        
        function summary = getSummary(self)
            summary = sprintf(['Factor matrix which follows a %s, '...
                '\n\t where the prior on its parameters follow ....'],...
                self.distribution);
        end
        
        function setOptimization(self,s)
            self.inference_method = lower(s);
        end
    end
    
    %% Check validation
    methods
        function set.factorsize(self, shape)
            if length(shape) ~= 2
                error('The size of the factor must be a 2D vector.')
            elseif any(shape <= 0)
                error('The size must be positive.')
            else
                self.factorsize = shape;
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