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
        getContribution2Prior(self);
        
        % Get stuff related to cost contribution
        getEntropy(self)
        getLogPrior(self)
        calcCost(self)
        
        % 
        getSamples(self,n)
        
    end
    
    methods
        
        function [E_MTTP, EPtP] = calcMTTPandSecondmoment(self, update_mode, ...
                Xm, Rm, eCore, eFact, eFact2, eFact2pairwise, eNoise)
            % Estimates,
            %   E_MTTP: denotes the Matrix-Times-Tensor-Product which is
            %       either the MTT-Khatri-Rao-Product or MTT-Kronecker-Product
            %       depending on whether a CP or Tucker model is fit.
            %   EPtP: The is the expected second moment of the factor matrix product,
            %       i.e. E[A'*A]. 
            
            ind=1:length(eFact);
            ind(update_mode) = [];
                
            if isempty(eCore) % CP model
                D = size(eFact{1},2);
                
                E_MTTP = eFact{ind(1)};
                for i = ind(2:end)
                    E_MTTP = krprod(eFact{i},E_MTTP);
                end
                E_MTTP = Xm*E_MTTP; 
                
                if nargout > 1
                    % Calculate second order moment
                    if self.data_has_missing
                        is_univariate_q = my_contains(self.distribution,...
                            {'exponential','uniform','truncated'},'IgnoreCase',true);
    %                     if is_univariate_q
    %                         if size(eFact2pairwise{update_mode},2) == D
    %                             % All factors have univariate Q-distribution
    %                             d_idx = 1:D; 
    %                         else
    %                             d_idx = D*( (1:D)-1)+(1:D);
    %                         end
    %                     else
                            d_idx = 1:size(eFact2pairwise{1},2);
    %                     end


                        if iscell(eNoise)
                            % Heteroscedastic noise
                            EPtP = bsxfun(@times, eFact2pairwise{ind(1)}(:,d_idx), eNoise{ind(1)});
                            for i = ind(2:end)
                                EPtP = krprod(bsxfun(@times, eFact2pairwise{i}(:,d_idx), eNoise{i}), EPtP);
                            end
                        else
                            % Homoscedastic noise
                            EPtP = eFact2pairwise{ind(1)}(:,d_idx);
                            for i = ind(2:end)
                                EPtP = krprod(eFact2pairwise{i}(:,d_idx), EPtP);
                            end
                        end

                        EPtP = Rm*EPtP;
                        if ~is_univariate_q
                            EPtP = permute(reshape(EPtP, self.factorsize(1), D, D), [2,3,1]);
                        end

                    else
                        % Full data (same for both homo- and heteroscedastic noise,
                        % as hetero noise is included in eFact2) 
                        EPtP = eFact2{ind(1)};
                        for i = ind(2:end)
                            EPtP = EPtP .* eFact2{i};
                        end
                    end
                end
            else % Tucker model
                
                %Xmkron
                X=unmatricizing(Xm,update_mode,cellfun(@(t) size(t,1),eFact)');
                cellfun(@(t) size(t,1),eFact)
                for j=ind
                    X=tmult(X,eFact{j}',j);            
                end
                E_MTTP=matricizing(X,update_mode);
                E_G_i=matricizing(eCore,update_mode);
                E_MTTP=(E_MTTP*E_G_i')*eNoise;

                
                if nargout > 1
                
                end
            end
            
        end
        
        function ci = getCredibilityInterval(self, usr_quantiles)
            % Get samples form the factor distribution
            sample_data = self.getSamples(1000); % TODO: Determine a suitable number..
            
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
                
            elseif (ismatrix(init) && ~isobject(init))...
                    && all(size(init) == self.factorsize)
                % includes only first moment
                self.factor = init;
                self.initialization = 'First Moment Provided by user';
                
% Including second moment is more cumbersome, as it requires redefining
%   getExpSecondMoment(). 
%             elseif iscell(init) && ismatrix(init{1}) && ismatrix(init{2})
%                 % includes both first and second moment
%                 if all(size(init{1}) == self.factorsize) ...
%                         && all(size(init{2}) == self.factorsize(2))
%                     self.factor = init{1};
%                 else
%                     error(['Incompatible initialization. Got sizes (%i,%i) ',...
%                         'and (%i, %i) but expected (%i,%i) and (%i,%i) for first and second moment.'],...
%                         size(init{1}), size(init{2}), self.factorsize, self.factorsize(2),self.factorsize(2))
%                     
%                 end
%                 
%                 self.initialization = 'First and Second Moment Provided by user';
                
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