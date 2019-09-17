classdef OrthogonalFactorMatrix < FactorMatrixInterface
    %OrthogonalFactorMatrix Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        supported_inference_methods = {'variational'};
        inference_default='variational';
    end
    
    % Attributes that are only relevant to this class and subclasses
    properties (Access = protected)
        efactor
        entropy
        init_efactor = false;
    end
    
    methods
        
        function obj = OrthogonalFactorMatrix(...
                shape, inference)
            
            obj.distribution = 'von Mises-Fisher Matrix Distribution';
            obj.factorsize = shape;
            
            obj.inference_method = obj.inference_default;
            obj.hyperparameter = HyperParameterNone(shape);
            
        end
        
        
        
        function updateFactor(self, update_mode, Xm, Rm, eFact, ~, ~, eNoise)
                
                if ~isempty(Rm) && numel(Rm) ~= nnz(Rm)
                    error('Orthogonal factor matrix expects fully observed data!')
                end
                
                ind=1:length(eFact);
                ind(update_mode) = [];
                
                % Calculate sufficient statistics
                kr = eFact{ind(1)};
                for i = ind(2:end)
                    kr = krprod(eFact{i},kr);
                end
                
                Xmkr = Xm*kr;
                if ~iscell(eNoise)
                    self.factor = eNoise*Xmkr;
                else
                    self.factor = Xmkr;
                end
                
                
                N = size(Xmkr,1);
                [UU,SS,VV]=svd(self.factor, 'econ');
                [f,V,lF]=hyperg(N,diag(SS),3);
                assert(~isinf(lF))
                
                % Get expected value
                self.efactor = UU*diag(f)*VV';
                
                self.entropy = lF-sum(self.factor(:).*self.efactor(:));
                assert(isreal(self.entropy))
                
        end
        
        function updateFactorPrior(self, varargin)
            %pass, the prior is fixed to be uniform across the sphere.
        end
        
        function eFact = getExpFirstMoment(self, eNoise)
            if isempty(self.efactor)
                eFact = self.factor; %Initialization
                self.init_efactor = true;
            else
                eFact = self.efactor;
            end
            
            if nargin > 1
                assert(all(eNoise==1), 'Heteroscedastic noise is not allowed on the orthogonal factor')
            end
            
        end
        
        function eFact2 = getExpSecondMoment(self, eNoise)
            
            eFact2 = eye(size(self.factor,2)); 
            if nargin > 1
                assert(all(eNoise==1), 'Heteroscedastic noise is not allowed on the orthogonal factor')
            end
        end
        
        function eFact2elem = getExpSecondMomentElem(self, eNoise)
            eFact2elem = self.efactor.^2;
            if nargin > 1
                assert(all(eNoise==1), 'Heteroscedastic noise is not allowed on the orthogonal factor')
            end
        end
        
        function eFact2elempair = getExpElementwiseSecondMoment(self)
            [mu, covar] = computeUnfoldedMoments(self.efactor, zeros(size(self.factor,2)));
            eFact2elempair = mu+covar;
        end
        
        function cost = calcCost(self)
            if strcmpi(self.inference_method,'variational')
                cost = self.getLogPrior()+sum(sum(self.getEntropy()));
            elseif strcmpi(self.inference_method,'sampling')
                cost = self.getLogPrior();
            end
            
        end
        
        function entropy = getEntropy(self)
            %Entropy calculation
            entropy = self.entropy;
            
            assert(~any(isnan(entropy(:))),'Entropy was NaN')
            assert(~any(isinf(entropy(:))),'Entropy was Inf')
        end
        
        function logp = getLogPrior(self)
            % Note the logprior is only 0 when the prior is uniform on the
            % sphere and the volume of the sphere cancels out in logp and
            % entropy.... TODO!!
            logp = 0;
            
            if strcmpi(self.inference_method,'sampling')
                logp = nan;
                warning('Check if log prior is calculated correctly when sampling.')
            end
        end
        
        function samples = getSamples(num_samples)
            % Simulation of the Matrix Bingham–vonMises–Fisher Distribution,
            % With Applications toMultivariate and Relational Data
            error('Not implemented!!')
        end
        
    end
    
end

