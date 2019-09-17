classdef MultivariateNormalFactorMatrix < FactorMatrixInterface
    %MULTIVARIATENORMALFACTORMATRIX Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        supported_inference_methods = {'variational'};
        inference_default='variational';
    end
    
    % Attributes that are only relevant to this class and subclasses
    properties (Access = protected)
        observations_share_prior;
        debug=true;
        covariance
    end
    
    methods
        
        function obj = MultivariateNormalFactorMatrix(...
                shape, prior_choice, has_missing, inference)
            
            obj.hyperparameter = prior_choice;
            
            obj.observations_share_prior = true;
            obj.factorsize = shape;
            obj.data_has_missing = has_missing;
           
            obj.distribution = 'Multivariate Normal Distribution';
            assert(strcmpi(inference,obj.inference_default), ' Specified inference procedure is available (%s)', inference)
            obj.inference_method = obj.inference_default;
            
            
        end
        
        function updateFactor(self, update_mode, Xm, Rm, eFact, eFact2, eFact2pairwise, eNoise)
            
            ind=1:length(eFact);
            ind(update_mode) = [];
            D = size(eFact{1},2);

            % Calculate sufficient statistics
            kr = eFact{ind(1)};
            for i = ind(2:end)
                kr = krprod(eFact{i},kr);
            end
            Xmkr = Xm*kr;

            % Expected second moment of the product of the other modes.
            if self.data_has_missing
                if iscell(eNoise)
                    % Heteroscedastic noise
                    ekrkr = bsxfun(@times, eFact2pairwise{ind(1)}, eNoise{ind(1)});
                    for i = ind(2:end)
                        ekrkr = krprod(bsxfun(@times, eFact2pairwise{i}, eNoise{i}), ekrkr);
                    end
                else
                    % Homoscedastic noise
                    ekrkr = eFact2pairwise{ind(1)};
                    for i = ind(2:end)
                        ekrkr = krprod(eFact2pairwise{i}, ekrkr);
                    end
                end
                
                ekrkr = Rm*ekrkr;
                ekrkr = permute(reshape(ekrkr, self.factorsize(1), D, D), [2,3,1]);

            else
                % Full data (same for both homo- and heteroscedastic noise,
                % as hetero noise is included in eFact2) 
                ekrkr = ones(D,D);
                for i = ind
                    ekrkr = ekrkr .* eFact2{i};
                end
            end
            
            % Update factor mean and covariance
            if self.observations_share_prior
                % Get prior influence
                ePrior = self.hyperparameter.getExpFirstMoment();
                if isscalar(ePrior) % Constr or scale
                    ePrior = ePrior*eye(D);
                elseif isvector(ePrior)
                    ePrior = diag(ePrior);
                elseif ismatrix(ePrior)
                    ePrior = ePrior;
                else
                    error('The first moment of the prior has an unexpected form.')
                end
                
                % Update using Variational Bayesian Inference
                if iscell(eNoise)
                    self.covariance = my_pagefun(@inv,...
                            bsxfun(@times, ekrkr, permute(eNoise{update_mode},[2,3,1]))...
                            +ePrior);
                    self.covariance = my_pagefun(@symmetrizing,self.covariance);

                    self.factor = bsxfun(@times, squeeze(permute(my_pagefun(...
                            @mtimes, self.covariance, permute(Xmkr,[2,3,1])), [3,1,2])),...
                            eNoise{update_mode});
                else
                    self.covariance = my_pagefun(@inv,...
                            eNoise*ekrkr+ePrior);
                    
                    %self.covariance = symmetrizing(self.covariance, self.debug);
                    self.covariance = my_pagefun(@symmetrizing,self.covariance);
                    
                    self.factor = eNoise*squeeze(permute(my_pagefun(...
                        @mtimes, self.covariance, permute(Xmkr,[2,3,1])), ...
                        [3,1,2]));
                end
                % TODO: Update using sampling
            else
                error('not implemented')
            end

        end       
        
        function updateFactorPrior(self, eFact2, distr_constr)
            if strcmp(string(class(self.hyperparameter)),'GammaHyperParameter')
                if nargin < 3
                    distr_constr = [];
                end
                self.hyperparameter.updatePrior(eFact2, distr_constr);
            else
                self.hyperparameter.updatePrior(eFact2);
            end
        end
        
        function eFact = getExpFirstMoment(self, eNoise)
            if nargin == 1
                eFact = self.factor;
            elseif nargin == 2
                eFact = bsxfun(@times, self.factor, eNoise);
            end
        end
        
        function eFact2 = getExpSecondMoment(self, eNoise)
            if self.observations_share_prior && nargin == 1
                if ~self.data_has_missing && ismatrix(self.covariance)
                    eFact2 = self.covariance*self.factorsize(1)...
                        + symmetrizing(self.factor'*self.factor, self.debug);
                else
                    eFact2 = sum(self.covariance,3)...
                        + symmetrizing(self.factor'*self.factor, self.debug);
                end
            elseif nargin == 2
                eFact2 = sum(...
                          bsxfun(@times, self.covariance, permute(eNoise,[2,3,1]))... % Heteroscedastic noise
                          ,3)...
                        + symmetrizing(self.factor'*diag(eNoise)*self.factor, self.debug);
            else
                eFact2 = [];
            end
        end
        
        function eFact2elem = getExpSecondMomentElem(self, eNoise)
            if size(self.covariance,3) == 1
                eFact2elem = bsxfun(@plus,self.factor.^2, diag(self.covariance)');
            else
                [~,D,N] = size(self.covariance);
                eFact2elem = self.factor.^2 + ...
                    self.covariance(bsxfun(@plus,[1:D+1:D*D]',[0:N-1]*D*D))'; %#ok<NBRAK>
            end

            if nargin == 2
                eFact2elem = bsxfun(@times, eFact2elem, eNoise);
            end
        end
        
        function eFact2elempair = getExpElementwiseSecondMoment(self)
            [mu, covar] = computeUnfoldedMoments(self.factor, permute(self.covariance,[3,1,2]));
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
            if self.observations_share_prior
                
                if ~self.data_has_missing && ismatrix(self.covariance)
                    entropy = sum(log(diag(chol(self.covariance))))...
                        +self.factorsize(2)/2*(1+log(2*pi));
                    
                    entropy = self.factorsize(1)*entropy;
                else
                    entropy = 0;
                    for i = 1:self.factorsize(1)
                        entropy = entropy...
                            + sum(log(diag(chol(self.covariance(:,:,i)))))...
                            +self.factorsize(2)/2*(1+log(2*pi));
                    end
                end
                
                
            else
                entropy = nan;
            end
            
            assert(~any(isnan(entropy(:))),'Entropy was NaN')
            assert(~any(isinf(entropy(:))),'Entropy was Inf')
        end
        
        function logp = getLogPrior(self)
            % Gets the prior contribution to the cost function.
            % Note. The hyperparameter needs to be updated before calculating
            % this.
            
            if self.observations_share_prior
                
                ePrior = self.hyperparameter.getExpFirstMoment();
                logp = numel(self.factor)/2*log(2*pi);
                
                if isscalar(ePrior)
                    logp = logp + prod(self.factorsize)/2*self.hyperparameter.getExpLogMoment()...
                        - 1/2 * trace(self.getExpSecondMoment())*ePrior;
                    
                elseif isvector(ePrior)
                    logp = logp + self.factorsize(1)/2*sum(sum(self.hyperparameter.getExpLogMoment()))...
                        - 1/2 * diag(self.getExpSecondMoment())' * ePrior(:);
                    
                elseif ismatrix(ePrior)
                    logp = logp + self.factorsize(1)/2*sum(sum(self.hyperparameter.getExpLogMoment()))...
                        - 1/2 * sum(sum(self.getExpSecondMoment() .* ePrior));
                else
                    error('Unknown prior form.')
                end
            else
                logp=nan;
            end
            
            if strcmpi(self.inference_method,'sampling')
                logp = nan;
                warning('Check if log prior is calculated correctly when sampling.')
            end
        end
        
        function samples = getSamples(self, num_samples)
            
            if nargin < 2
                samples = mvnrnd(self.factor, self.covariance);
            else
                samples = zeros([self.factorsize, num_samples]);
                for i = 1:num_samples
                    samples(:,:,i) = mvnrnd(self.factor, self.covariance);
                end
            end
            
        end
                
        
    end
    
end

