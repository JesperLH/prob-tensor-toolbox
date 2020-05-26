classdef MultivariateNormalFactorMatrix < FactorMatrixInterface
    %MULTIVARIATENORMALFACTORMATRIX Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        supported_inference_methods = {'variational','sampling'};
        inference_default='variational';
        
    end
    
    % Attributes that are only relevant to this class and subclasses
    properties (Access = protected)
        observations_share_prior;
        debug=true;
        covariance;
        prior_mean=[];
    end
    
    methods
        
        function obj = MultivariateNormalFactorMatrix(...
                shape, prior_choice, has_missing, inference)
            
            obj.hyperparameter = prior_choice;
            
            obj.observations_share_prior = true;
            obj.factorsize = shape;
            obj.data_has_missing = has_missing;
           
            obj.distribution = 'Multivariate Normal Distribution';
            assert(any(strcmpi(inference,obj.supported_inference_methods)),...
                ' Specified inference procedure is not available (%s)', inference)
            obj.inference_method = inference;
            obj.covariance=eye(shape(2));
%             obj.prior_mean = randn(shape);            
        end
        
        function updateFactor(self, update_mode, Xm, Rm, eCore, covCore, eFact, eFact2, eFact2pairwise, eNoise)
            D = size(eFact{update_mode},2);
            
            % Calculate MTTKRP and Expected Second Moment
            [Xmkr, ekrkr] = self.calcMTTPandSecondmoment(update_mode, ...
                Xm, Rm, eCore, covCore, eFact, eFact2, eFact2pairwise, eNoise);
            
            % Update factor mean and covariance
            if self.observations_share_prior
                % Get prior influence
                ePrior = self.hyperparameter.getExpFirstMoment();
                if isscalar(ePrior) % Constr or scale
                    ePrior = ePrior*eye(D);
                elseif isvector(ePrior)
                    ePrior = diag(ePrior);
                elseif ismatrix(ePrior)
                    %pass, as ePrior = ePrior;
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
                    self.covariance = my_pagefun(@symmetrizing,self.covariance);

                    self.factor = eNoise*squeeze(permute(my_pagefun(...
                        @mtimes, self.covariance, permute(Xmkr,[2,3,1])), ...
                        [3,1,2]));
                end
                
                if ~isempty(self.prior_mean)
                    self.factor = self.factor + squeeze(my_pagefun(@mtimes,...
                        self.covariance, permute(self.prior_mean*ePrior,[2,3,1])))';
                end
                
            else
                error('not implemented')
            end
            
            % Gibbs sampling: Generate a single sample from the distribution.
            if strcmpi(self.inference_method,'sampling')
               self.factor = mvnrnd(self.factor,self.covariance);
               self.covariance = 0;
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
            if self.observations_share_prior
                
                % Always include the first moment or a single sample 
                if nargin == 1
                   eFact2 = symmetrizing(self.factor'*self.factor, self.debug); 
                elseif nargin == 2
                    eFact2 = symmetrizing(self.factor'*diag(eNoise)*self.factor, self.debug);
                else
                    eFact2 = [];
                end      
                
                % Include estimated covariance
                if strcmpi(self.inference_method,'variational')
                    if nargin == 1
                        if size(self.covariance,3)>1
                            eFact2 = eFact2 + sum(self.covariance,3);
                        else
                            eFact2 = eFact2 + self.covariance*self.factorsize(1);
                        end
                    elseif nargin == 2
                        eFact2 = eFact2 + sum(bsxfun(@times,... % Heteroscedastic noise
                            self.covariance, permute(eNoise,[2,3,1])),3);
                    end
                end
                          
            end

        end
        
        function eFact2elem = getExpSecondMomentElem(self, eNoise)
            % First moment or sample
            eFact2elem = self.factor.^2;
            
            % Include estimated covariance
            if strcmpi(self.inference_method,'variational')
                if size(self.covariance,3) == 1
                    eFact2elem = bsxfun(@plus,eFact2elem, diag(self.covariance)');
                else
                    [~,D,N] = size(self.covariance);
                    eFact2elem = eFact2elem + ...
                        self.covariance(bsxfun(@plus,[1:D+1:D*D]',[0:N-1]*D*D))'; %#ok<NBRAK>
                end
            end
            
            % Multiply by heteroscedastic noise
            if nargin == 2
                eFact2elem = bsxfun(@times, eFact2elem, eNoise);
            end
        end
        
        function eContr2Prior = getContribution2Prior(self)
            
            if strcmpi(self.hyperparameter.prior_property,'sparse')
                error('Not implemented')
            else
                eContr2Prior = self.getExpSecondMoment();
                if ~isempty(self.prior_mean)
                    eContr2Prior = eContr2Prior + self.prior_mean'*self.prior_mean...
                        - symmetrizing(self.factor'*self.prior_mean + self.prior_mean'*self.factor);
                    
                end
            end 
            
        end
        
        function eFact2elempair = getExpElementwiseSecondMoment(self)
            if strcmpi(self.inference_method,'variational')
                [mu, covar] = computeUnfoldedMoments(self.factor, permute(self.covariance,[3,1,2]));
            else
                [mu, covar] = computeUnfoldedMoments(self.factor, []);
            end
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
                logp = -numel(self.factor)/2*log(2*pi);
                
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
                
                if ~isempty(self.prior_mean)
                    if isvector(ePrior)
                        logp = logp -1/2*sum(sum(bsxfun(@times, ePrior , self.prior_mean.^2)))... % Here <mu.^2> is needed
                                +sum(sum(bsxfun(@times, ePrior , self.prior_mean.*self.getExpFirstMoment()))); % here only <mu>
                    else
                        logp = logp -1/2*sum(sum(bsxfun(@times, ePrior , self.prior_mean'*self.prior_mean)))... % Here <mu.^2> is needed
                                +sum(sum(bsxfun(@times, ePrior , self.prior_mean'*self.getExpFirstMoment()))); % here only <mu>
                    end
                end
                
            else
                logp=nan;
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

