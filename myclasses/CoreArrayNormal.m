classdef CoreArrayNormal < CoreArrayInterface
    %COREARRAYNORMAL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        supported_inference_methods = {'variational','sampling'};
        inference_default='variational';
        
    end
    
    properties
        sigma_covar=[];
        scale_hyperparam;
    end
    
    % Attributes that are only relevant to this class and subclasses
    properties (Access = protected)
        
        prior_mean=[];
        all_factors_are_orthogonal;
        prior_choice=[];
        model_beta_scale;
    end
    
    
    methods
        
        function obj = CoreArrayNormal(...
                shape, prior_choice, has_missing, inference, all_orthogonal)
            
            obj.hyperparameter = prior_choice;
            obj.prior_choice = lower(prior_choice);
            
            obj.coresize = shape;
            obj.data_has_missing = has_missing;
            obj.all_factors_are_orthogonal = all_orthogonal;
            
            obj.model_beta_scale = false; %(all_orthogonal && ~strcmpi(obj.prior_choice,'scale'))%...;
            %|| strcmpi(obj.prior_choice,'ard')
            
            obj.distribution = 'Multivariate Normal Distribution';
            assert(any(strcmpi(inference,obj.supported_inference_methods)),...
                ' Specified inference procedure is not available (%s)', inference)
            obj.inference_method = inference;
            
            obj.scale_hyperparam = struct('Evalue',1,'Elogvalue',0,...
                'alpha0',1, 'beta0', 1,...
                'est_alpha',1, 'est_beta', 1,...
                'logp',nan, 'entropy', nan);
            %obj.prior_mean = randn(shape);
        end
        
        function updateScaleHyperParam(self)
            EGsq = self.getExpSecondMoment();
            ePrior = self.hyperparameter.getExpFirstMoment();
            if isscalar(ePrior)
                ePrior = reshape(ones(prod(self.coresize),1)*ePrior, self.coresize);
            else
                ePrior = reshape(ePrior,self.coresize);
            end
            
            self.scale_hyperparam.est_alpha = self.scale_hyperparam.alpha0 + 0.5*numel(self.core);
            self.scale_hyperparam.est_beta = self.scale_hyperparam.beta0 + 0.5 * diag(EGsq)'*ePrior(:);
            assert(length(self.scale_hyperparam.est_beta) == 1, 'Scale was not a scalar...')
            self.scale_hyperparam.Evalue = self.scale_hyperparam.est_alpha/self.scale_hyperparam.est_beta;
            self.scale_hyperparam.Elogvalue = psi(self.scale_hyperparam.est_alpha)-log(self.scale_hyperparam.est_beta);
            
        end
        
        
        function updateCore(self, X, R, eFact, eFact2, eNoise)
            
            if self.model_beta_scale && ~isempty(self.core)
                 self.updateScaleHyperParam();
            end
            eBeta = self.scale_hyperparam.Evalue;
            
            krp = 1;
            for i=ndims(X):-1:1
                krp = kron(krp,eFact{i});
            end
            
            XU = krp'*X(:);
            if self.all_factors_are_orthogonal || isempty(eFact2)
                % All factor matrices are orthogonal
                ePrior = self.hyperparameter.getExpFirstMoment();
                if isscalar(ePrior)
                    ePrior = ones(prod(self.coresize),1)*ePrior;
                end
                
                self.sigma_covar = (eNoise +eBeta*ePrior).^-1;
                
                if ~iscell(eNoise)
                    % Homoscedastic noise
                    self.core = reshape(XU .* self.sigma_covar * eNoise, self.coresize);
                elseif iscell(eNoise)
                    % Heteroscedastic noise
                    self.core = nan; %XU .* self.sigma_covar; %? TODO? Should eFact contain noise estimate?
                else
                    self.core = nan;
                end
                self.sigma_covar = diag(self.sigma_covar(:));
                
            else
                eSigContri = 1;
                for i = ndims(X):-1:1
                    eSigContri = kron(eSigContri,eFact2{i});
                end              
                
                ePrior = self.hyperparameter.getExpFirstMoment();
                if isscalar(ePrior)
                    ePrior = eye(prod(self.coresize))*ePrior;
                else %if isvector(ePrior)
                    ePrior = diag(ePrior(:));
                end
                
                self.sigma_covar = (eNoise * eSigContri + eBeta*ePrior)\eye(prod(self.coresize));
                self.sigma_covar = (self.sigma_covar+permute(self.sigma_covar,[2,1,3]))/2;
                % Homoscedastic
                self.core = reshape((self.sigma_covar*eNoise)*XU,self.coresize);
                
            end
        end
        
        
        function updateCorePrior(self, eContribution, distr_constr)
            if nargin < 2
                eContribution = self.getExpSecondMoment();
                distr_constr = 0.5;
            end
            
            if self.model_beta_scale
             %    self.updateScaleHyperParam(); Should not be allowed to
             %    change without recaluclating  logp.
            end 
            

            eBeta = self.scale_hyperparam.Evalue;
            
            self.hyperparameter.updatePrior(eContribution*eBeta, distr_constr);
        end
        
        
        function [H_gamma, prior_gamma]=entropy_gamma(~,a,b,a0,b0)
            H_gamma=-log(b)+a-(a-1).*psi(a)+gammaln(a);
            prior_gamma=-gammaln(a0)+a0*log(b0)+(a0-1)*(psi(a)-log(b))-b0*a./b;
        end
        
        
        function eCore = getExpFirstMoment(self)
            eCore = self.core;
        end
        
        
        function eGtG = getExpSecondMoment(self)%, eNoise)
            eGtG = self.core(:)*self.core(:)' + self.sigma_covar;
        end
        
        
        function cost = calcCost(self)
            [entr_scale, logp_scale] = self.entropy_gamma(self.scale_hyperparam.est_alpha, self.scale_hyperparam.est_beta,...
                self.scale_hyperparam.alpha0, self.scale_hyperparam.beta0);
            if strcmpi(self.inference_method,'variational')
                cost = self.getLogPrior()+self.getEntropy();
                if self.model_beta_scale
                    cost = cost + entr_scale + logp_scale;
                end
            elseif strcmpi(self.inference_method,'sampling')
                cost = self.getLogPrior();
                if self.model_beta_scale
                    cost = cost + logp_scale;
                end
            end
            
        end
        
        function entropy = getEntropy(self)
            %Entropy calculation
            % multivariate (note cholesky decomp for numerical stability)
            entropy=sum(log(diag(chol(self.sigma_covar)))) ...
                + 0.5*prod(self.coresize)*(1+log(2*pi));
            
            assert(~any(isnan(entropy(:))),'Entropy was NaN')
            assert(~any(isinf(entropy(:))),'Entropy was Inf')
        end
        
        function logp = getLogPrior(self)
            % Gets the prior contribution to the cost function.
            % Note. The hyperparameter needs to be updated before calculating
            %             this.
            if strcmpi(self.hyperparameter.prior_property,'sparse')
                det_val = 1/2*sum(self.hyperparameter.getExpLogMoment());
            elseif strcmpi(self.hyperparameter.prior_property,'ard')
                
                M = length(self.coresize);
                val = zeros(M,1);
                invD = cumsum([0,self.coresize(end:-1:1)]); % Indexing
                for i=1:M
                    val(i) = sum(self.hyperparameter.prior_log_value(invD(i)+1:invD(i+1)));
                end
%                 det_val_old = n_obs_correction/2*sum(self.hyperparameter.getExpLogMoment());
                det_val = sum((prod(self.coresize)./self.coresize(end:-1:1)')/2.*val);

            elseif strcmpi(self.hyperparameter.prior_property,'scale')
                det_val = prod(self.coresize)/2*sum(self.hyperparameter.getExpLogMoment());
                
            end
            
            logp = prod(self.coresize)/2*(-log(2*pi)) ...   
                + det_val;
            
            if self.model_beta_scale
                logp = logp+prod(self.coresize)/2*self.scale_hyperparam.Elogvalue...
                    -0.5*sum(diag(self.getExpSecondMoment())...
                    .*self.hyperparameter.getExpFirstMoment()*self.scale_hyperparam.Evalue);
            else
                logp = logp ...
                    -0.5*sum(diag(self.getExpSecondMoment())...
                    .*self.hyperparameter.getExpFirstMoment());
            end
        end
        
        function samples = getSamples(self, num_samples)
            g = self.core(:);
            if numel(self.sigma_covar) == prod(self.coresize)
                S = diag(self.sigma_covar(:));
            else
                S = self.sigma_covar;
            end
            
            if nargin < 2
                samples = mvnrnd(g, S);
            else
                samples = zeros([prod(self.coresize), num_samples]);
                for i = 1:num_samples
                    samples(:,:,i) = mvnrnd(g, S);
                end
            end
            %
        end
        
        
    end
    
    
end

