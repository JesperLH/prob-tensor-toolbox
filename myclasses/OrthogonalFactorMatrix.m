classdef OrthogonalFactorMatrix < FactorMatrixInterface
    %OrthogonalFactorMatrix Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        supported_inference_methods = {'variational','sampling'};
        inference_default='variational';
    end
    
    % Attributes that are only relevant to this class and subclasses
    properties (Access = protected)
        efactor
        entropy
        init_efactor = false;
        num_updates = 0;
        log_volume_stiefel=[];
    end
    
    methods
        
        function obj = OrthogonalFactorMatrix(...
                shape, inference)
            
            obj.distribution = 'von Mises-Fisher Matrix Distribution';
            obj.factorsize = shape;
            
            if nargin < 2
                obj.inference_method = obj.inference_default;
            else
                obj.inference_method = inference;
            end
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
                
                if strcmpi(self.inference_method,'variational')
                    % Get expected value
                    if self.num_updates < 1 % Use MAP estimate for the first update. Note the entropy will be wrong..
                        self.efactor = UU*VV';
                        self.entropy = -1e+100;
                    else
                        self.efactor = UU*diag(f)*VV';
                        self.entropy = lF-sum(self.factor(:).*self.efactor(:));
                    end
                    self.num_updates = self.num_updates + 1;
                
                    assert(isreal(self.entropy))
                    
                 elseif strcmpi(self.inference_method,'sampling')
                    % Gibbs sampling matrix vMF
                     self.efactor = self.getSamples(1);                 
                end           
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
            [mu, covar] = computeUnfoldedMoments(self.efactor, []);
            eFact2elempair = mu+covar;
        end
        
        function eContr2Prior = getContribution2Prior(self)
            eContr2Prior = [];
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
            % entropy (e.g. in variational inference).
            logp = 0;
            
            if strcmpi(self.inference_method,'sampling')
                % Only the log(volume) is necessary for a uniform prior (F=0).
                if isempty(self.log_volume_stiefel)
                    I=self.factorsize(1); D=self.factorsize(2);
                    %volume = (2^D * pi^(I*D/2)) / ( pi^(D*(D-1)/4) * (prod(gamma(1/2*(I-(1:D) +1)))) ) ;
                    self.log_volume_stiefel = D*log(2) + I*D/2*log(pi) - D*(D-1)/4*log(pi) ...
                        - sum(gammaln(1/2*(I-(1:D) +1) ) );
                end
                logp= -self.log_volume_stiefel;
            end
        end
        
        function samples = getSamples(self,num_samples)
            % Based on
            % Hoff (2009): Simulation of the Matrix Bingham-von Mises-Fisher Distribution, With Applications to Multivariate and Relational Data
            % https://rdrr.io/cran/rstiefel/
            % TODO: Validate current implementation!
            
%             warning('Implementation not properly tested yet! Sampling may not work..')
            num_burnin = max(100, round(num_samples)*0.2);
            samples = sample_matrix_vonMises_Fisher(self.factor,num_burnin+num_samples);
            samples = samples(:,:,num_burnin+1:end);
        end
        
    end
    
end

