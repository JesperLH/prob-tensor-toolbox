classdef GammaNoiseHeteroscedastic < NoiseInterface
    
    properties
        hp_alpha = 1e-4;
        hp_beta = 1e-4;
        
        
        SSE;
        noise_value;
        noise_log_value;
        has_missing_values=false;
        mode_idx;
    end
    
    properties (Constant)
        supported_inference_methods = {'variational','sampling'};
        inference_default = 'variational';
    end
    
    properties (Access = private)
        est_beta;
        est_alpha;
        isFactorUnivariate;
    end
    
    methods
        
        function obj = GammaNoiseHeteroscedastic(noise_mode, X, R, usr_hp_alpha, usr_hp_beta,...
                inference_method, isFactorUnivariate)
            
            if isempty(R)
                obj.number_of_elements = numel(X)/size(X,noise_mode); % Number of elements for each obs in noise_mode
            else
                obj.number_of_elements = sum(matricizing(R,noise_mode),2);
                obj.has_missing_values = ~(numel(X) == sum(obj.number_of_elements));
            end
            
            obj.SST = sum(matricizing(X, noise_mode).^2, 2);
            obj.mode_idx = noise_mode;
            
            
            if exist('usr_hp_beta','var') && ~isempty(usr_hp_alpha) && ~isempty(usr_hp_beta)
                obj.hp_alpha = usr_hp_alpha;
                obj.hp_beta = usr_hp_beta;
            end
            
            
            obj.est_alpha = repmat(obj.hp_alpha, size(X,noise_mode), 1);
            obj.est_beta = repmat(obj.hp_beta, size(X,noise_mode), 1);
            
            obj.noise_value = obj.est_alpha./obj.est_beta;
            obj.noise_log_value = psi(obj.est_alpha)-log(obj.est_beta);
            
            
            
            if exist('isFactorUnivariate','var')
                obj.isFactorUnivariate = isFactorUnivariate;
                assert(length(isFactorUnivariate) == ndims(X),'Must be specified for each factor');
            else
                obj.isFactorUnivariate = true(ndims(X),1);
            end
            
            
            if exist('inference_method','var')
                obj.inference_method = inference_method;
            else
                obj.inference_method = obj.inference_default;
            end
        end
        
        function updateSST(self, Xm, eNoise)
            ind = 1:length(eNoise);
            ind(self.mode_idx) = [];
            
            eNoises_kr = eNoise{ind(1)};
            for i = ind(2:end)
               eNoises_kr = krprod(eNoise{i}, eNoises_kr); 
            end
            self.SST = sum(bsxfun(@times, Xm.^2, eNoises_kr'), 2); 
        end
        
        function sse = calcSSE(self,Xm, Rm, eFact, eFact2, eFact2pairwise, eNoise)
            
            [kr, ~, krkr] = self.calcSufficientStats(eFact, eFact2);
            
            self.updateSST(Xm, eNoise);
            
            if self.has_missing_values
                ind = 1:length(eFact);
                ind(self.mode_idx) = [];
                
                B_unf = computeUnfoldedMoments(eFact(ind),{[]});

                kr_ = B_unf{1};
                kr2_ = eFact2pairwise{ind(1)};
                if iscell(eNoise)
                    noise_ = eNoise{ind(1)};
                end
                
                for i = 2:length(ind)
                    kr_ = krprod(B_unf{i}, kr_);
                    kr2_ = krprod(eFact2pairwise{ind(i)}, kr2_);
                    if iscell(eNoise)
                        noise_ = krprod(eNoise{ind(i)}, noise_);
                    end
                    
                end
                
%                 sigA = eFact2{1}-eFact{1}.^2;
%                 sigB = kr2-kr.^2;
                
                if iscell(eNoise)
                    
                    A_unf = computeUnfoldedMoments(...
                        bsxfun(@rdivide, eFact{self.mode_idx}, eNoise{self.mode_idx})...
                        ,[]);
                    
                else
                    A_unf = computeUnfoldedMoments(eFact{self.mode_idx},[]);
                    
                end
                sigA_ = eFact2pairwise{self.mode_idx}-A_unf;
                sigB_ = kr2_ - kr_;
                
                if iscell(eNoise)
                    
                    kr_ = bsxfun(@times, kr_, noise_);
                    sigB_ = bsxfun(@times, sigB_, noise_);
                    
                    M3 = sum(Rm.*(A_unf*(kr_+sigB_)'),2)+...
                        sum(Rm.*(sigA_*(kr_+sigB_)'),2);
                    
                    self.SSE=self.SST+M3-2*sum(Xm.*(...
                        bsxfun(@rdivide, eFact{self.mode_idx}, eNoise{self.mode_idx})*kr')...
                        ,2);
                else
                    
                    M3 = sum(sum((Rm.*(A_unf*kr_'))))+...
                        sum(sum(Rm.*(A_unf*sigB_')))+...
                        sum(sum(Rm.*((sigA_)*kr_')))+...
                        sum(sum(Rm.*(sigA_*sigB_')));
                    
                    self.SSE=(self.SST+M3-2*sum(sum(Xm.*(eFact{self.mode_idx}*kr'))));
                end
                
                
                
            else
                for i = length(eFact):-1:1
                    eFact__{i} = bsxfun(@rdivide, eFact{i}, sqrt(eNoise{i}));
                end

                [~, ~, krkr] = self.calcSufficientStats(eFact__, eFact2);
                
                %self.SSE=(self.SST+sum(sum(krkr))-2*sum(sum(Xm.*(eFact{1}*kr'))));
                self.SSE=self.SST ... 
                    +sum(bsxfun(@times, eFact2pairwise{self.mode_idx}, krkr(:)'),2)...
                    -2*sum(...eFact{self.mode_idx}
                    bsxfun(@rdivide, eFact{self.mode_idx}, eNoise{self.mode_idx})...
                    .*(Xm*kr),2);
            end
            assert(all(self.SSE >= 0))
            sse = self.SSE;
        end
        
        function updateNoise(self, Xm, Rm, eFact, eFact2, eFact2pairwise, eNoise)
            % Update error
            self.calcSSE(Xm,Rm,eFact,eFact2, eFact2pairwise, eNoise);
            
            %
            self.est_alpha = self.hp_alpha+self.number_of_elements/2;
            self.est_beta = self.hp_beta+self.SSE/2;
            
            if strcmpi(self.inference_method,'variational')
                self.noise_value = self.est_alpha ./ self.est_beta;
                self.noise_log_value = psi(self.est_alpha)-log(self.est_beta);
            elseif strcmpi(self.inference_method,'sampling')
                % Gamrnd samples from a gamma distribution with shape (alpha) and
                % scale (1./ beta)
                self.noise_value = gamrnd(self.est_alpha, 1./self.est_beta);
                self.noise_log_value = log(self.noise_value);
            end
            
            assert(isreal(self.noise_log_value))
        end
        
        function cost = calcCost(self, other_exp_log_noise)
            
            if nargin == 1
                other_exp_log_noise = 0;
            end
            
            cost = -0.5*sum(self.number_of_elements)*log(2*pi)...
                +0.5*sum(self.number_of_elements.*self.noise_log_value)...
                +0.5*sum(other_exp_log_noise(:))... % <- normalization constant
                -0.5*sum(self.SSE.*self.noise_value); %<- reconstruction error
            
            cost = cost +  self.calcPrior;
            
            if strcmpi(self.inference_method, 'variational')
                cost = cost+self.calcEntropy();
            end
            
        end
        
        function cost = calcPriorAndEntropy(self)
            cost = self.calcPrior;
            
            if strcmpi(self.inference_method, 'variational')
                cost = cost+self.calcEntropy();
            end
        end
        
        function val = getExpFirstMoment(self)
            val = self.noise_value;
        end
        
        function val = getExpLogMoment(self)
            val = self.noise_log_value;
        end
        
        function val = getSSE(self)
            val = self.SSE;
        end
        
    end
    
    methods (Access = private)
        function [kr, kr2, krkr] = calcSufficientStats(self, eFact, eFact2)
            % Assumes the noise is on the self.mode_idx mode
            % Note that this assumes eFact2 to be an elementwise squared
            % expectation
            % It is not the same as <eFact'eFact> ... TODO...d
            % Note: eFact2 is <A'*diag(tau_A)*A>
            
            D = size(eFact{1},2);
            kr=ones(1,D);
            kr2=ones(1,D);
            krkr = ones(D,D);
            
            if self.has_missing_values
                krkr = [];
            else
                kr2 = [];
            end
            
            ind = 1:length(eFact);
            ind(self.mode_idx) = [];
            
            for i=ind
                kr=krprod(eFact{i}, kr);
                if self.has_missing_values
%                     kr2=krprod(eFact2{i}, kr2);
                else
%                     if self.isFactorUnivariate(i)
%                         krkr=krkr.*((eFact{i}'*eFact{i}).*(ones(D)-eye(D))+diag(sum(eFact2{i},1)));
%                     else
                        krkr = krkr.*eFact2{i};
%                     end
                end
            end
        end
        
        function entropy_contr = calcEntropy(self)
            entropy_contr =sum(-log(self.est_beta))...
                +sum(self.est_alpha)...
                -sum((self.est_alpha-1).*psi(self.est_alpha))...
                +sum(gammaln(self.est_alpha)...
                );
            
%             entropy_contr =sum(...
%                 -log(self.est_beta)...
%                 +self.est_alpha...
%                 -(self.est_alpha-1).*psi(self.est_alpha)...
%                 +gammaln(self.est_alpha)...
%                 );

        end
        
        function prior_contr = calcPrior(self)
            n_tau = size(self.noise_value,1);
            prior_contr = -gammaln(self.hp_alpha)*n_tau...
                +self.hp_alpha*log(self.hp_beta)*n_tau...
                +(self.hp_alpha-1)*sum(self.noise_log_value)...
                -self.hp_beta*sum(self.noise_value);
            
        end
    end
    
    
end