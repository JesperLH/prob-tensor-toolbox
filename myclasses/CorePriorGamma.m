classdef CorePriorGamma < HyperParameterInterface
    %CorePriorGamma Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Constant)
        valid_prior_properties = {'ard','sparse','scale'};
        supported_inference_methods = {'variational','sampling'};
        inference_default = 'variational';
    end
    
    properties
        hp_alpha = 1e-4; % Shape
        hp_beta = 1e-4; % Rate
        
        display = true;
        distr_constr;
        coresize=[];
        num_rand_var;
    end
    
    properties %(Access = private)
        est_alpha = [];
        est_beta = [];
    end
    
    methods
        function obj = CorePriorGamma(...
                factor_distribution, enforced_property, ...
                coresize, a_shape, b_rate, inference_method)
            
            obj.factorsize = [];
            if exist('b_rate','var') && ~isempty(a_shape) && ~isempty(b_rate)
                obj.hp_alpha = a_shape;
                obj.hp_beta = b_rate;
            end
            
            obj.prior_property = enforced_property;
            if strcmpi(enforced_property, 'sparse')
                obj.prior_value = obj.hp_alpha/obj.hp_beta*ones(prod(coresize),1);
            elseif strcmpi(enforced_property, 'ard')
                obj.prior_value = obj.hp_alpha/obj.hp_beta*ones(sum(coresize),1);
            elseif strcmpi(enforced_property, 'scale')
                obj.prior_value = obj.hp_alpha/obj.hp_beta;
            end
            obj.prior_size = size(obj.prior_value);
            obj.coresize = coresize;
            obj.prior_log_value = (psi(obj.hp_alpha)-log(obj.hp_beta))*ones(...
                size(obj.prior_value));
            
            if exist('inference_method','var')
                obj.inference_method = inference_method;
            else
                obj.inference_method = obj.inference_default;
            end
            
        end
        
        
        function val = getExpFirstMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if strcmpi(self.prior_property,'ard')
                val = 1;
                M = length(self.coresize);
                invD = cumsum([0,self.coresize(end:-1:1)]); % Indexing
                for i=1:M
                    val = krprod(val,self.prior_value(invD(i)+1:invD(i+1)));
                end
            else
                val = self.prior_value;
            end
        end
        
        
        function val = getExpLogMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if strcmpi(self.prior_property,'ard')
                M = length(self.coresize);
                val = 1;
                invD = cumsum([0,self.coresize(end:-1:1)]); % Indexing
                for i=1:M % M:-1:1 % Order is already switched....
                    val = krprod(val,self.prior_log_value(invD(i)+1:invD(i+1)));
                end
                
            else
                val = self.prior_log_value;
            end
        end
        
        
        function updatePrior(self, eContribution, distr_constant)
            D = self.coresize;
            
            % The number of dependent parameters and update rule is
            % different for each prior property.
            if strcmpi(self.prior_property, 'ard')
                
                n_repeated_updates = 5;
                self.updateSliceArdPrior(eContribution, distr_constant,n_repeated_updates);
                return; % ARD prior is handled in separate function
                
            elseif strcmpi(self.prior_property, 'scale')
                self.est_alpha = self.hp_alpha+prod(D)*distr_constant;
                self.est_beta = self.hp_beta+trace(eContribution)*distr_constant;
                
            elseif strcmpi(self.prior_property, 'sparse')
                self.est_alpha = self.hp_alpha+sum(distr_constant);
                self.est_beta = self.hp_beta+diag(eContribution)*distr_constant;
                
            else
                error('No update rule for property "%s"',self.prior_property)
            end
            
            % Select and apply inference method
            if strcmpi(self.inference_method,'variational')
                self.prior_value = self.est_alpha./self.est_beta;
                self.prior_log_value = psi(self.est_alpha)-log(self.est_beta);
                
            elseif strcmpi(self.inference_method,'sampling')
                % Gamrnd samples from a gamma distribution with shape (alpha) and
                % scale (1./ beta)
                self.prior_value = gamrnd(self.est_alpha, 1 ./ self.est_beta);
                self.prior_log_value = log(self.prior_value);
            end
            assert(all(isreal(self.prior_log_value)))
            
        end
        
        
        function cost = calcCost(self)
            if strcmpi(self.inference_method,'variational')
                cost = self.calcPrior()+self.calcEntropy();
            elseif strcmpi(self.inference_method,'sampling')
                cost = self.calcPrior();
            end
            assert(isreal(cost), 'Not a real number!')
        end
        
        
        function entropy_contr = calcEntropy(self)
            if isempty(self.est_beta)
                entropy_contr = numel(self.prior_value)*(...
                    -log(self.hp_beta)+self.hp_alpha...
                    -(self.hp_alpha-1)*psi(self.hp_alpha)...
                    +gammaln(self.hp_alpha));
            else
                entropy_contr =-sum(log(self.est_beta(:)))...
                    + numel(self.est_beta)/numel(self.est_alpha)...  
                    * sum(self.est_alpha(:)-(self.est_alpha(:)-1)...
                    .* psi(self.est_alpha(:))+gammaln(self.est_alpha(:)));
                
            end
        end
        
        
        function prior_contr = calcPrior(self)
            prior_contr = numel(self.prior_value)*...
                (-gammaln(self.hp_alpha)+self.hp_alpha*log(self.hp_beta))...
                +(self.hp_alpha-1)*sum(self.prior_log_value(:))...
                -self.hp_beta*sum(self.prior_value(:));
        end
        
        
    end
    
    
    methods (Access = private)
        
        function updateSliceArdPrior(self, eContri, distr_c, num_updates)
            
            eContri = reshape(diag(eContri),self.coresize);
            
            M = length(self.coresize);
            invD = cumsum([0,self.coresize(end:-1:1)]); % Indexing
            if ~isempty(self.est_alpha)
                Psi = self.est_alpha./self.est_beta; % Expected Psi
            else
                Psi = self.prior_value(1)*ones(sum(self.coresize),1);
                self.est_alpha = nan(sum(self.coresize),1);
                self.est_beta = nan(sum(self.coresize),1);
            end
            
            %%
            for i_up = 1:num_updates
                %[~,mode_update_order] = sort(self.coresize,'descend'); % Largest mode first
                mode_update_order = randperm(M); % at random
                %mode_update_order = 1:M; % 
                for i_m = mode_update_order
                    % Update each mode-ARD
                    ind = 1:M; ind(i_m) = [];
                    
                    eP=1;
                    for i=(M-ind(end:-1:1)+1)
                       eP = krprod(eP,Psi(invD(i)+1:invD(i+1)));
                    end
                    
                    assert(size(eP,1)==prod(self.coresize(ind)),'Sizes should not differ!')
                    eC = matricizing(eContri,i_m);
                    idx_ = invD(M-i_m+1)+1:invD(M-i_m+1+1);
                    self.est_alpha(idx_) = self.hp_alpha + prod(self.coresize(ind))*distr_c;
                    self.est_beta(idx_) = self.hp_beta + distr_c * sum(eC .* eP',2);

                    if strcmpi(self.inference_method,'variational')
                        Psi(idx_) = self.est_alpha(idx_)./self.est_beta(idx_);
                    else%if strcmpi(self.inference_method,'sampling')
                        Psi(idx_) = gamrnd(self.est_alpha(idx_), 1./self.est_beta(idx_));
                    end
                
                end
            end
            
            if strcmpi(self.inference_method,'variational')
                self.prior_value = self.est_alpha./self.est_beta;
                self.prior_log_value = psi(self.est_alpha) -log(self.est_beta);
                
            elseif strcmpi(self.inference_method,'sampling')
                % Gamrnd samples from a gamma distribution with shape (alpha) and
                % scale (1./ beta)
                self.prior_value = gamrnd(self.est_alpha, 1 ./ self.est_beta);
                self.prior_log_value = log(self.prior_value);
            end
            
            assert(all(isreal(self.prior_log_value)))
            
            
        end
        
    end
    
    
end