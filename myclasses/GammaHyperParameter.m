classdef GammaHyperParameter < HyperParameterInterface
    %UNTITLED10 Summary of this class goes here
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
        num_obs
        
    end
    
    properties %(Access = private)
        est_alpha = [];
        est_beta = [];
        effective_num_obs;
    end
    
    methods
        function obj = GammaHyperParameter(...
                factor_distribution, enforced_property, ...
                factorsize, a_shape, b_rate, inference_method, effective_obs)
            if nargin <7 || isempty(effective_obs)
                if strcmpi(factor_distribution,'exponential')
                    obj.distr_constr = 1;
                else
                    obj.distr_constr = 0.5;
                end
                obj.effective_num_obs = factorsize(1)*obj.distr_constr;
            else %Shared..
                obj.distr_constr = [];
                obj.effective_num_obs = effective_obs;
            end
            
            obj.factorsize = [];
            if exist('b_rate','var') && ~isempty(a_shape) && ~isempty(b_rate)
                obj.hp_alpha = a_shape;
                obj.hp_beta = b_rate;
            end
            
            if obj.display
                fprintf('Using default shape (%2.2e) and rate (%2.2e)\n', ...
                    obj.hp_alpha, obj.hp_beta)
            end
            
            obj.prior_property = enforced_property;
            if strcmpi(enforced_property, 'sparse')
                obj.prior_value = obj.hp_alpha/obj.hp_beta*ones(factorsize);
            elseif strcmpi(enforced_property, 'ard')
                obj.prior_value = obj.hp_alpha/obj.hp_beta*ones(1,factorsize(2));%,1);
            elseif strcmpi(enforced_property, 'scale')
                obj.prior_value = obj.hp_alpha/obj.hp_beta;
            end
            obj.prior_size = size(obj.prior_value);
            obj.num_obs = factorsize(1);
            
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
            if nargin == 1
                val = self.prior_value;
            else
                if all(self.prior_size == factorsize)
                    val = self.prior_value;
                elseif length(self.prior_size) == 2 && self.prior_size(2) == factorsize(2)
                    val = repmat(self.prior_value, factorsize(1), 1);
                elseif self.prior_size == 1
                    val = repmat(self.prior_value, factorsize);
                else
                    error('Unknown relation between prior and factor size')
                end
            end
        end
        
        function val = getExpLogMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if nargin == 1
                val = self.prior_log_value;
            else
                if all(self.prior_size == factorsize)
                    val = self.prior_log_value;
                elseif length(self.prior_size) == 2 && self.prior_size(2) == factorsize(2)
                    val = repmat(self.prior_log_value, factorsize(1), 1);
                elseif self.prior_size == 1
                    val = repmat(self.prior_log_value, factorsize);
                else
                    error('Unknown relation between prior and factor size')
                end
            end
        end
        
        function updatePrior(self, eFact2, distr_constant)
            if ~exist('distr_constant','var') || isempty(distr_constant)
                distr_constant = self.distr_constr;
            end
            
            % The same prior can be shared by multiple factor matrices
            if iscell(eFact2)
                if strcmpi(self.prior_property,'sparse')
                    if all(distr_constant == distr_constant(1))
                        eFact2=sum(cat(3,eFact2{:}),3)*distr_constant(1);
                    else
                        for i = 1:length(eFact2)
                            eFact2{i} = eFact2{i}*distr_constant(i);
                        end
                        eFact2=sum(cat(3,eFact2{:}),3);
                    end
                    
                else
                    % scale or ard
                    fact_sizes = cell2mat(cellfun(@size, eFact2, 'UniformOutput',false));
                    is_normal = all(fact_sizes == size(eFact2{1},2), 2);
                    % Sum and diagonalize Exponential distr contribution.
                    eFact2(~is_normal) = cellfun(@(x) diag(sum(x,1)) ,eFact2(~is_normal), 'Uniformoutput', false);
                    % multiply by correct distr constant
                    for i = 1:length(eFact2)
                        eFact2{i} = eFact2{i}*distr_constant(i);
                    end
                    eFact2 = sum(cat(3,eFact2{:}),3); % sum all contributions.
                    
                end
            else
                if ~all(size(eFact2) == size(eFact2,2)) && ~strcmpi(self.prior_property,'sparse')
                    eFact2 = diag(sum(eFact2, 1));
                end
                
                eFact2 = eFact2*self.distr_constr;
            end
            
            [~, D] = size(eFact2);
            N = self.effective_num_obs;
            
            % The number of dependent parameters and update rule is
            % different for each prior property.
            if strcmpi(self.prior_property, 'scale')
                self.est_alpha = self.hp_alpha+N*D;
                self.est_beta = self.hp_beta+trace(eFact2);
                
            elseif strcmpi(self.prior_property, 'ard')
                self.est_alpha = self.hp_alpha+N;
                self.est_beta = self.hp_beta+diag(eFact2)';
                
            elseif strcmpi(self.prior_property, 'sparse')
                self.est_alpha = self.hp_alpha+sum(distr_constant);
                self.est_beta = self.hp_beta+eFact2;
                
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
                self.prior_value = gamrnd(repmat(self.est_alpha,self.prior_size)...
                    , 1 ./ self.est_beta);
                self.prior_log_value = log(self.prior_value);
            end
            assert(all(isreal(self.prior_log_value)))
            
        end
        
        function cost = calcCost(self)
            if strcmpi(self.inference_method,'variational')
                cost = self.calcPrior()+sum(sum(self.calcEntropy()));
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
                entropy_contr =-sum(log(self.est_beta(:)))+...
                    prod(self.prior_size)*(self.est_alpha-(self.est_alpha-1)...
                    *psi(self.est_alpha)+gammaln(self.est_alpha));
            end
        end
        
        function prior_contr = calcPrior(self)
            if isempty(self.est_beta)
                prior_contr = prod(self.prior_size)*...
                    (-gammaln(self.hp_alpha)+self.hp_alpha*log(self.hp_beta))...
                    +(self.hp_alpha-1)*sum(self.prior_log_value(:))...
                    -self.hp_beta*sum(self.prior_value(:));
                
            else
                prior_contr = prod(self.prior_size)*...
                    (-gammaln(self.hp_alpha)+self.hp_alpha*log(self.hp_beta))...
                    +(self.hp_alpha-1)*sum(self.prior_log_value(:))...
                    -self.hp_beta*sum(self.prior_value(:));
            end
        end
        
    end
end