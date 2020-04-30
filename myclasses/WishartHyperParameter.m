classdef WishartHyperParameter < HyperParameterInterface
    %UNTITLED10 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        %distr_constr;
        W;
        Winv
        v;
        debug = true;
        rv_value;
    end
    
    properties (Constant)
        valid_prior_properties = {'Wishart'};%{'ard','sparse','scale'};
        supported_inference_methods = {'variational','sampling'};
        inference_default = 'variational';
    end
    
    properties %(Access = private)
        est_W = [];
        est_v = [];
    end
    
    methods
        function obj = WishartHyperParameter(...
                factorsize, inference_method)
            
            obj.factorsize = factorsize;
            N = factorsize(1);
            D = factorsize(2);
            obj.W = eye(D)*1e-3;
            obj.v = D+2;
            obj.Winv = symmetrizing(obj.W\eye(D), obj.debug);
            
            obj.est_W = obj.W;
            obj.est_v = obj.v;
            obj.prior_size = size(obj.prior_value);
            
            
            
            if exist('inference_method','var')
                obj.inference_method = inference_method;
            else
                obj.inference_method = obj.inference_default;
            end
            
            obj.prior_property = 'Wishart';
            
            
        end
          
        function val = getExpFirstMoment(self, factorsize)
            % Returns the expected value of the first moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if isempty(self.rv_value)
                val = self.est_W*self.est_v;
            else
                val = self.rv_value;
            end
        end
        
        function val = getExpLogMoment(self, factorsize)
            % Returns the expected value of the logdet moment (mean)
            % If factorsize is given, then the output is replicated to match.
            if strcmpi(self.inference_method,'variational')
                D=self.factorsize(2);
                val = sum(psi((self.est_v+1-(1:D))/2))+D*log(2)...
                    +2*sum(log(diag(chol(self.est_W))));
            elseif strcmpi(self.inference_method,'sampling')
                val = log(det(self.rv_value));
                % 2*sum(log(diag(chol(W))))
            end
            
        end
        
        function updatePrior(self, eFact2)
            
            % The same prior can be shared by multiple factor matrices
            if iscell(eFact2)
                    eFact2=sum(cat(3,eFact2{:}),3);
            end
            
            %[N, D] = size(eFact2);
            
            self.est_v = self.v + self.factorsize(1);
            self.est_W = (self.Winv + eFact2)\eye(size(eFact2,1));
            if strcmpi(self.inference_method,'variational')
                self.rv_value = self.est_W*self.est_v;
            elseif strcmpi(self.inference_method,'sampling')
                self.rv_value = wishrnd(self.est_W, self.est_v);
            end
        end
        
        function cost = calcCost(self)
            if strcmpi(self.inference_method,'variational')
                cost = self.calcPrior()+sum(sum(self.calcEntropy()));
            elseif strcmpi(self.inference_method,'sampling')
                cost = self.calcPrior();
            end
            
        end
        
        function entropy_contr = calcEntropy(self)
            
            D=self.factorsize(2);
            entropy_contr = -self.calcLogB(self.est_W, self.est_v, D)...
                            -(self.est_v-D-1)/2*self.getExpLogMoment()...
                            +self.est_v*D/2;
        end
        
        function prior_contr = calcPrior(self)
            D = self.factorsize(2);
            prior_contr = self.calcLogB(self.W, self.v, D)...
                          +(self.v-D-1)*self.getExpLogMoment()...
                          -1/2*sum(sum(self.Winv .* self.getExpFirstMoment()));

        end
        
    end
    
    methods (Access = protected)
        
        function b = calcLogB(~, W, v, D)
            b = -v/2*2*sum(log(diag(chol(W))))... % log(det(W))
                -(v*D/2*log(2)...
                +D*(D-1)/4*log(pi)...
                +sum(gammaln((v+1-(1:D))/2)));
        end        
    end
end