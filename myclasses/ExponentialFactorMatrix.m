classdef ExponentialFactorMatrix < TruncatedNormalFactorMatrix
    % ExponentialFactorMatrix
    %   Detailed explanation goes here
    
    %% Required functions
    methods
        function obj = ExponentialFactorMatrix(...
                shape, prior_choice, has_missing,...
                inference_method)
            % Call base constructer
            obj = obj@TruncatedNormalFactorMatrix(...
                shape, prior_choice, ...
                has_missing,inference_method);
            
            obj.hyperparameter = prior_choice;
            obj.distribution = 'Elementwise Exponential Distribution';
            obj.factorsize = shape;
            obj.optimization_method = 'hals';
            obj.data_has_missing = has_missing;
            
            
            if exist('inference_method','var')
                obj.inference_method = inference_method;
            else
                obj.inference_method = obj.inference_default;
            end
            
            
        end
        
        function updateFactor(self, update_mode, Xm, Rm, eFact, eFact2, eFact2elem, eNoise)
            
            if strcmpi(self.optimization_method,'hals')
                %hals_update@TruncatedNormalFactorMatrix(self,update_mode, Xm, Rm, eFact, eFact2, eNoise);
                self.hals_update(update_mode, Xm, Rm, eFact, eFact2, eFact2elem, eNoise)
            else
                error('Unknown optimization method')
            end
            
        end
        
        %function entro = getEntropy(self)
        %%% Entroy remains the same as in TruncatedNormalFactorMatrix,
        %%% since the Q-distribution is TruncatedNormal due to the squared
        %%% loss function.
        %end
        
        function logp = getLogPrior(self)
            % Gets the prior contribution to the cost function.
            % Note. The hyperparameter needs to be updated before calculating
            % this.
            
            logp = sum(sum(self.hyperparameter.getExpLogMoment(self.factorsize)))...
                - sum(sum(bsxfun(@times, self.hyperparameter.prior_value ,...
                self.getExpFirstMoment())));
            
            if strcmpi(self.inference_method,'sampling')
                warning('Check if log prior for exponential distribution, is equal when sampling and using vb')
            end
        end
    end
    
    %% Class specific functions
    methods (Access = protected)
        
        function calcSufficientStats(self, kr2_sum, eNoise)
            % Calculate elementwise variance
            if any(strcmpi(self.inference_method,{'variational','sampling'}))
                %self.factor_sig2 = 1./bsxfun(@plus, kr2_sum*eNoise, ...
                %        self.hyperparameter.getExpFirstMoment(self.factorsize));
                self.factor_sig2 = 1./bsxfun(@plus, kr2_sum.*eNoise, ...
                    zeros(self.factorsize));
                % TODO: The variance will become infinite if
                % kr2_sum.*eNoise is zero for any reason! This is actually
                % valid in some scenarios for the Exponential distr (since
                % the variance is not regularized) while it is invalid for
                % the truncated normal distribution.
                
                assert(~any(isinf(self.factor_sig2(:))),'Infinite sigma value?') % Sanity check
                assert(all(self.factor_sig2(:)>=0),'Variance was negative!')
            else
                
            end
        end
        
        function [mu] = calculateNormalMean(self, d, not_d, Xmkr, eNoise, krkr, Rkrkr)
            % Calculate the non-truncated mean value
            lambda = self.hyperparameter.getExpFirstMoment(self.factorsize);
            if self.data_has_missing
                D= size(Xmkr,2);
                IND = D*(d-1)+not_d;
                mu = ((Xmkr(:,d)-(sum(Rkrkr(:,IND).*self.factor(:,not_d),2))...
                    ).*eNoise-lambda(:,d)).*self.factor_sig2(:,d);
            else
                mu= ((Xmkr(:,d)-self.factor(:,not_d)*krkr(not_d,d)).*eNoise...
                    -lambda(:,d)).*self.factor_sig2(:,d);
            end
            assert(~any(isinf(mu(:))),'Infinite mean value?')
            self.factor_mu(:,d) = mu;
        end
        
    end
end

