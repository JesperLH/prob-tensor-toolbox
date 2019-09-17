function [factor_inference, noise_inference ] = ...
    processAndValidateInferenceSchemes(inference_scheme, Nmodes )
%PROCESSANDVALIDATEINFERENCESCHEMES Summary of this function goes here
%   Detailed explanation goes here

noise_inference = 'variational';

if isa(inference_scheme,'char')
    % Use same inference scheme for all distributions
    factor_inference = repmat({inference_scheme},Nmodes,2);
    noise_inference = inference_scheme;
elseif all(size(inference_scheme) == [Nmodes, 2])
    % Custom inference scheme selection. Note this may result in warnings.
    factor_inference = inference_scheme;
elseif all(size(inference_scheme) == [Nmodes+1, 2])
    factor_inference = inference_scheme(1:end-1,:);
    noise_inference  = inference_scheme(end,1);
    
elseif max(size(inference_scheme)) == 2 
    % Tensor => different scheme for factor and hyperparameter
        if Nmodes == 2
            % Matrix case
            warning(['Inference schemes are ambiguous!!\n',...
                     'For sampling the 1st factor and performing varational inference',...
                     ' on the second factor, write \n\t{''sampling'' ; ''variational''}\n' ,...
                     'For sampling the factor and performing varational inference ',...
                     ' on the hyperparameters, write \n\t{''sampling'' , ''variational''}'])
             
            if size(inference_scheme,1) == 1
                factor_inference = repmat(inference_scheme,2,1);
            elseif size(inference_scheme,2) == 1
                factor_inference = repmat(inference_scheme,1,2);
            end
            
        else
           % Tensor: Different schemes for factors and hyperparamters
           if size(inference_scheme,2) == 1
               inference_scheme = inference_scheme';
           end
           factor_inference = repmat(inference_scheme,Nmodes,1);
           
        end
elseif max(size(inference_scheme)) == Nmodes
    if size(inference_scheme,1) == 1
        inference_scheme = inference_scheme';
    end
    factor_inference = repmat(inference_scheme,2);

elseif max(size(inference_scheme)) == Nmodes+1
    if size(inference_scheme,1) == 1
        inference_scheme = inference_scheme';
    end
    factor_inference = repmat(inference_scheme(1:end-1),2);
    noise_inference = inference_scheme(end);
    
else
    error('Unknown size of inference scheme! Got (%i x %i), but ndims(X)=%i',...
            ndims(X))
        
end


end

