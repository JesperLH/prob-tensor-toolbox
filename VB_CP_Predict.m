function [A, AtA, lambda, elbo, model] = VB_CP_Predict(...
    X, trained_model, constraints, use_trained_factors, use_trained_noise, varargin)
%VB_CP_PREDICT Fits a Candecomp/PARAFAC (CP) tensor decomposition model to
%   the data "X" using an already trained model "trained_model" with
%   constraints on the modes specified by "constraints". This function
%   assumes some of the factor matrices are fixed from the training to test
%   data while others arent.
% INPUTS:
% There are 3 required inputs:
%   X:              A tensor (2nd order or higher) with the observed data.
%                       Missing values should be represented by NaN.
%   trained:        A model trained using VB_CP_ALS with the same
%                   constraints.
%   constraints:    A cell(ndims(X),1) containing a string which specify
%                   the constraint on the corresponding mode. See "help
%                   VB_CP_ALS" for more information.
%   use_trained_factors:
%                   A binary indicator of length ndims(X) indicating
%                   whether a trained factor from the model should be used
%                   (indicated by 1) or if a new factor should be fitted
%                   (indicated by 0).
%   use_trained_noise: 
%                   Either a binary scalar (homoscedastic noise) or vector
%                   (heteroscedastic noise). Again 1 indicates the trained
%                   noise should be used while 0 indicates it should be
%                   fitted by the model.
%   varargin:       Variable input arguments, same as for "help VB_CP_ALS"

% Determine which factors are fixed
trained_factors = cell(ndims(X),1);
for i = 1:ndims(X)
    if use_trained_factors(i)
        trained_factors{i} = trained_model.factors{i};
    end
end

% Determine which noise estimates are fixed
if isscalar(use_trained_noise)
    if use_trained_noise
        trained_noise = trained_model.noise;
    end
else
    trained_noise = cell(ndims(X),1);
    for i = 1:ndims(X)
        if use_trained_factors(i)
            trained_noise{i} = trained_model.noise{i};
        end
    end
end

if sum(use_trained_noise)==0
    trained_noise = [];
end

% Specify model options
model_options = {varargin{:}, ... % Pass all input arguments to VB_CP_ALS
                'fix_factors', use_trained_factors, 'init_factors', trained_factors,...
                'fix_noise', use_trained_noise, 'init_noise', trained_noise,...
                'noise',~arrayfun(@isempty,trained_model.noise)};

D_train = trained_model.factors{1}.factorsize(2); % Components fit during training

% Fit model.
[A, AtA, lambda, elbo, model] = VB_CP_ALS(X, D_train, constraints, model_options{:});


end