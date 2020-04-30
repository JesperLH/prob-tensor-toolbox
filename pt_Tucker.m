function [E_CORE, E_FACT,E_FACT2, E_Lambda, Lowerbound, model,all_samples]=pt_Tucker(X,D,factor_constraints, core_constraint,varargin)
%pt_Tucker Fits a Tucker tensor decomposition model to the data "X"
% with "D=[D1,D2,...,DN]" components (rank) and subject to the supplied "constraints".
% INPUTS:
% There are 4 required inputs:
%   X:              A tensor (2nd order or higher) with the observed data.
%                       Missing values should be represented by NaN.
%   D:              The number of components (or rank) in each mode to be estimated.
%   constraints:    A cell(ndims(X),1) containing a string which specify
%                   the constraint on the corresponding mode. How to
%                   specify each constraint and the distribution of the
%                   factor is given below:
%
%       'normal'    The factor follows a multivariate normal distribution
%                   with one of the following priors:
%                       'constant'  Zero mean and unit covariance
%                       'scale'     Zero mean and isotropic covariance (precision follows a Gamma distribution)
%                       'ard'       Zero mean and diagonal covariance (each elements (columns) precision follows a Gamma distribution)
%                       'wishart'   Zero mean and full covariance (the precision matrix (inverse covariance) follows a Wishart distribution
%       'non-negative'
%                   Each element follows a univariate truncated normal
%                   distribution with one of the following priors:
%                       'constant', 'scale', 'ard'  (As defined for the normal)
%                       'sparse'    Zero mean and element specific precision (following a Gamma distribution)
%       'non-negative exponential'
%                   Each element follows a univariate exponential
%                   distribution, with one of the following priors:
%                   'constant', 'scale', 'ard', 'sparse  (As defined for "non-negative")
%       'orthogonal'
%                   This factor follows a von Mises-Fisher matrix
%                   distribution, where the prior is uniform across the
%                   sphere (e.g. concentration matrix = 0).
%                       NOTE: It is not possible for the user to specify a
%                       prior for this distribution. 
%       'infty'
%                   Each element Follows a Uniform(0,1) distribution, which
%                   has no priors.
%
%       Note:       If the same prior is specified on two or more modes, then the
%                   prior is shared (by default), e.g. 'normal ard' and
%                   'normal ard'. The prior can be made mode specific by
%                   appending 'individual' (or indi') when specifying the
%                   constraint, e.g. 'normal ard indi'.
%   varargin:   Passing optional input arguments
%      'conv_crit'      Convergence criteria, the algorithm stops if the
%                       relative change in lowerbound is below this value
%                       (default 1e-8).
%      'maxiter'        Maximum number of iteration, stops here if
%                       convergence is not achieved (default 500).
%      'model_tau'      Model the precision of homoscedastic noise
%                       (default: true)
%      'model_lambda'   Model the hyper-prior distributions on the factors
%                       (default: true)
%      'model_psi'      Model the hyper-prior distributions on the core
%                       array (default: true)
%      'fixed_tau'      Number of iterations before modeling noise
%                       commences (default: 1).
%      'fixed_lambda'   Number of iterations before modeling hyper-prior
%                       distributions (default: 1).
%      'fixed_psi'      Number of iterations before modeling of core prior
%                       commences (default: 1).
%      'inference'      Defines how probabilistic inference should be
%                       performed (default: 'variational', alternative: 'sampling').
%      'init_factors'   A cell array with inital factors (first moment only).
%
% OUTPUTS:
%   'E_CORE'        Expected first moment of the core array.
%   'E_FACT':       Expected first moment of the factor matrices.
%   'E_FACT2':      Expected second moment of the factor matrices.
%   'E_Lambda':     Expected first moment of the hyper-prior distributions.
%   'Lowerbound':   The value of the evidience lowerbound at each
%                   iteration (for variational inference).
%   'model'         Distributions and lots of stuff... (currently mostly
%                   for debugging)
%   'all_samples'   If a variable is sampled (Gibbs sampling), then its
%                   value at each iteration is returned.
%
%% Initialize paths
if exist('./setup_paths.m') && ~exist('CoreArrayInterface.m')
    evalc('setup_paths');
end

%% Read input and Setup Parameters
paramNames = {'conv_crit', 'maxiter', ...
    'model_tau', 'fixed_tau', 'tau_a0', 'tau_b0',...
    'model_lambda', 'fixed_lambda', 'lambda_a0', 'lambda_b0'...
    'model_psi', 'fixed_psi', 'psi_a0', 'psi_b0'...
    'init_method','inference','noise','init_factors','impute missing',...
    'init_noise', 'fix_factors', 'fix_noise'};
defaults = {1e-8, 100, ...
    true , 1  , 1e-5, 1e-6,...
    true , 1 , 1e-4, 1e-4,...
    true , 1 , 1e-5, 1e-6,...
    [], 'variational',false(ndims(X),1), [], false,[], [], []};
% Initialize variables to default value or value given in varargin
[conv_crit, maxiter, ...
    model_tau, fixed_tau, tau_alpha0, tau_beta0,...
    model_lambda, fixed_lambda, lambda_alpha0, lambda_beta0,...
    model_psi, fixed_psi, alpha_psi, beta_psi,...
    init_method, inference_scheme, hetero_noise_modeling, initial_factors,...
    usr_impute_missing, init_noise, dont_update_factor, dont_update_noise]...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});

debug_mode = false;
fixed_core = 0; 
update_core_prior =model_psi;
update_core_array =true;

if debug_mode
    model_tau = true;
    model_lambda = true;
    update_core_prior = true;
    fixed_core = 0; 
    fixed_lambda = 1;%3;
    fixed_tau = 1;
    fixed_psi = 1;
    conv_crit = 1e-8;
    alpha_psi = 1e-5;
    beta_psi = 1e-6;
    tau_alpha0 = 1e-5;
    tau_beta0 = 1e-6;
end

init_nway_tucker = false; init_ml_als = false; init_using_sampling=false;
if strcmpi(init_method,'sampling')
    init_using_sampling=true;
elseif strcmpi(init_method,'nway')
    init_nway_tucker = true;
elseif strcmpi(init_method,'ml')
    init_ml_als = true;
elseif ~isempty(init_method)
    error('Initialization method "%s" is unknown.',init_method)
else % default
    init_using_sampling=true;
%     init_nway_tucker = true;
%     init_ml_als = true;
end

all_orthogonal = all(my_contains(factor_constraints,'orthogonal','IgnoreCase',true));
any_othogonal_factors = any(my_contains(factor_constraints,'orthogonal','IgnoreCase',true));
% all_orthogonal = false;

if all_orthogonal
end

% Check input is allowed (and sensible)
assert(ndims(X) == length(hetero_noise_modeling), ' Unknown noise modeling strategy')
assert(length(factor_constraints) == ndims(X),...
    'The order of the data is not equal to the number of constraints.')
hetero_noise_modeling = logical(hetero_noise_modeling);

% The modeling delay time must be lower than the maximum number of iterations
if maxiter < fixed_lambda
    fixed_lambda=floor(maxiter/2);
end
if maxiter < fixed_tau
    fixed_tau=floor(maxiter/2);
end

factor_constraints = strtrim(factor_constraints); % Remove leading and trailing whitespaces
inference_scheme = strtrim(inference_scheme);

% Determine how factor and noise inference should be performed.
[inference_scheme, noise_inference ] = processAndValidateInferenceSchemes(...
    inference_scheme, ndims(X) );
only_variational_inference = all(all(strcmpi('variational',inference_scheme)));
some_variables_are_sampled = any(any(strcmpi('sampling',inference_scheme)));
all_samples = cell(ndims(X)+2, 2);

%% Initialize the factor matrices and noise
R_obs=~isnan(X); % Set observed (true) or not observed (false) status.
has_missing = ~all(R_obs(:));
impute_x = ~R_obs;

if has_missing && usr_impute_missing
    R_obs(:) = true; % Set to true to avoid any marginalization
    has_missing = false;
elseif has_missing
    error('Marginalization of missing values are not support.')
end

Nx=ndims(X); %Tensor order
N=size(X); %Elements in each dimension

% Initialize distributions according to the specified constraints
[factors, priors, shares_prior] = initializePriorDistributions(...
    N,D, has_missing, ...
    factor_constraints, inference_scheme);

core_inference_scheme = inference_scheme{1,1};%%'variational';
core_prior = CorePriorGamma('normal',core_constraint, D, alpha_psi, beta_psi, core_inference_scheme);
core_array = CoreArrayNormal(D,core_prior, has_missing, core_inference_scheme,all_orthogonal);

% Get relevant expected moments of the factors and setup X and R matricized
% for each mode.
E_FACT = cell(Nx,1);
E_FACT2 = cell(Nx,1);
E_Contribution2FactorPrior = cell(Nx,1);
E_Lambda = cell(Nx,1);
for i = 1:Nx  % Setup each factor
    
    init_E2 = false;
    % Initialize factor as specified by the user (if provided)
    if ~isempty(initial_factors) && ~isempty(initial_factors{i})
        if isobject(initial_factors{i}) ...
                && strcmp(class(initial_factors{i}),class(factors{i}))
            % FactorMatrix object passed
            factors{i} = initial_factors{i};
            priors{i} = factors{i}.hyperparameter; % !TODO Investigate what this call brakes!
        elseif (ismatrix(initial_factors{i}) && ~isobject(initial_factors{i}))...
                || iscell(initial_factors{i})
            if iscell(initial_factors{i})  % First and second moment passed
                factors{i}.initialize(initial_factors{i}{1});
                E_FACT2{i} = initial_factors{i}{2};
                init_E2 = true;
            else
                factors{i}.initialize(initial_factors{i});
            end
        end
    end    
    % Get/initialize expected values
    E_FACT{i} = factors{i}.getExpFirstMoment();
    if ~init_E2
        if my_contains(factor_constraints{i},'ortho','IgnoreCase',true)
            E_FACT2{i} = eye(D(i));
        else
            E_FACT2{i} = E_FACT{i}'*E_FACT{i} + N(i)*eye(D(i));
        end
    end
    E_Lambda{i} = priors{i}.getExpFirstMoment();
    
end


if usr_impute_missing
    % initial imputation
    assert('Handling missing values is untested at the moment.')
    X_recon = nmodel(E_FACT);%,E_CORE); % Just assume a CP model initially.
    X(impute_x) = X_recon(impute_x);
    clear X_recon
else
    %     X(~R_obs)=0; 
end

% Initializes elementwise second moment (possibly with pairwise interactions)
E_FACT2elementpairwise = cell(length(E_FACT),1);
for i = 1:Nx
    if has_missing || hetero_noise_modeling(i)
        E_FACT2elementpairwise{i} = computeUnfoldedMoments(E_FACT{i},{[]});
    elseif my_contains(factor_constraints{i},'sparse','IgnoreCase',true)
        E_FACT2elementpairwise{i} = E_FACT{i}.^2+1;
    else
        E_FACT2elementpairwise{i} = [];
    end
end

% Initialize Factor Contribution to Prior Updates
for i = 1:Nx
    if my_contains(factor_constraints{i},'sparse','IgnoreCase',true)
        E_Contribution2FactorPrior{i} = E_FACT2elementpairwise{i};
    else
        E_Contribution2FactorPrior{i} = E_FACT2{i};
    end
    
end

Etau = tau_alpha0/tau_beta0;
E_log_tau = psi(tau_alpha0)-log(tau_beta0);
noiseType = []; % TODO: Extend GammaNoise.m to handle both CP and Tucker

% Initialize Core Array
if init_nway_tucker
    nway_constr = 2*ones(1,ndims(X)); % unconstrained
    
    nway_constr(my_contains(factor_constraints,'orthogonal','IgnoreCase',true))=0;
    %     nway_constr(my_contains(factor_constraints,'nonneg','IgnoreCase',true))=1;
    %     % nonneg constraint gives missing function "nonneg" error in nway-tucker
    
    if debug_mode
        [E_FACT, E_CORE,~] = tucker(X, D,[],nway_constr);
    else
        [~,E_FACT, E_CORE,~] = evalc('tucker(X, D,[],nway_constr)');
    end
    
    for i =  find(my_contains(factor_constraints,'nonneg','IgnoreCase',true)) % TODO FIND ALTERNATIVE!
        E_FACT{i}=abs(E_FACT{i});
    end
    
    for i = 1:Nx
        if my_contains(factor_constraints{i},'ortho','IgnoreCase',true)
            E_FACT2{i} = eye(D(i));
        else
            E_FACT2{i} = E_FACT{i}'*E_FACT{i} + N(i)*eye(D(i));
        end
    end
    
elseif init_ml_als % 1 step of ML-ALS
    E_CORE = X;
    for i = 1:Nx
        if all_orthogonal % Only accurate for orthogonal factors
            E_CORE=tmult(E_CORE,E_FACT{i}',i);
        else
            E_CORE=tmult(E_CORE,pinv(E_FACT{i}),i);
        end
    end
    
elseif init_using_sampling && only_variational_inference
    if debug_mode
        [E_CORE, E_FACT,E_FACT2, E_Lambda,~,model]=pt_Tucker(X,D,factor_constraints, core_constraint,...
            'Inference','sampling','maxiter',10);
    else
        [~, E_CORE, E_FACT,E_FACT2, E_Lambda,~,model]=evalc(['pt_Tucker(',...
            'X,D,factor_constraints, core_constraint,'...
            '''Inference'',''sampling'',''maxiter'',10)']);
    end
    core_array.core = E_CORE;
    core_array.sigma_covar = model.core.sigma_covar;
    E_CORE_sq = core_array.getExpSecondMoment();
    covCore = core_array.sigma_covar;
    
else % Single VB or Gibbs updating of the core.
    core_array.updateCore(X, [], E_FACT, E_FACT2, Etau);
    E_CORE = core_array.getExpFirstMoment();
    E_CORE_sq = core_array.getExpSecondMoment();
    covCore = core_array.sigma_covar;
end

if init_ml_als || init_nway_tucker
    % Initialize relevant variables..
    % Prior on core %%%% TODO! This should be handled in the core-class
    E_psi_core = (alpha_psi/beta_psi*ones(size(E_CORE)));
    E_log_psi_core = (psi(alpha_psi)-log(beta_psi))*ones(size(E_CORE));
    
    % Update expected second moment of the core.
    sigma_sq_core=(Etau+E_psi_core).^-1;
    
    covCore = diag(sigma_sq_core(:));
    core_array.core = E_CORE;
    core_array.sigma_covar = covCore;
%     E_CORE_sq=E_CORE(:)*E_CORE(:)'+diag(sigma_sq_core(:));
    E_CORE_sq = core_array.getExpSecondMoment();
end


% % Initialize noise
%     % Homoscedastic noise
%     noiseType = GammaNoise(X,R_obs, tau_alpha0, tau_beta0, noise_inference);
%     % Initialize noise specified by user (if provided)
%     if ~isempty(init_noise)
%         if isobject(init_noise) && strcmp(class(noiseType), class(init_noise))
%             noiseType = init_noise;
%
%         elseif isvector(init_noise) && ~isobject(init_noise)
%             noiseType.noise_value = init_noise;
%             noiseType.noise_log_value = log(init_noise);
%
%         end
%     end
%     Etau = noiseType.getExpFirstMoment();
%     SST = noiseType.getSST();
% end
SST = sum(X(:).^2);

if debug_mode
    X_recon = nmodel(E_FACT, E_CORE);
    fprintf('Initial sol: %6.4e\n',rms(X(:)-X_recon(:)))
end

%% Initialize variables for storing Gibbs samples
initialize_sample_storage = true;
if some_variables_are_sampled && initialize_sample_storage
    for i = 1:Nx
        if strcmpi(inference_scheme{i,1},'sampling')
            all_samples{i,1}(:,:,maxiter) = zeros(size(E_FACT{i}));
        end

        if strcmpi(inference_scheme{i,2},'sampling') && ~isempty(E_Lambda{i})
            all_samples{i,2}(:,:,maxiter) = zeros(size(E_Lambda{i})); % precision prior
        end
    end
        
    if strcmpi(core_inference_scheme,'sampling')
        all_samples{end-1,1}(:,maxiter) = zeros(numel(E_CORE),1); % core-array
        all_samples{end-1,2}(:,maxiter) = zeros(numel(core_prior.getExpFirstMoment()),1); % prior on core array
    end

    %  all_samples{end,1}(iter) = Etau; % noise % TODO: Only VB atm..

end



%% Setup how information should be displayed.
disp([' '])
disp(['Probabilistic Tucker Decomposition'])        %TODO: Better information about what is running..
disp(['A D=[' num2str(D) '] component model will be fitted.']);
fprintf(core_array.getSummary()); fprintf('\n')
for i=1:Nx, fprintf(factors{i}.getSummary()); fprintf('\n'); end

disp([' '])
disp(['To stop algorithm press control C'])
disp([' ']);
if debug_mode
    dheader = sprintf(' %16s | %16s | %16s | %16s | %16s | %16s | %16s || %16s | %16s | %16s | %16s | %16s | %16s | %16s | rmse, corr, var(X), var(recon)', ...
        'core','A1','A2','A3','A1sq','A2sq','A3sq',...
        'Iteration','Cost', 'rel. \Delta cost', 'Noise (s.d.)', 'Var. Expl.', 'Time (sec.)', 'Time CPU (sec.)');
else
    dheader = sprintf(' %16s | %16s | %16s | %16s | %16s | %16s | %16s |', ...
        'Iteration','Cost', 'rel. \Delta cost', 'Noise (s.d.)', 'Var. Expl.', 'Time (sec.)', 'Time CPU (sec.)');
end
dline = repmat('------------------+',1,7);

%% Start Inference Procedure (Optimization, Sampling, or both)
iter = 0;
old_cost = -inf;
delta_cost = inf;
Lowerbound = zeros(maxiter,1,'like', SST);
rmse = zeros(maxiter,1,'like', SST);
Evarexpl = zeros(maxiter,1,'like', SST);
total_realtime = tic;
total_cputime = cputime;

fprintf('%s\n%s\n%s\n',dline, dheader, dline);
R = matricizing(R_obs,1);

if debug_mode
    fprintf(' %16.4e | %16.4e | %16.4e | %16.4e | %16.4e | %16.4e | %16.4e \n',...
        norm(E_CORE(:)), norm(E_FACT{1}), norm(E_FACT{2}), norm(E_FACT{3}), norm(E_FACT2{1}), norm(E_FACT2{2}), norm(E_FACT2{3}));
end
while  delta_cost>=conv_crit && iter<maxiter || ... %
        ((iter <= fixed_lambda) && model_lambda) ...
        || ((iter <= fixed_tau) && model_tau)  ...
        || ((iter <= fixed_psi) && update_core_prior) ...
        || ((iter <= fixed_core) && update_core_array)
    
    iter = iter + 1;
    time_tic_toc = tic;
    time_cpu = cputime;
    
    %% Update factor matrices (one at a time)
    for i = 1:Nx
        
        % Update factor distribution parameters
        if  isempty(dont_update_factor) || ~dont_update_factor(i)
            factors{i}.updateFactor(i, X, R, E_CORE, covCore, E_FACT, E_FACT2, E_FACT2elementpairwise, Etau)
        end
        
        % Get the expected value (sufficient statistics)
        if hetero_noise_modeling(i)
            E_FACT{i} = factors{i}.getExpFirstMomefnt(Etau{i});
            E_FACT2{i} = factors{i}.getExpSecondMoment(Etau{i});
        else
            E_FACT{i} = factors{i}.getExpFirstMoment();
            E_FACT2{i} = factors{i}.getExpSecondMoment();
        end
        E_Contribution2FactorPrior{i} = factors{i}.getContribution2Prior();
        
        % Calculate the expected second moment between pairs (missing,
        % noise modeling) or each element (sparsity) or not at all.
        %         if has_missing || hetero_noise_modeling(i) % add or all factors are univariate (for space)
        %             E_FACT2elementpairwise{i} = factors{i}.getExpElementwiseSecondMoment();
        %         elseif my_contains(factor_constraints{i},'sparse','IgnoreCase',true)
        %             E_FACT2elementpairwise{i} = factors{i}.getExpSecondMomentElem();
        %         else
        %             E_FACT2elementpairwise{i} = [];
        %         end
        
        % Updates local (factor specific) prior (hyperparameters)
        if model_lambda && iter > fixed_lambda && ~shares_prior(i)...
                && (isempty(dont_update_factor) || ~ dont_update_factor(i))
            factors{i}.updateFactorPrior(E_Contribution2FactorPrior{i})
            E_Lambda{i} = priors{i}.getExpFirstMoment();
        end
        
    end
    
    % Calculate ELBO/cost contribution
    cost = 0;
    for i = 1:Nx
        if ~shares_prior(i)
            cost = cost + factors{i}.calcCost()...
                + priors{i}.calcCost();
            assert(isreal(cost))
        end
    end
    
    %% Updates shared priors (hyperparameters)
    if any(shares_prior)
        for j = unique(nonzeros(shares_prior(:)))'
            shared_idx = find(shares_prior == j)';
            fsi = shared_idx(1);
            
            if model_lambda && iter > fixed_lambda && ...
                    (isempty(dont_update_factor) ||...
                    all(~dont_update_factor(shared_idx)))
                % Update shared prior
                distr_constr = 0.5*ones(length(shared_idx),1); % Normal or truncated normal
                distr_constr(my_contains(factor_constraints(shared_idx),'expo','IgnoreCase',true)) = 1;
                factors{fsi}.updateFactorPrior(E_Contribution2FactorPrior(shared_idx), distr_constr);
            end
            
            %Get cost for the shared prior and each factor.
            cost = cost + priors{fsi}.calcCost();
            for f = shared_idx
                cost = cost+factors{f}.calcCost();
                E_Lambda{f} = priors{f}.getExpFirstMoment();
                
                assert(all(all(E_Lambda{fsi} == E_Lambda{f})), 'Error, shared prior is not shared');
            end
        end
    end
    
    %% Update core-array G
    if update_core_array && iter > fixed_core
        core_array.updateCore(X, R, E_FACT, E_FACT2, Etau);
    end
    E_CORE = core_array.getExpFirstMoment();
    E_CORE_sq = core_array.getExpSecondMoment();
    covCore = core_array.sigma_covar;
    
    if update_core_prior && iter > fixed_psi
        core_array.updateCorePrior(E_CORE_sq,0.5);
    end
    E_psi_core = core_prior.getExpFirstMoment();
    E_log_psi_core = core_prior.getExpLogMoment();
    
    cost = cost + core_prior.calcCost() + core_array.calcCost();
%     core_array.hyperparameter.calcCost()
    %% Update missing imputation
    % TODO:
    if usr_impute_missing
        % Impute new values
        X_recon = nmodel(E_FACT, E_CORE);  % TODO: Dont reconstruct explicitly, if few values.
        X(impute_x) = X_recon(impute_x);
        clear X_recon
        
        % Technically not corrected for VB as <x_miss.^2> is wrong (missing variance term..)
        if ~any(hetero_noise_modeling)
            SST = sum(X(:).^2);
            %             noiseType.SST = SST;
        end
        
    end
    
    %% Update noise precision
    % TODO: Extend "GammaNoise.m" to work for both CP and Tucker.
    
    % Calculate residual error
    E_Rec=nmodel(E_FACT,E_CORE);
    if all_orthogonal
        SSE = SST + trace(E_CORE_sq)-2 *X(:)'*E_Rec(:)  ;
        
    else
        EPtP = 1;
        for i = Nx:-1:1
            EPtP = kron(EPtP,E_FACT2{i});
        end
        SSE = SST + sum(sum( E_CORE_sq .* EPtP)) -2 *X(:)'*E_Rec(:);
%         SSE = SST + sum(sum( (E_CORE(:)*E_CORE(:)'+covCore) .* EPtP)) -2 *X(:)'*E_Rec(:);
    end
    
    if debug_mode
        fprintf(' %16.4e | %16.4e | %16.4e | %16.4e | %16.4e | %16.4e | %16.4e |',...
            norm(E_CORE(:)), norm(E_FACT{1}), norm(E_FACT{2}), norm(E_FACT{3}), norm(E_FACT2{1}), norm(E_FACT2{2}), norm(E_FACT2{3}));
    end
    
    assert(SSE>=0,'SSE was negative!')
    % Update noise precision (tau)
    if model_tau && iter > fixed_tau
        est_tau_alpha= tau_alpha0+prod(N)/2;
        est_tau_beta = tau_beta0+SSE/2;
        Etau=est_tau_alpha/est_tau_beta;
        E_log_tau=psi(est_tau_alpha)-log(est_tau_beta);
    else
        est_tau_alpha = tau_alpha0;
        est_tau_beta = tau_beta0;
    end
    
    [H_tau, prior_tau]=entropy_gamma(est_tau_alpha, ...
        est_tau_beta, tau_alpha0, tau_beta0);
    
    cost = cost ... % from factors and their priors
        -prod(N)/2*log(2*pi)+prod(N)/2*E_log_tau-0.5*SSE*Etau ...
        +prior_tau+H_tau ;
    
    assert(isreal(cost), 'Cost was not a real number!')
    delta_cost = (cost-old_cost)/abs(cost);
    
    Lowerbound(iter) = cost;
    rmse(iter) = sum(SSE);
    Evarexpl(iter) = 1-sum(SSE)/sum(SST);
    old_cost = cost;
    
    %% Display
    if mod(iter,100)==0
        fprintf('%s\n%s\n%s\n',dline, dheader, dline);
    end
    
    if mod(iter,1)==0
        time_tic_toc = toc(time_tic_toc);
        time_cpu = cputime-time_cpu;
        
        if any(hetero_noise_modeling)
            % TODO
        else
            if debug_mode
                fprintf(' %16i | %16.4e | %16.4e | %16.4e | %16.4f | %16.4f | %16.4f |%16.4f |%16.4f |%16.4f |%16.4f |\n',...
                    iter, cost, delta_cost, sqrt(1./Etau), Evarexpl(iter), time_tic_toc, time_cpu, rms(X(:)-E_Rec(:)),corr(X(:),E_Rec(:)),var(X(:)),var(E_Rec(:)));
            else
                fprintf(' %16i | %16.4e | %16.4e | %16.4e | %16.4f | %16.4f | %16.4f |\n',...
                    iter, cost, delta_cost, sqrt(1./Etau), Evarexpl(iter), time_tic_toc, time_cpu);
            end
        end
    end
    
    %% Check update was valid (Only works if ALL inference is variational)
    % delta_cost is only garanteed to be positive for variational
    % inference, so for sampling, it should just keep going until the
    % change is low enough.
    if delta_cost <0 && any_othogonal_factors && iter <= 5
        warning('Lowerbound decreased during the MAP updating of the orthogonal factors.')
        delta_cost = max(abs(delta_cost),conv_crit*10); % Ensure the algorithm doesnt terminate..
    end
        
    
    if only_variational_inference && ~usr_impute_missing 
        if iter > 1 % Due to MAP estimate of orthogonal factors..
            [~, c_fixed_psi, c_fixed_lambda, c_fixed_tau] = isElboDiverging(...
                iter, delta_cost, conv_crit,...
                {'hyperprior (core)', update_core_prior, fixed_psi},...
                {'hyperprior (lambda)', model_lambda, fixed_lambda},...
                {'noise (tau)', model_tau, fixed_tau});
            fixed_psi = min(fixed_psi, c_fixed_psi);
            fixed_lambda = min(fixed_lambda, c_fixed_lambda);
            fixed_tau = min(fixed_tau, c_fixed_tau);
        end
    else
        delta_cost = abs(delta_cost);
    end
    
    %% Save samples
    if some_variables_are_sampled
                for i = 1:Nx
                    if strcmpi(inference_scheme{i,1},'sampling')
                        all_samples{i,1}(:,:,iter) = E_FACT{i}; % factor
                    end
        
                    if strcmpi(inference_scheme{i,2},'sampling') && ~isempty(E_Lambda{i})
                        all_samples{i,2}(:,:,iter) = E_Lambda{i}; % precision prior
                    end
                end
                
                if strcmpi(core_inference_scheme,'sampling')
                    all_samples{end-1,1}(:,iter) = E_CORE(:); % core-array
                    all_samples{end-1,2}(:,iter) = E_psi_core(:); % prior on core array
                end
                
%                  all_samples{end,1}(itedr) = Etau; % noise
    end
    
    
end
total_realtime = toc(total_realtime);
total_cputime = cputime-total_cputime;

fprintf('Total realtime: %6.4f sec.\nTotal CPU time: %6.4f sec.\n',total_realtime,total_cputime)

%% Algorithm completed, now finalize function output.
Lowerbound(iter+1:end)=[];

% if nargout > 3 % TODO: Investigate if evalc('') does not provide nargout.
model.factors = factors;
model.priors = priors;
model.core = core_array;
%    model.noise = noiseType;
model.noise = Etau;
model.cost = Lowerbound;
model.rmse = rmse;
model.varexpl = Evarexpl;
model.SST = SST;
model.realtime = total_realtime;
model.cputime = total_cputime;

if any(hetero_noise_modeling)
    for i = 1:Nx
        E_FACT{i} = factors{i}.getExpFirstMoment();
        E_FACT2{i} = factors{i}.getExpSecondMoment();
    end
end

fprintf('Algorithm ran to completion\n')
end

function [H_gamma, prior_gamma]=entropy_gamma(a,b,a0,b0)

H_gamma=-log(b)+a-(a-1).*psi(a)+gammaln(a);
prior_gamma=-gammaln(a0)+a0*log(b0)+(a0-1)*(psi(a)-log(b))-b0*a./b;


end