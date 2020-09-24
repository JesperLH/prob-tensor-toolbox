function [EA,P,elbo] = pt_PARAFAC2(X,D,cp_constraints,varargin)
%%PT_PARAFAC2 Fits a Bayesian PARAFAC2 model to N-way data with one varying
%%mode. The parameters are inferred using Gibbs sampling or Variational
%%inference.
% !!!! WIP: This function is still under development and might/will not work !!!!
%
% INPUTS:
%   X:  The data either as an array or cell array, the later if the varying
%       mode have different sizes.
%   D:  The number of components to fit.
%   
%
%
% OUTPUTS:
%   EA: Expected value of the factor matrices
%   P:  Expected value of the Orthogonal matrices
%   elbo: Evidence lowerbound (VB) or loglikelihood (Gibbs)


disp_cp_step = false;
maxiter_cp = 5; %#ok<NASGU> % Number of iterations of the CP step.


%% Read input and Setup Parameters
paramNames = {'conv_crit', 'maxiter' ...
    'model_tau', 'fixed_tau', 'tau_a0', 'tau_b0',...
    'model_lambda', 'fixed_lambda', 'lambda_a0', 'lambda_b0'...
    'init_method','inference'};
defaults = {1e-6, 10, ...
    true , 5  , 1, 1e-4,...
    true , 2 , 1, 1e-4,...
	0, 'variational',false(ndims(X),1), [], false,[], [], []};
% Initialize variables to default value or value given in varargin
[conv_crit, maxiter, ...
    model_tau, fixed_tau, tau_alpha0, tau_beta0,...
    model_lambda, fixed_lambda, lambda_alpha0, lambda_beta0,...
    init_method, inference_scheme]...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});


%% If an array is passed, change it into a cell array
if ~iscell(X)
   sizeX = size(X);
   Xtmp = permute(X,[ndims(X), 1:ndims(X)-1]);
   X = cell(sizeX(end),1);
   for k = 1:sizeX(end)
       X{k} = reshape(Xtmp(k,:),sizeX(1:end-1));
   end
   clear Xtmp
    
end

n_fixed = size(X{1});
n_fixed(end) = [];
n_samples = length(X);

n_vary = cellfun(@(t) size(t,ndims(t)), X); % num. elem. in last dim.


%% Prior distributions
if nargin < 3 || isempty(cp_constraints)
    cp_constraints = repmat("normal ard",1,ndims(X{1})+1); 
else
    assert((ndims(X{1})+1) == length(cp_constraints),...
        'Constraints must be specified for the fixed factor matrices, C and F.')
end
% TODO: Make elbo roboust to varying constraints!
% TODO: Ensure the funny mode is alway unconstrained.. ? Why?

% TODO: How to specify priors? and remember to call it in pt_CP
F_prior = 1e-3;
C_prior = eye(D);%*1e-3; % Maybe a diagonal for each K... 
prior_tau_a=1;
prior_tau_b=1e+3;
has_missing = false; %Note, missing values is not supported.

SST = sum(cellfun(@(t) sum(t(:).^2), X));
%% TODO: Initialization schemes
% - Providing varying mode factors for init.
% - Providing already estimated factors

% Initialization by random
for n=length(n_fixed):-1:1
    EA{n} = randn(n_fixed(n),D);
end
C = randn(n_samples,D); % diagonal matrices
F = randn(D,D);

% initialization by nway (3-way data only)
% opt_nway(2) = 10;
% opt_nway(5) = 2;
% [A,F,C,P,fit,AddiOutput]=parafac2(X,D,[0,0],opt_nway);



% Initialize noise (TODO)
Etau=tau_alpha0./tau_beta0;% 1./(SST/prod(n_fixed)*sum(n_vary));
Elog_tau = log(Etau);

% Initial values 
cp_model.factors = [EA,F,C];
cp_model.noise = Etau;

% Shifting mode
factorPk = cell(n_samples,1);
for k=n_samples:-1:1
    factorPk{k} = OrthogonalFactorMatrix([n_vary(k),D],'variational');
end

%% Run iterative update scheme
% fit_old = -inf;
fit_new = -inf;
elbo_cp = -inf;
delta_cost = 1;
iter = 0;
while iter < maxiter && (iter == 1 || delta_cost > conv_crit)
    iter = iter+1;
    fit_old = fit_new;
    
    %% Update Pk
    Akr = ones(1,D);
    for n=length(n_fixed):-1:1 
        Akr = krprod(Akr,EA{n});
    end
%     for n=1:length(n_fixed)
%         Akr = krprod(EA{n},Akr); % Same as pCP, for dev purpose
%     end
    for k=n_samples:-1:1
    % Update the von Mises-Fisher Distribution
    factorPk{k}.updateFactor(1, matricizing(X{k},ndims(X{k})),[],[],[],...
        {ones(1,D),Akr*diag(C(k,:))*F'},[],[],Etau)
    P{k} = factorPk{k}.getExpFirstMoment();

    % Data projection #TODO: No need to matricize X all the time..
    Y(:,:,k) = matricizing(X{k},ndims(X{k}))'*P{k};
    end
   
    % Reshape into N-way array
    if length(n_fixed)>1
        Y = reshape(Y,[n_fixed,D,n_samples]);
    end
    
    %% Update A, C, F using PARAFAC
    elbo_cp_old = elbo_cp(end);
    if disp_cp_step
        [EA, EAtA, ELambda, elbo_cp,cp_model] = VB_CP_ALS(Y,D,cp_constraints,...
        'maxiter',maxiter_cp,'inference','variational','model_tau',false,...
        'model_lambda',model_lambda,'fixed_lambda',fixed_lambda,...
        'init_factors',cp_model.factors,... % TODO [A,H,C]?... also second moment! (incl. in factor?)
        'init_noise',cp_model.noise); % TODO Etau class!
    
    else
    
        [~,EA, EAtA, ELambda, elbo_cp,cp_model] = evalc(['VB_CP_ALS(Y,D,cp_constraints,',...
        '''maxiter'',maxiter_cp,''inference'',''variational'',''model_tau'',false,',...
        '''model_lambda'',model_lambda,''fixed_lambda'',fixed_lambda,',...
        '''init_factors'',cp_model.factors,',... 
        '''init_noise'',cp_model.noise);']);
    end
    % TODO: This check doesn't make sense, as Y is different from iter to
    % iter, so the elbo can go either way or what?
    fprintf('Elbo CP: (%6.4e, %6.4e, %6.4e)\t',elbo_cp_old,elbo_cp(end), (elbo_cp(end)-elbo_cp_old)/abs(elbo_cp_old))
    
    Akr = ones(1,D);
    AtA = ones(D,D);
    for n=length(n_fixed):-1:1 % TODO: Fix index..
        Akr = krprod(Akr,EA{n});
        AtA = AtA .* EAtA{n};
    end
%     for n=1:length(n_fixed)
%         Akr = krprod(EA{n},Akr); % Same as pCP, for dev purpose
%     end
    
    % Extract F and C. #TODO: Is this necessary?
    F = EA{length(n_fixed)+1};
    FtF = EAtA{length(n_fixed)+1};
    C = EA{length(n_fixed)+2};
    CtC = EAtA{length(n_fixed)+2};
    
    if length(n_fixed)>1
        Y = reshape(Y,[prod(n_fixed),D,n_samples]);
    end
    %% Update Tau
    % Calculate fit
    S3 = 0;
    for k=1:n_samples
       S3 = S3 + sum( (Y(:,:,k)*F) .* Akr,1)*C(k,:)';
    end
    S2 = sum(sum( AtA .* FtF .* CtC));
    SSE = SST + S2 -2*S3;
    
    % Update tau parameters
    cp_model.noise.number_of_elements = sum(cellfun(@numel,X));
    if model_tau && iter > fixed_tau
        cp_model.noise.est_alpha = cp_model.noise.hp_alpha + sum(cellfun(@numel,X))/2;
        cp_model.noise.est_beta = cp_model.noise.hp_beta + SSE/2;
        
        % NB. noise_value and noise_log_value are not updated unless
        % updateNoise() is called. This is done to keep the sample during
        % an iteration.. This leads to some ugly code.. 
        % #TODO: Find a nicer solution and ensure this is not a problem
        % elsewhere...
        cp_model.noise.noise_value = cp_model.noise.est_alpha ./ cp_model.noise.est_beta;
        cp_model.noise.noise_log_value = psi(cp_model.noise.est_alpha)-log(cp_model.noise.est_beta);
        Etau = cp_model.noise.getExpFirstMoment();
        Elog_tau = cp_model.noise.getExpLogMoment();
    else
        % Not pretty... TODO: Nicecify..
        cp_model.noise.est_alpha = cp_model.noise.hp_alpha;
        cp_model.noise.est_beta = cp_model.noise.hp_beta;
        cp_model.noise.noise_value = cp_model.noise.hp_alpha ./ cp_model.noise.hp_beta;
        cp_model.noise.noise_log_value = psi(cp_model.noise.hp_alpha)-log(cp_model.noise.hp_beta);
        Etau = cp_model.noise.getExpFirstMoment();
        Elog_tau = cp_model.noise.getExpLogMoment();
    end
    
    %% Calculate loglikelihood (sampling) or evidence lowerbound (VB)
    elbo = 0;
    % Contribution from the projection matrices
    for k = 1:n_samples
        elbo = elbo + factorPk{k}.calcCost();
    end
    % Contribution from the factor matrices
    for n = 1:ndims(Y)
        elbo = elbo + cp_model.factors{n}.calcCost();
    end
    % Contribution from the second level prior distributions
    for n = find(cp_model.shares_prior(:)==0)' % Unconnected priors
        if ~isempty(n)
            elbo = elbo + cp_model.factors{n}.hyperparameter.calcCost(); 
        end
    end
    for j = unique(nonzeros(cp_model.shares_prior(:)))' % shared priors
        shared_idx = find(cp_model.shares_prior == j)';
        fsi = shared_idx(1);
        elbo = elbo + cp_model.factors{fsi}.hyperparameter.calcCost(); 
    end
    
      
    % Hmm.. cp_model.noise.calcCost() also includes the likelihood term!
    %elbo = elbo + sum(cellfun(@numel, X))/2 * (-log(2*pi) + Elog_tau) -1/2*Etau*SSE;
    cp_model.noise.SST=SST;
    cp_model.noise.SSE=SSE;
    cp_model.noise.noise_value = Etau;
    cp_model.noise.noise_log_value = Elog_tau;
    cp_model.noise.number_of_elements = sum(cellfun(@numel, X));
    elbo = elbo +  cp_model.noise.calcCost();

    fit_new = elbo;
    delta_cost = (fit_new-fit_old)/abs(fit_old);
    fprintf('Iter: %04i \tfit: %6.4e \tdelta(fit): %6.4e \t VarExpl: %4.3f\n',...
        iter,fit_new,delta_cost,1-SSE/SST)
    
    if iter>1
        isElboDiverging(iter,delta_cost,conv_crit);
    end
    1+1;
    
end

% TODO: (Re)order stuff for return values

end

