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





%% Read input and Setup Parameters
paramNames = {'conv_crit', 'maxiter' ...
    'model_tau', 'fixed_tau', 'tau_a0', 'tau_b0',...
    'model_lambda', 'fixed_lambda', 'lambda_a0', 'lambda_b0'...
    'init_method','inference'};
defaults = {1e-6, 10, ...
    true , 20  , 1, 10,...
    true , 10 , 1, 10,...
	'', 'variational'};
% Initialize variables to default value or value given in varargin
[conv_crit, maxiter, ...
    model_tau, fixed_tau, tau_alpha0, tau_beta0,...
    model_lambda, fixed_lambda, lambda_alpha0, lambda_beta0,...
    init_method, inference_scheme]...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});

conv_crit_cp = conv_crit;
inference_scheme_cp = inference_scheme;
only_variational_inference = strcmpi(inference_scheme,'variational');
model_tau_cp = false;
disp_cp_step = false;
maxiter_cp = 5; %#ok<NASGU> % Number of iterations of the CP step.

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
    cp_constraints{end-1} = 'normal constr'; % Prior on H/F
else
    assert((ndims(X{1})+1) == length(cp_constraints),...
        'Constraints must be specified for the fixed factor matrices, C and F.')
end
% #TODO: Restrictions on how priors are specified?
has_missing = false; %Note, missing values is not supported.

SST = sum(cellfun(@(t) sum(t(:).^2), X));
%% TODO: Initialization schemes
% - Providing varying mode factors for init.
% - Providing already estimated factors

% Initialize noise (TODO)
Etau= tau_alpha0./tau_beta0; %  1./(SST/prod(n_fixed)*sum(n_vary)); %
Elog_tau = log(Etau);

% Shifting mode
factorPk = cell(n_samples,1);
for k=n_samples:-1:1
    factorPk{k} = OrthogonalFactorMatrix([n_vary(k),D],'variational');
    [U,~,~]=svd(matricizing(X{k},ndims(X{k})),'econ');
    P{k}=U(:,1:D);%+0.01*randn(size(X{k},n_vary(k)),D);

    Y(:,:,k) = matricizing(X{k},ndims(X{k}))'*P{k};
end


if strcmpi(init_method,'random')
    % Initialization by random
    for n=length(n_fixed):-1:1
        EA{n} = rand(n_fixed(n),D);
    end
    C = rand(n_samples,D); % diagonal matrices
    F = rand(D,D);
    % Initial values 
    cp_model.factors = [EA,F,C];
    cp_model.noise = Etau;
    EA{end+1} = F;
    EA{end+2} = C;
    
elseif strcmpi(init_method,'ml')

% initialization by nway (3-way data only)
% opt_nway(2) = 10;
% opt_nway(5) = 2;
% [A,F,C,P,fit,AddiOutput]=parafac2(X,D,[0,0],opt_nway);



else
    % Initialization by pt_CP
    % Reshape into N-way array
    if length(n_fixed)>1
        Y = reshape(Y,[n_fixed,D,n_samples]);
    end

    % Update A, C, F using PARAFAC
    [EA, EAtA, ELambda, elbo_cp,cp_model] = pt_CP(Y,D,cp_constraints,...
    'maxiter',5,'inference','sampling',...inference_scheme_cp,...
    'model_tau',false,...
    'model_lambda',false, ...model_lambda,'fixed_lambda',fixed_lambda,...'init_factors',cp_model.factors,... 'init_noise',cp_model.noise,...
    'tau_a0',tau_alpha0, 'tau_b0', tau_beta0,...
    'lambda_a0',lambda_alpha0,'lambda_b0', lambda_beta0);

    %!NB! If initialized by another inference scheme than
    %"inference_scheme_cp", then we need to change that later on.
    cp_model.noise.inference_method = inference_scheme_cp;
    for n = 1:length(cp_model.factors)
        cp_model.factors{n}.inference_method = inference_scheme_cp;
        if ~isempty(cp_model.priors{n}.inference_method )
            cp_model.priors{n}.inference_method = inference_scheme_cp;
        end
    end

    % Number of elements and SST are from the PARAFAC2 model (Only needed for
    % tau.. which we update separately anyway..
    % Hmm... if SST and num. elem. arent passed, then the call to CP does not
    % necessarily decrease between iterations (as Y|A,lambda,tau is different
    % from X|A,lambda,tau).
    % cp_model.noise.number_of_elements = sum(cellfun(@numel,X));
    % cp_model.noise.SST = SST;

    F = EA{length(n_fixed)+1};
    C = EA{length(n_fixed)+2};
    if length(n_fixed)>1
        Y = reshape(Y,[prod(n_fixed),D,n_samples]);
    end

end

%%
disp([' '])
disp(['Bayesian PARAFAC2 Decomposition'])
disp(['A ' num2str(D) ' component model will be fitted'])
dheader = sprintf(' %12s | %12s | %12s | %12s | %12s | %12s | %12s ',...
    'Iteration','ELBO.','dELBO','E[Var. Expl]','Time','E_tau', '||Y||');
dline = repmat('--------------+',1,7);

disp(dline); disp(dheader); disp(dline);
fprintf(' %12i | %12.4e | %12.4e | %12.4f | %12.4f | %12.4e | %12.4e |\n',...
    0, nan,nan,nan,nan, Etau, norm(Y(:)));

%% Run iterative update scheme
% fit_old = -inf;
fit_new = -inf;
delta_cost = 1;
iter = 0;
tic_iter = tic;
while iter < maxiter && (iter == 1 || delta_cost > conv_crit) ...
        || (model_tau && iter <= fixed_tau) ...
        || (model_lambda && iter <= fixed_lambda)
    iter = iter+1;
    fit_old = fit_new;
    
    %% Update Pk
    Akr = ones(1,D);
    for n=length(n_fixed):-1:1 
        Akr = krprod(Akr,EA{n});
    end
    
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

    %% Update A, C, F using probabilistic PARAFAC
    % Note, tau is not update here, as the summary statistics are wrong.
    % Tau is updated afterwards to get the correct summary stats.
    % #TODO: Maybe it is only the number of elements that is wrong? Also,
    % if that is the case, then Jørgensen et al. heteroscedastic slab noise
    % is just HCP on the concentration mode. 

    fixed_lambda_cp = max(fixed_lambda-iter, 0);
    fixed_tau_cp = max(fixed_tau-iter,0);
    try
        if disp_cp_step
            [EA, EAtA, ELambda, elbo_cp,cp_model] = pt_CP(Y,D,cp_constraints,...
            'maxiter',maxiter_cp,'inference',inference_scheme_cp,...
            'model_tau',model_tau_cp,'fixed_tau',fixed_tau_cp, ...
            'model_lambda',model_lambda,'fixed_lambda',fixed_lambda_cp,...
            'init_factors',cp_model.factors,... 
            'init_noise',cp_model.noise,'conv_crit',conv_crit_cp); 

        else

            [~,EA, EAtA, ELambda, elbo_cp,cp_model] = evalc(['pt_CP(Y,D,cp_constraints,',...
            '''maxiter'',maxiter_cp,''inference'',inference_scheme_cp,',...
            '''model_tau'',model_tau_cp,''fixed_tau'',fixed_tau_cp,',...
            '''model_lambda'',model_lambda,''fixed_lambda'',fixed_lambda_cp,',...
            '''init_factors'',cp_model.factors,''conv_crit'',conv_crit_cp,',... 
            '''init_noise'',cp_model.noise);']);
        end

    catch ME
       fprintf('CP step diverged/failed!  \n')
       
    end
        
    % Extract F and C. #TODO: Is this necessary?
    F = EA{length(n_fixed)+1}; % #TODO: Do we get F or F'?
    FtF = EAtA{length(n_fixed)+1};
    C = EA{length(n_fixed)+2};
    CtC = EAtA{length(n_fixed)+2};
%     EA(end-1:end) = [];
%     EAtA(end-1:end) = [];
    
    Akr = ones(1,D);
    AtA = ones(D,D);
    for n=length(n_fixed):-1:1 % TODO: Fix index..
        Akr = krprod(Akr,EA{n});
        AtA = AtA .* EAtA{n};
    end
    
    
    
    if length(n_fixed)>1
        Y = reshape(Y,[prod(n_fixed),D,n_samples]);
    end
    %% Update Tau
    % Calculate fit
    S3 = 0;
    for k=1:n_samples
       S3 = S3 + sum( (Y(:,:,k)*F) .* Akr,1)*C(k,:)'; %#TODO: Tripple tjeck
    end
    S2 = sum(sum( AtA .* FtF .* CtC));
    SSE = SST + S2 -2*S3;
    assert(SSE>=0)
    
    % Update tau parameters
    if model_tau && iter > fixed_tau
        cp_model.noise.est_alpha = cp_model.noise.hp_alpha + sum(cellfun(@numel,X))/2;
        cp_model.noise.est_beta = cp_model.noise.hp_beta + SSE/2;
    else
        % No change in tau
        cp_model.noise.est_alpha = cp_model.noise.hp_alpha;
        cp_model.noise.est_beta = cp_model.noise.hp_beta;
        
    end
    cp_model.noise.noise_value = cp_model.noise.est_alpha ./ cp_model.noise.est_beta;
    cp_model.noise.noise_log_value = psi(cp_model.noise.est_alpha)-log(cp_model.noise.est_beta);
    
    Etau = cp_model.noise.getExpFirstMoment();
    Elog_tau = cp_model.noise.getExpLogMoment();
    
    [H_gamma, prior_gamma]=entropy_gamma(cp_model.noise.est_alpha,cp_model.noise.est_beta,cp_model.noise.hp_alpha,cp_model.noise.hp_beta);
    
    
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
    
      
    % Contribution from noise precision prior
    elbo_tau = H_gamma + prior_gamma; % P and Q of tau
    
    % Contribution from likelihood term
    elbo_ll = sum(cellfun(@numel, X))/2 * (-log(2*pi) + Elog_tau) -1/2*Etau*SSE;
    
    % Evidence lowerbound (variational inf.) or log-likelihood (Gibbs sampling)
    elbo = elbo + elbo_ll + elbo_tau;  

    fit_new = elbo;
    delta_cost = (fit_new-fit_old)/abs(fit_old);
%     fprintf('Iter: %04i \tfit: %6.4e \tdelta(fit): %6.4e \t VarExpl: %4.3f\n',...
%         iter,fit_new,delta_cost,1-SSE/SST)
    
    if iter > 5
        assert((1-SSE/SST)>0,'E[Var. Expl.] is negative... Bad initial solution?')
    end
    %% Display
    if rem(iter,1)==0
        d_time = toc(tic_iter);
        fprintf(' %12i | %12.4e | %12.4e | %12.4f | %12.4f | %12.4e | %12.4e |\n',...
            iter, elbo, delta_cost,1-SSE/SST , d_time, Etau, norm(Y(:)));
        tic_iter=tic;
    end
    
    %% Check
    if iter>1 && only_variational_inference
        isElboDiverging(iter,delta_cost,conv_crit);
%         [~, c_fixed_lambda, c_fixed_tau] = isElboDiverging(...
%             iter, delta_cost, conv_crit,...
%             {'hyperprior PF2 (lambda)', model_lambda, fixed_lambda},...
%             {'noise PF2 (tau)', model_tau, fixed_tau});
%         fixed_lambda = min(fixed_lambda, c_fixed_lambda);
%         fixed_tau = min(fixed_tau, c_fixed_tau);
    elseif iter > 1
        delta_cost = abs(delta_cost);
    end
    1+1;
    
end

% TODO: (Re)order stuff for return values
1+1;
end

function [H_gamma, prior_gamma]=entropy_gamma(a,b,a0,b0)

H_gamma=-log(b)+a-(a-1).*psi(a)+gammaln(a);
prior_gamma=-gammaln(a0)+a0*log(b0)+(a0-1)*(psi(a)-log(b))-b0*(a./b);


end
