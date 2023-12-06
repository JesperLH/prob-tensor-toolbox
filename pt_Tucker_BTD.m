function [E_G, E_U, ELBO] = pt_Tucker_BTD(X, D, core_prior_type) %, E_G_init)
%% PT_TUCKER_BTD fits a Tucker decomposition model to X. The number of
% components in each mode are determined by D. If a matrix is input, then
% a block-diagonal structure is assumed (rows are blocks and columns are
% number of modes). The factors are orthogonal (matrix von Mises-Fisher)
% and core elements real valued (normal distribution).
%
% INPUT:
%   X:      The N-dimensional data array
%   D:      The number of components in each mode (columns) for each block
%           (rows). NB. A row vector is Tucker decomposition.
%   core_prior_type:    'scale' or 'sparse' (default) prior on the core arrays 
%
% OUTPUT:
%   E_G:    Expected value of the core array
%   E_U:    Expected value of the factors
%   ELBO:   Evidence lowerbound (Variational Bayes)


% Model:
% X_{i1,i2,...,in}=\sum_j1,j2,...,jn Core_j1,j2...,jn*U_{i1,j1}*U_{i2,j2}*...*U_{in,jn}
%
% Priors
%  U_p ~ VMF(0)
%  G_lmn ~ N(0,lambda_lmn^-1)
%  lambda_lmn ~ Gamma(\alpha,\beta)...
%  tau ~ Gamma(a,b)
% Q-distribution:
%  Loadings:  vMF(F_,k_p)
%  Core: \prod_{lmn} N(\mu_{lmn},\sigma_{lmn})
%

[i,j] = find(size(X)<D);
if ~isempty(i)
    s = sprintf('Core size larger than number of obervations:\n');
    for ij = 1:length(i)
        s = sprintf('%s\t For core %i in mode %i - %i > %i\n',s,i(ij),j(ij), D(i(ij),j(ij)), size(X,j(ij)));
    end
    
    error(['%s Due to orthogonality, the latent dimension cannot be ' ...
        'larger than the number of observations in the corresponding mode.'],s)
end


% Create a low-rank representation for zero and non-zero elements in the
% core array.
if size(D,1) > 1 % when there are >1 core
    Dtot=cumsum([zeros(1,size(D,2));D],1);
    for i_mode = 1:size(D,2)
        I_M{i_mode}=false(Dtot(end,i_mode),size(D,1));
        for j=1:size(D,1)
            I_M{i_mode}(Dtot(j, i_mode)+1:Dtot(j+1, i_mode),j) = 1;
        end
    end
    M = logical(nmodel(I_M));
else % if there is only one core
    M = true(D);
    Dtot=[zeros(size(D));D];
    for j = ndims(X):-1:1
        I_M{j} = true(D(j),1);
    end
end

% Hyper parameters
alpha_tau= 1e-3; beta_tau = 1e-3;
alpha_lambda = 1e-3; beta_lambda = 1e-3;
% alpha_tau= 1; beta_tau = 1;
% alpha_lambda = 1; beta_lambda = 1;
max_iter = 100;
conv_crit=1e-6;


update_core = true;
update_factors = true;
update_core_prior = true;
update_noise_precision = true;

if nargin < 3 || isempty(core_prior_type)
    %core_prior_type = 'scale';
    core_prior_type = 'sparse';
end

fixed_lambda = 8;
fixed_tau = 5; % Important to learn tau before switching away from MAP estimation of vMF
fixed_orth_map_est = 6;

%% Initialize random variables
Nx=ndims(X);
N=size(X);

% Initialize Factors
E_U=cell(1,Nx);
for i=1:Nx
    E_U{i}=zeros(N(i),Dtot(end,i));
    for j = 1:size(D,1)
        E_U{i}(:,Dtot(j, i)+1:Dtot(j+1, i))=orth(randn(N(i),D(j,i)));
    end
end
H_U = 0;


% Initialize Core array
E_G=zeros(sum(D,1));
Rec=0;
for j=1:size(D,1)
    MM=1;
    XU=X-Rec;
    for jj=1:Nx
        XU=tmult(XU,E_U{jj}(:,Dtot(j, jj)+1:Dtot(j+1, jj))',jj);
        MM=tmult(MM,I_M{jj}(:,j),jj);
    end
    MM=logical(MM);
    E_G(MM)=XU;
    Rect=reshape(E_G(MM),D(j,:));
    for jj=1:Nx
        Rect=tmult(Rect,E_U{jj}(:,Dtot(j, jj)+1:Dtot(j+1, jj)),jj);
    end
    Rec=Rec+Rect;
end
E_G = E_G .* M;

% Initialize core precision priors
E_lambda_core = (alpha_lambda/beta_lambda*ones(size(E_G))).* M;
E_log_lambda_core = (psi(alpha_lambda)-log(beta_lambda)).* M;
% Initialize noise precision
E_tau =   alpha_tau/beta_tau;%1e-2*numel(X)/SST;
E_log_tau = psi(alpha_tau)-log(beta_tau);

% Update expected second moment of the core.
sigma_sq_core=M.*((E_tau+E_lambda_core).^-1);
if update_core && update_factors
    if false % init via Tucker from the nway-toolbox
        [E_U, E_G] = tucker(X, sum(D,1));
        E_G = E_G .* M;
    end
end
E_G_sq=E_G.^2+sigma_sq_core;


%% Fit decomposition using alternating update scheme

iter=0;
SST=sum(X(:).^2);
stats.ELBO=-inf;
dELBO_relative=inf;
tic;

disp([' '])
disp(['VB Tucker optimization'])
% disp(['A ' num2str(D) ' component model will be fitted'])
dheader = sprintf('%12s | %12s | %12s | %12s | %12s | %12s | %12s | %12s | %12s | %12s','Iteration','ELBO.','dELBO','E[Var. Expl]','Time','E_tau', 'RMS', 'SSE', '||CORE||','|Core_prior|','||U1||','||U2||','||U3||');
dline = sprintf('-------------+--------------+--------------+--------------+--------------+');

X_recon = nmodel(E_U,E_G);
sse_recon = norm(X(:)-X_recon(:),2)^2/norm(X(:),2)^2;
disp(dline); disp(dheader); disp(dline);
fprintf('%12i | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e\n',...
    0, nan,nan,nan,nan, E_tau, rms(X(:)-X_recon(:)), sse_recon, norm(E_G(:),2)^2, norm(E_lambda_core(:),2)^2);

while dELBO_relative>=conv_crit && iter<max_iter || ...
        (iter <= fixed_lambda) || (iter <= fixed_tau)
    
    if sse_recon < 1e-12
        fprintf('Terminated by SSE criteria')
        break
    end
    
    if mod(iter,100)==0
        disp(dline); disp(dheader); disp(dline);
    end
    iter=iter+1;
    ELBO_old=stats.ELBO;
    
    % Estimate mode loadings
    if update_factors
        for times=1:1 %(10+1*(iter==1))
            H_U=zeros(1,Nx);
            for j=1:size(D,1) % randperm(size(D,1))%
                
                % Remove effect of the j'th block and calculate
                % residual error
                E_Ut=E_U;
%                 E_Ut{1}(:,Dtot(j, 1)+1:Dtot(j+1, 1)) = 0; 
                % NB technically
                for i = 1:Nx, E_Ut{i}(:,Dtot(j, i)+1:Dtot(j+1, i))=0; end % but just zeroing one will surfice

                X_residual = X-nmodel(E_Ut,E_G); %  y_t
                
                % Update each mode of the j'th block
                for i= 1:Nx% randperm(Nx)%
%                     fprintf('Updating core j=%i in mode i=%i with size (%i x %i)\n', j, i, N(i), D(j,i))
                    NN=1:Nx;
                    NN(i)=[];
                    XUnoti=X_residual;
%                     XUnoti2=X;
                    E_G_rel=E_G;
                    for jj=NN 
                        XUnoti=tmult(XUnoti,E_U{jj}(:,Dtot(j, jj)+1:Dtot(j+1, jj))',jj); % core times U_t^m
                        E_G_rel_size=size(E_G_rel);  % extract subcore
                        E_G_rel=matricizing(E_G_rel,jj);
                        E_G_rel=E_G_rel(I_M{jj}(:,j),:);
                        E_G_rel_size(jj)=sum(I_M{jj}(:,j));
                        E_G_rel=unmatricizing(E_G_rel,jj,E_G_rel_size);
                    end
                    %%
                    
                    XUnoti_i=matricizing(XUnoti,i);
                    E_G_rel=matricizing(E_G_rel,i);
                    F=(XUnoti_i*E_G_rel(I_M{i}(:,j),:)')*E_tau;
                    [UU,SS,VV]=svd(F, 'econ');%0);
                    if size(F,2)>size(UU,2)
                        disp('Not linearly independent')
                        %keyboard
                        error('Core %i for mode %i did not have %i linearly independent factors - if this error presists, try a smaller core.',j,i,size(F,2))
                    end
                    [f,V,lF_j]=hyperg(N(i),diag(SS),3);
                    
                    assert(~isinf(lF_j))
                    if iter<=fixed_orth_map_est
                        EU=UU*VV'; % Use MAP estimate for the first few iterations
                    else
                        EU=UU*diag(f)*VV'; % Expectation of U
                    end
                    E_U{i}(:,Dtot(j, i)+1:Dtot(j+1, i))=EU;
                    H_U(i)= H_U(i)+lF_j-sum(sum(F.*EU));
                end
                
                % Update j'th core
                XU=X_residual;
                MM=1;
                for jj=1:Nx
                    XU=tmult(XU,E_U{jj}(:,Dtot(j, jj)+1:Dtot(j+1, jj))',jj);
                    MM=tmult(MM,I_M{jj}(:,j),jj);
                end
                MM=logical(MM);
                if update_core
                    % update core
                    sigma_sq_core(MM)=((E_tau+E_lambda_core(MM)).^-1);
                    
                    E_G(MM)=(XU(:).*sigma_sq_core(MM))*E_tau;
                    
                    E_G_sq(MM)=E_G(MM).^2+sigma_sq_core(MM);
                    %  H_G=0.5*(log(2*pi)+1+log(sigma_sq_core));
                end
                
            end
        end
    end
    
    % Calculate residual error
    E_Rec=nmodel(E_U,E_G);
    E_Rec_sq=sum(E_Rec(:).^2);
    for j=1:size(D,1)
        E_G_rel=E_G;
        for jj=1:Nx
            E_G_rel_size=size(E_G_rel);
            E_G_rel=matricizing(E_G_rel,jj);
            E_G_rel=E_G_rel(I_M{jj}(:,j),:);
            E_G_rel_size(jj)=sum(I_M{jj}(:,j));
            E_G_rel=unmatricizing(E_G_rel,jj,E_G_rel_size);
        end
        for jj=1:Nx
            E_G_rel=tmult(E_G_rel,E_U{jj}(:,Dtot(j, jj)+1:Dtot(j+1, jj)),jj);
        end
        E_Rec_sq=E_Rec_sq-sum(E_G_rel(:).^2);
    end

    %%
    
    H_G = 0.5*sum(log(sigma_sq_core(M(:)))) + 0.5*sum(prod(D,2),1)*(1+log(2*pi));
    E_SSE=SST+sum(E_G_sq(:))+E_Rec_sq-2*X(:)'*E_Rec(:);

    % Update noise precision (tau)
    if update_noise_precision && iter > fixed_tau
        est_tau_alpha= alpha_tau+prod(N)/2;
        est_tau_beta = beta_tau+E_SSE/2;
    else
        est_tau_alpha = alpha_tau;
        est_tau_beta = beta_tau;
    end
    E_tau=est_tau_alpha/est_tau_beta;
    E_log_tau=psi(est_tau_alpha)-log(est_tau_beta);
    [H_tau, prior_tau]=entropy_gamma(est_tau_alpha, ...
        est_tau_beta, alpha_tau, beta_tau);
    
    % Update Lambda on the core.
    if update_core_prior && iter > fixed_lambda
        if strcmpi(core_prior_type,'scale')
            est_lambda_alpha= alpha_lambda + numel(E_G_sq)/2;
            est_lambda_beta = beta_lambda + sum(E_G_sq(:))/2;
            
        elseif strcmpi(core_prior_type,'sparse')
            est_lambda_alpha= alpha_lambda + 1/2;
            est_lambda_beta = beta_lambda + E_G_sq/2;
        end
    else
        est_lambda_alpha = alpha_lambda;
        if strcmpi(core_prior_type,'scale')
           est_lambda_beta = beta_lambda;
        elseif strcmpi(core_prior_type,'sparse')
            est_lambda_beta = beta_lambda*ones(size(E_G));
        end
        
    end
    E_lambda_core=est_lambda_alpha./est_lambda_beta;
    E_log_lambda_core=psi(est_lambda_alpha)-log(est_lambda_beta);

    E_lambda_core = E_lambda_core .* M;
    if strcmpi(core_prior_type,'scale')
           E_log_lambda_core = E_log_lambda_core*M;
    elseif strcmpi(core_prior_type,'sparse')
        est_lambda_beta = est_lambda_beta(M(:));
    end
    E_log_lambda_core = E_log_lambda_core(M(:));
    [H_lambda_core, prior_lambda_core]=entropy_gamma(est_lambda_alpha, ...
        est_lambda_beta, alpha_lambda, beta_lambda);
    
    % Calculate lowerbound
    % likelihood
    stats.ELBO=-prod(N)/2*log(2*pi)+prod(N)/2*E_log_tau-0.5*E_SSE*E_tau;
    % prior and entropy on tau
    stats.ELBO=stats.ELBO+prior_tau+H_tau;
    % prior and entropy on lambda
    stats.ELBO=stats.ELBO+sum(prior_lambda_core(:))+sum(H_lambda_core(:));
    % prior and entropy on core
    stats.ELBO=stats.ELBO + sum(0.5*(E_log_lambda_core(:)-log(2*pi)))...
        -0.5*sum(E_G_sq(:).*E_lambda_core(:))...
        +H_G;
    % prior and entropy on factors
    stats.ELBO=stats.ELBO + 0 + sum(H_U); % VMF prior is log(1)=0 ignoring same constants in prior and entropy

    log_volume_stiefel = zeros(size(D));
    for d = 1:size(D,1) % each core
        for n = 1:Nx
            I=N(n); Dstielf=D(d,n);
                    %volume = (2^D * pi^(I*D/2)) / ( pi^(D*(D-1)/4) * (prod(gamma(1/2*(I-(1:D) +1)))) ) ;
            log_volume_stiefel(d,n) = Dstielf*log(2) + I*Dstielf/2*log(pi) - Dstielf*(Dstielf-1)/4*log(pi) ...
                        - sum(gammaln(1/2*(I-(1:Dstielf) +1) ) );
        end
    end
    stats.ELBO = stats.ELBO + sum(-log_volume_stiefel(:));

    
    ELBO(iter)=stats.ELBO;
    dELBO_relative=(ELBO(iter)-ELBO_old)/abs(ELBO(iter));
    
    % Display iteration
    if rem(iter,1)==0
        X_recon = nmodel(E_U,E_G);
        sse_recon = norm(X(:)-X_recon(:),2)^2/norm(X(:),2)^2;
        fprintf('%12i | %12.4e | %12.4e | %12.4f | %12.4f | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e | %12.4e\n',...
            iter, ELBO(iter),dELBO_relative,(SST-E_SSE)/SST,toc, E_tau, rms(X(:)-X_recon(:)), sse_recon, norm(E_G(:),2)^2, norm(E_lambda_core(:),2)^2, norm(E_U{1}(:))^2, norm(E_U{2}(:))^2, norm(E_U{3}(:))^2);
        tic;
    end
    if iter > fixed_orth_map_est
        try
            [~, c_fixed_lambda, c_fixed_tau] = isElboDiverging(...
                iter, dELBO_relative, conv_crit,...
                {'hyperprior (lambda)', update_core_prior, fixed_lambda},...
                {'noise (tau)', update_noise_precision, fixed_tau});
            fixed_lambda = min(fixed_lambda, c_fixed_lambda);
            fixed_tau = min(fixed_tau, c_fixed_tau);
        catch ME
            warning('Lowerbound diverged, try again with different starting point.')
%             [E_G, E_U, ELBO] = pt_Tucker_BTD(X, D); %, E_G_init)
            return
        end
        
    end
    
end
% Display final iteration
fprintf('%12.0f | %12.4f | %6.5e | %12.4e \n',iter, ELBO(iter),dELBO_relative/abs(ELBO(iter)),toc);

% Note, this can happen, if the SNR is so low, that no model can be found.
% assert(sum(abs(E_G(:)))~=0, 'The core array was entirely off, this should not happen. What is the scale of your data?')

end
%------------------------------------------------------------------
function [H_gamma, prior_gamma]=entropy_gamma(a,b,a0,b0)

H_gamma=-log(b)+a-(a-1).*psi(a)+gammaln(a);
prior_gamma=-gammaln(a0)+a0*log(b0)+(a0-1)*(psi(a)-log(b))-b0*a./b;


end