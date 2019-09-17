function A = initialize_with_mu_updates(X, D, num_updates)
    if nargin < 3
        num_updates=50;
    end

    fprintf('Initializing factors using MLE with %i MU updates... ',num_updates);
    t0 = tic;

    options_ncp.hals = 0;
    options_ncp.mu = 1;
    options_ncp.maxiter = num_updates;
    options_ncp.maxiter_sub = 10; %#ok<STRNU> %number of hals updates
    [~,A,~,~]=evalc('cpNonNeg(X, D,[],options_ncp);');
    toc;
end