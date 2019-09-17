function  LLIK = AAcovGen(Xtest,XC,S,sigma_sq)

% Input:
% Xtest     test data of size Time X Voxel
% XC        T x K matrix XC from AA analysis
% S         K x V matrix from AA analysis
% sigma_sq  1 x V vector of noise estimates from AA 
% 
% Output:
% LLIK      measure of covariance generalization, see paper of Nathan et
%%           al.
        [T,K]=size(XC);
        V=size(S,2);
        X_sig  = XC*S;
        [U L] = svd( X_sig', 'econ' );          
        idx=find(diag(L)>1e-12*L(1,1));
        U=U(:,idx);
        L=L(idx,idx);
        Keff=length(idx);
        D_sig = diag( L.^2 )./ T;        
        D_noi = sigma_sq;
        % terms re-used in likelihood:
        UL_sig  = bsxfun(@times, U, sqrt(D_sig)'); % weight components by kth eig.value
        XtDinvU = Xtest* bsxfun(@times, U, 1./D_noi');
        x_logdet = T * (V*log(2*pi) + sum(log(D_noi)) + log(det( eye(Keff) + UL_sig'*bsxfun(@times,UL_sig, 1./D_noi')  ) ));
        % cross-terms in x for all timepoints:
        x_crossx = sum(sum( bsxfun(@times, Xtest.^2, 1./D_noi ),1 ),2)  -  sum(sum( (XtDinvU/( diag(1./D_sig) + U'*bsxfun(@times, U, 1./D_noi') )).*XtDinvU ));
        % log-likelihood of terms
        LLIK = -0.5 *( x_logdet + x_crossx );
    

