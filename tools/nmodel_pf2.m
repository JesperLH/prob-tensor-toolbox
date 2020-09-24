function [X] = nmodel_pf2(A,H,C,P, return_array, idx_vary)
if nargin < 5
    return_array = false;
end
% Reconstruct the full data from the PARAFAC2 model with one or more
% varying modes.
% INPUT:    A:  Cell array of fixed factors
%           H:  Cell array of rotation matrix for each varying mode
%           C:  Matrix with activation in the sample mode
%           P:  Cell array (K,M) with an orthogonal matrix for each k
%           sample and each m mode variation.
%
% OUTPUT:   X:  Cell array with the reconstructed data.

if ~iscell(A)
    A = {A};
end
if ~iscell(H)
    H = {H};
end
if size(P,1)==1
    P = P';
end

[K,D] = size(C);
X = cell(K,1);
for k = 1:K
    Bkr = ones(1,D);
    for i_vary = size(P,2):-1:1
       Bkr = krprod(Bkr,P{k,i_vary}*H{i_vary}); 
    end

    Akr = ones(1,D);
    for i_fixed = length(A):-1:1
        Akr = krprod(Akr, A{i_fixed});
    end

%     sx = [cellfun(@(t) size(t,1), A),cellfun(@(t) size(t,1), P(k,:))];
    sx = [cellfun(@(t) size(t,1), A(:))', cellfun(@(t) size(t,1), P(k,:)')];
    sx(sx==1) = [];
    X{k} = reshape(krprod(Bkr,Akr)*(C(k,:))',sx);
end

if return_array
    % This will fail if the dimensions are not equal
    X = cat(ndims(X{1})+1,X{:}); 
end


end