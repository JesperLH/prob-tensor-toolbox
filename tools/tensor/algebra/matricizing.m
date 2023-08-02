function Y=matricizing(X,n)

% The matricizing operation, i.e.
% matricizing(X,n) turns the multi-way array X into a matrix of dimensions:
% (size(X,n)) x (size(X,1)*...*size(X,n-1)*size(X,n+1)*...*size(X,N))
% where N=ndims(X)
%
% Input:
% X     Multi-way array
% n     Dimension along which matrizicing is performed
%
% Output:
% Y     The corresponding matriziced version of X

N=ndims(X);
% Y=reshape(permute(X, [n 1:n-1 n+1:N]),size(X,n),numel(X)/size(X,n));
if isempty(n)
    
elseif n == N+1
    Y = X(:).';
else
    sX = size(X);
    n2 = setdiff(1:N,n);
    Y = reshape(permute(X, [n n2]),prod(sX(n)),prod(sX(n2)));
end
