function [G] = tt_tensor_fixed_rank(X,D)
% TT_TENSOR_FIXED_RANK(X,D) calculates the tensor train with the specified
% size D.
%

if nargin < 1
    N = 20:-1:16;
    D = [1,length(N)+1:-1:3, 1];
    X_clean = generateTensorTrain(N, D);
    X = addTensorNoise(X_clean, 10);
end
N = size(X);

G = cell(ndims(X),1);
%%
C = X;
for n = 1:ndims(X)-1
    if n
    m=N(n)*D(n); C=reshape(C,[m,numel(C)/m]);
    
    [U,S,V] = svd(C, 'econ');
    G{n} = reshape(U(:,1:D(n+1)), D(n), N(n), D(n+1));
    C = (V(:,1:D(n+1))*S(1:D(n+1),1:D(n+1)))';

    1+1;
end
G{end} = (V(:,1:D(end-1))*S(1:D(end-1),1:D(end-1)))';

% G{1} = squeeze(G{1});
end