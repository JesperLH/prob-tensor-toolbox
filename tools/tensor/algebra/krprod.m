function A=krprod(B,C);

% Khatri Rao product
% input
% B m x n matrix
% C p x n matrix
% output
% A m*p x n matrix
sb=size(B,1);
sc=size(C,1);
A=zeros(sb*sc,size(B,2),'like',B);
for k=1:size(B,2)
    A(:,k)=reshape(C(:,k)*B(:,k)',sb*sc,1);
end

    
    