function A=krsum(B,C);

% Khatri Rao sum---- need better name?
% input
% B m x n matrix
% C p x n matrix
% output
% A m*p x n matrix
sb=size(B,1);
sc=size(C,1);
A=zeros(sb*sc,size(B,2));
for k=1:size(B,2)
    A(:,k)=reshape(bsxfun(@plus, C(:,k), B(:,k)'),sb*sc,1);
end

%eof
end
    
    