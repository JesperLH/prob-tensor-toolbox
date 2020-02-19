function [ss]=tprod(X,iX,Y,iY)
% code snippet taken from 
% Saleh Bawazeer
% https://se.mathworks.com/matlabcentral/fileexchange/16275-tprod-arbitary-tensor-products-between-n-d-arrays
iX=iX(:); iY=iY(:);
[res,ia,ib]=intersect(iX,iY);
if size(res,1)>1
    res = res';
end
if size(ia,1)>1
    ia = ia';
end
if size(ib,1)>1
    ib = ib';
end
%check sizes 
%need to add some tests for dimensions 
numer_of_sums=length(res(res<0));

[fX,ifX]=setdiff(iX,res); 
[fY,ifY]=setdiff(iY,res);

res1=[res fX']; 
ia1 =[ia ifX']; 
ib1 =[ib [1:length(fX)]+length(iY)];

res2=[res1 fY']; 
ia2 =[ia1 [1:length(fY)]+length(iX)]; 
ib2 =[ib1 ifY'];

[fS,iS]=sort(res2); 
ia2(iS); 
ib2(iS);

ss=permute(X,ia2(iS)).*permute(Y,ib2(iS)); 
for i=1:numer_of_sums 
ss=sum(ss,i); 
end 
S=size(ss); 
ss=reshape(ss,S(numer_of_sums+1:end)); 
return;