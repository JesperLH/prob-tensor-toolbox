function T=outerprod(FACT)

% The outer products:
% T=sum_f (U1f o U2f o U3f o ...o UNf) where Uif is a vector corresponding 
% to the f'th factor of the i'th dimension.
%
% Input:
% FACT      Cell array containing the factor-vectors corresponding to 
%           the loadings of each dimensions in a PARAFAC model, i.e. Ui=FACT{i}
% Output: 
% T         The multi-way array created from the sum of outer products

krpr=FACT{2};
N=zeros(1,length(FACT));
N(2)=size(FACT{2},1);
for i=3:length(FACT)
    krpr=krprod(FACT{i},krpr);
    N(i)=size(FACT{i},1);
end
T=FACT{1}*krpr';
N(1)=size(FACT{1},1);
T=unmatricizing(T,1,N);
 


