%% Calculating the RV-coefficient between two matrices.
% Proposed by [Robert and Escoufier, 1976], the RV-coefficient is a 
% multivariate generalization of the squared Pearson correlation coefficient,
% and can be used to measure the correlation between the subspace of two 
% matrices X (q x N) and Y (q x M),
% 
% 
% $$RV (X,Y) = \frac{\mathrm{trace}\left(X^\top Y Y^\top X\right)}{\mathrm{trace}\left(X^\top X\right)\mathrm{trace}\left(Y^\top Y\right)}$$
% 
% If the coefficient is close 1, the two matrices describe the same subspace, while
% a coefficient of 0 means they share no information.
%
%[Robert and Escoufier, 1976] Robert, P. and Escoufier, Y. (1976). A unifying 
%tool for linear multivariate statistical methods: the rv-coefficient. Applied 
%statistics, pages 257–265.

function rv = coeffRV(X,Y)

if size(X,1) ~= size(Y,1)
    error('X and Y must share first dimension')
end

if nargin == 2
xy = sum(sum((X'*Y).*(X'*Y)));
xx = sum(sum((X'*X).*(X'*X)));
yy = sum(sum((Y'*Y).*(Y'*Y)));

rv = xy/(sqrt(xx)*sqrt(yy));
else
   %RECURSE!!!  
end
