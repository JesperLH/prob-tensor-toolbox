%% Measures the sparseness of each column in A.
% On a normalized scale, the sparsest possible vector (only a single
% component is non-zero) will have a sparseness of 1, whereas a vector with
% all elements equal should have a sparseness of 0.
%
% The implemented measure is presented in [1] section 3.1 . This sparseness
% measure is based on the relationship between the L_1 norm and the L_2
% norm.
%
% [1] Hoyer PO. Non-negative matrix factorization with sparseness
% constraints. Journal of machine learning research. 2004;5(Nov):1457-69. 
%
function spar = measure_sparseness(A)

N = size(A,1);
spar = (sqrt(N)-sum(abs(A),1)./sqrt(sum(A.^2,1))) / (sqrt(N)-1);

end