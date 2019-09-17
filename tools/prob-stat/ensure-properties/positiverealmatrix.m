%% Checks (and enforces) a matrix to be positive and real
% This function checks if a given matrix A is real and positive, and then 
% enforces these properties. It is used to ensure that the eigenvalues of 
% a spectral decomposition are real and positive which is theoretically 
% true for a positive-semidefinite matrix (such as a covariance matrix).
%
%
% The debug option (0 or 1) determines if additional checks are made and
% feedback to the user is given. Not making these checks are faster.
function A = positiverealmatrix(A,inDebugMode)
if ~isreal(A)
    if nargin ~= 1 && inDebugMode
       sum_img = sum(abs(imag(A(:))));
       
       if sum_img > 1e-16
            error('Matrix contained complex values, the absolute sum of complex values were %12.6e\n',sum_img);
       end
       A = real(A);
       min_neg = sum(sum(A(A<0)));
       if -min_neg > 1e-16
            error('Matrix contained negative values, the min value were %12.6e\n',sum_img);
       end
    else
        A = real(A); %Realising hahaha
    end
end