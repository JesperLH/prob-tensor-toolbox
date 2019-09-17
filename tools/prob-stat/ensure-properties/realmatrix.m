%% Checks (and enforces) a matrix to be real
% This function checks if a given matrix A is real and the enforces this
% property. 
%
% Note: An error is thrown is the complex value is significant.
%
% The debug option (0 or 1) determines if additional checks are made and
% feedback to the user is given. Not making these checks are faster.
function A = realmatrix(A,inDebugMode)
if ~isreal(A)
    if nargin ~= 1 && inDebugMode
       sum_img = sum(abs(imag(A(:))));
       
       if sum_img > 1e-16
            error('Matrix contained complex values, the absolute sum of complex values were %12.6e\n',sum_img);
       end
       A = real(A);
    else
        A = real(A); %Realising hahaha
    end
end