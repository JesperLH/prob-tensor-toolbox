%% Symmetrizing a matrix to numerical precision
% This function checks if a given matrix A is symmetric via an in-build
%  MATLAB function, and then symmetrizes the matrix, by taking the average
%  of the lower and upper triangular matrix.
%
% The debug option (0 or 1) determines if additional checks are made and
% feedback to the user is given. Not making these checks are faster.
function A = symmetrizing(A,inDebugMode)
if ~issymmetric(A)
    if nargin ~= 1 && inDebugMode
        relative_error = (norm(A'-A,'fro')/norm(A,'fro'))^2;
        if relative_error >1e-6
            error('Matrix Symmetrized, max relative error was %12.6e\n',relative_error);
        elseif relative_error > 1e-16
            warning('Matrix Symmetrized, max relative error was %12.6e\n',relative_error);
        end
        
        A = (A+A')/2;
        % Checking if the resulting matrix A is positive semidefinte, which
        % is ensured by having positive or zero eigenvalues.
        %
        % However, numerical instability might give negative eigenvalues
        % that are close to zero. These can be truncated to zero in any
        % application where the eigenvalues are relevant.
%         eigenvalues=eig(A); 
%         eigenvalues=eigenvalues(eigenvalues<0);
%         if norm(eigenvalues)^2 > 1e-16
%             error('Resulting matrix was not positive semidefinite')
%         end
    else
        A = (A+A')/2; %Symmetrizing
    end
end