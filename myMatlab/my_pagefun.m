%% MY_PAGEFUN mimics the MATLAB function pagefun, but supports CPU computation
% Note this function is likely more memory intensive, as well as slower
% than the strictly GPU pagefun. However, it enables easy switching between
% CPU and GPU code.
%
% Two extensive packages for multi dimensional arrays/matrices are:
%
% 1) MTIMESX - Fast Matrix Multiply with Multi-Dimensional Support
% 2) MMX - Multithreaded matrix operations on N-D matrices
% 
% Both available at MATLAB Central. These are faster and probably more
% memory efficient, but requires source code compiling.
function AB = my_pagefun(fhandle,A,B)

if isa(A,'gpuArray') || (nargin== 3 && isa(B,'gpuArray'))
    if nargin < 3
        AB = pagefun(fhandle,A);
    else
        AB = pagefun(fhandle,A,B);
    end
else
    if nargin < 3
       sa = size(A);
       NDdim = prod(sa(3:end));
       A = reshape(A,[sa(1),sa(2),NDdim]);
       AB = nan(size(A),'like',A);
       for i = 1:size(A,3)
           AB(:,:,i) = fhandle(A(:,:,i));
       end
       AB = reshape(AB,sa);
       return
    end
    % Get the array sizes.
    sa = size(A);
    sb = size(B);
    if length(sa) < length(sb)
        sa = [sa,ones(1,length(sb)-length(sa))];
    elseif length(sb) < length(sa);
        sb = [sb,ones(1,length(sa)-length(sb))];
    end

    % Determine which dimensions should be replicated, to allow easy page
    % multiplication.
    numReshapeDimA = [1,1,(sa(3:end) == 1).*sb(3:end)];
    numReshapeDimB = [1,1,(sb(3:end) == 1).*sa(3:end)];
    numReshapeDimA(numReshapeDimA == 0) = 1;
    numReshapeDimB(numReshapeDimB == 0) = 1;

    % Unfold A and B, such that they are 3D arrays, with the 3rd dimension
    % being the number of pages
    sab = [sa(1),sb(2),max([sa(3:end);sb(3:end)])];
    NDdim = prod(sab(3:end));
    A = reshape(repmat(A,numReshapeDimA),[sa(1:2),NDdim]);
    B = reshape(repmat(B,numReshapeDimB),[sb(1:2),NDdim]);

    %% Every page is then multipled.
    % Note that using parfor, does not speed-up calculation on a single machine
    if isequal(fhandle,@times) || isequal(fhandle,@plus) || ...
       isequal(fhandle,@uplus) || isequal(fhandle,@minus) || ...
       isequal(fhandle,@uminus) || isequal(fhandle,@rdivide) ||...
       isequal(fhandle,@ldivide) || isequal(fhandle,@power) 
        AB = nan(sa(1),sa(2),NDdim,'like',A); %handle is scalar multiplication
        isscalar = true;
    else
        AB = nan(sa(1),sb(2),NDdim,'like',A);
        isscalar = false;
    end
    for i = 1:size(AB,3)
        AB(:,:,i) = fhandle(A(:,:,i),B(:,:,i));
    end

    % reshape to desired output format
    if isscalar
        AB = reshape(AB,[sa(1),sa(2),sab(3:end)]);
    else
        AB = reshape(AB,[sa(1),sb(2),sab(3:end)]);
    end

end
end