function ad = amariDist(X,Y,q)
%%AMARIDIST Calculates the Amari distance between X and Y (size(N,q) ) 
% Note dist(X,Y) = 0, iff X==Y
if nargin < 3
    q = size(X,2);
end

if ~any(size(X)==q)
    error('X must have q components')
elseif ~any(size(Y)==q)
    error('Y must have q components')
end
    
if size(X,1) ~= q
    X = X';
end
if size(Y,1) ~= q
    Y = Y';
end
%max(Q),[],2) % max af i over all j
%max(Q),[],1) % max af j over all i

try
    Q = abs(X*pinv(Y));
        %ad = sum(sum(Q,2).*max(Q,[],2)'-1)...
    %    +sum(sum(Q,1).*max(Q,[],1)-1);
    ad = sum(sum(Q,2)./max(Q,[],2)-1)...
        +sum(sum(Q,1)./max(Q,[],1)-1);
    ad = 1/(q)*ad;
catch
    warning('Pinv(Y) failed! Was NaN or Inf')
    mkdir('./errors/')
    save(sprintf('./errors/amariDist_%i',cputime),'X','Y','q')
    
    ad = nan;
end

