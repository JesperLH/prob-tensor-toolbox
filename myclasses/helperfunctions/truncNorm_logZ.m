function logZhatOUT =...
    truncNorm_logZ( lowerBIN , upperBIN , muIN , sigmaIN )
%% Init
if ~isa(muIN,'gpuArray')
    lowerB = lowerBIN;
    upperB = upperBIN;
    mu = muIN;
    sigma = sigmaIN;
else
    lowerB = gather(lowerBIN);
    upperB = gather(upperBIN);
    mu = gather(muIN);
    sigma = gather(sigmaIN);
end

if any(lowerB(:)>upperB(:)) % Note this can be handled by case 2
    error('You have given lowerbounds that are greater than the upperbounds')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
% establish bounds
%%%%%%%%%%%%%%%%%%%%%%%%%%
a = (lowerB - mu)./(sqrt(2*sigma));
b = (upperB - mu)./(sqrt(2*sigma));


%%%
logZhatOUT = nan(size(a),'like',a);
%% Case 1: Both limits are infinite
index = isinf(a) & isinf(b);
% Same signs
idx_same = (sign(a) == sign(b));
idx_diff = index & ~idx_same;
idx_same = index & idx_same;
logZhatOUT(idx_same) = -inf;
% Different signs
logZhatOUT(idx_diff) = 0;

%%
% Case 2: Bounds pointing the wrong way a > b, return 0 by convention
% This never happens if upperB > lowerB, unless there is some issues of
% numerical stability (which I haven't encountered).
index = a > b;
if any(index(:))
    logZhatOUT(index) = -inf;
end

% Case 3: If only a is -infinite 
index = isinf(a) & ~isinf(b);
[logZhat(index)] =...
    a_is_inf(b(index));

%Case 4: If only b is infinite 
index = ~isinf(a) & isinf(b);
[logZhat(index)] =...
    b_is_inf(a(index));


% Case 5: Neither a or b is inf and a < b
index = ~(isinf(a) | isinf(b)) & (a < b);
[logZhat(index)] =...
    neither_is_inf(a(index),b(index));


%%
index = ~(isinf(a) & isinf(b) | (a > b)); 

logZhatOUT(index) = logZhat(index);


end

%% Case 1: Both limits are infinite
function [ logZhatOUT] =...
    both_limits_inf(a,b)
    logZhatOUT = nan(size(a),'like',a);
    
    % Same signs
    idx = sign(a) == sign(b);
    logZhatOUT(idx) = -inf;
    
    %% Different signs
    idx = ~idx;
    logZhatOUT(idx) = 0;
    
end

%% Case 3: If only a is -infinite
function [ logZhat] =...
    a_is_inf(b)
    logZhat = nan(size(b),'like',b);
    %% if b is greather than 26
    idx = b > 26;
    logZhat(idx) = log1p(-exp(log(0.5)+log(erfcx(b(idx)))-b(idx).^2));
    
    %% else b < 26
    idx = ~idx;
    logZhat(idx) = log(0.5)+log(erfcx(-b(idx)))-b(idx).^2;
    
end

%% Case 4: If only b is infinite 
function [ logZhat] =...
    b_is_inf(a)
    logZhat = nan(size(a),'like',a);
    % if a is less than -26
    idx = a < -26;
    logZhat(idx) = log1p(-exp(log(0.5)+log(erfcx(-a(idx)))-a(idx).^2));
    % else a > -26
    idx = ~idx;
    logZhat(idx) = log(0.5)+log(erfcx(a(idx)))-a(idx).^2;

end

%% Case 5: Neither a or b is inf and a < b
function [ logZhat ] =...
    neither_is_inf(a,b)
    logZhat = nan(size(a),'like',a);
    
    %% then the signs are different, which means b>a (upper>lower by definition), and b>=0, a<=0.
    % but we want to take the bigger one (larger magnitude) and make it positive, as that
    % is the numerically stable end of this tail.
    idx = abs(b) >= abs(a);
    if any(idx(:))
        idx = idx & a >= -26;
        logZhat(idx) = log(0.5) - a(idx).^2 + log( erfcx(a(idx))...
                       - exp(-(b(idx).^2 - a(idx).^2)).*erfcx(b(idx)) );

        % Otherwise: a is too small and the calculation will be unstable, so
        % we just put in something very close to 2 instead.
        % Again this uses the relationship
        % erfc(a) = 2 - erfc(-a). Since a<0 and b>0, this
        % case makes sense.  This just says 2 - the right
        % tail - the left tail.
        idx = abs(b) >= abs(a) & (a < -26);
        logZhat(idx) = log(0.5) + log( 2 - exp(-b(idx).^2).*erfcx(b(idx)) - exp(-a(idx).^2).*erfcx(-a(idx)) );

    end
    %% abs(a) is bigger than abs(b), so we reverse the calculation...
    idx = abs(b) < abs(a);
    if any(idx(:))
    idx = idx & (b <= 26);
        % do things normally but mirrored across 0
        logZhat(idx) = log(0.5) - b(idx).^2 + log( erfcx(-b(idx)) - exp(-(a(idx).^2 - b(idx).^2)).*erfcx(-a(idx)));
    end
        
    idx = abs(b) < abs(a) & (b > 26);
    if any(idx(:))
        % b is too big and the calculation will be unstable, so
        % we just put in something very close to 2 instead.
        % Again this uses the relationship
        % erfc(a) = 2 - erfc(-a). Since a<0 and b>0, this
        % case makes sense. This just says 2 - the right
        % tail - the left tail.
        logZhat(idx) = log(0.5) + log( 2 - exp(-a(idx).^2).*erfcx(-a(idx)) - exp(-b(idx).^2).*erfcx(b(idx)) );
    end
    
end