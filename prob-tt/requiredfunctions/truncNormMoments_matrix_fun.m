function [ logZhatOUT, ZhatOUT , muHatOUT , sigmaHatOUT ] =...
    truncNormMoments_matrix_fun( lowerBIN , upperBIN , muIN , sigmaIN )
% Calculates the truncated zeroth, first, and second moments of a
% univariate normal distribution.
%
% This code is an optimized version of John P Cunninghams function, e.g.
% "truncNormMoments.m", available at
% https://github.com/cunni/epmgp/blob/master/truncNormMoments.m 
%
% See Cunninghams code for an excellent description of what is going on.

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

% 
% fprintf('Min(a) = %f \t Max(a)=%f\n',min(a(~isinf(a))),max(a(~isinf(a))))
% fprintf('Min(b) = %f \t Max(b)=%f\n',min(b(~isinf(b))),max(b(~isinf(b))))

%%%
logZhatOUT = nan(size(a),'like',a);
ZhatOUT = logZhatOUT;
muHatOUT = logZhatOUT;
sigmaHatOUT = logZhatOUT;
meanConst = logZhatOUT;
varConst = logZhatOUT;
%% Case 1: Both limits are infinite
index = isinf(a) & isinf(b);
% Note this function does the same this block, but is just slower
%[logZhatOUT(index),ZhatOUT(index),muHatOUT(index),sigmaHatOUT(index)] =...
%    both_limits_inf(a(index),b(index),mu(index),sigma(index));

% Same signs
idx_same = (sign(a) == sign(b));
idx_diff = index & ~idx_same;
idx_same = index & idx_same;
logZhatOUT(idx_same) = -inf;
ZhatOUT(idx_same) = 0;
muHatOUT(idx_same) = a(idx_same);
sigmaHatOUT(idx_same) = 0;

% Different signs
%idx = ~idx;
logZhatOUT(idx_diff) = 0;
ZhatOUT(idx_diff) = 1;
muHatOUT(idx_diff) = mu(idx_diff);
sigmaHatOUT(idx_diff) = sigma(idx_diff);

%%
% Case 2: Bounds pointing the wrong way a > b, return 0 by convention
% This never happens if upperB > lowerB, unless there is some issues of
% numerical stability (which I haven't encountered).
index = a > b;
if any(index(:))
    logZhatOUT(index) = -inf;
    ZhatOUT(index) = 0;
    muHatOUT(index) = mu(index);
    sigmaHatOUT(index) = 0;
end

%% Case 3: If only a is -infinite 
index = isinf(a) & ~isinf(b);
%[logZhat(index),meanConst(index),varConst(index)] =...
%    a_is_inf(b(index),mu(index),upperB(index));

idx = b > 26 & index;
logZhat(idx) = log1p(-exp(log(0.5)+log(erfcx(b(idx)))-b(idx).^2));

% else b < 26
idx = b < 26 & index;
logZhat(idx) = log(0.5)+log(erfcx(-b(idx)))-b(idx).^2;

% the mean/var calculations are insensitive to these calculations, as
% we do not deal in the log space. 
meanConst(index) = -2./erfcx(-b(index));
varConst(index) = -2./erfcx(-b(index)).*(upperB(index) + mu(index));

%% Case 4: If only b is infinite 
index = ~isinf(a) & isinf(b);
%[logZhat(index),meanConst(index),varConst(index),] =...
%    b_is_inf(a(index),mu(index),lowerB(index));

% if a is less than -26
idx = a < -26 & index;
logZhat(idx) = log1p(-exp(log(0.5)+log(erfcx(-a(idx)))-a(idx).^2));
% else a > -26
idx = a > -26 & index;
logZhat(idx) = log(0.5)+log(erfcx(a(idx)))-a(idx).^2;

% the mean/var calculations are insensitive to these calculations, as
% we do not deal in the log space. 
meanConst(index) = 2./erfcx(a(index));
varConst(index) = 2./erfcx(a(index)).*(lowerB(index) + mu(index));


%% Case 5: Neither a or b is inf and a < b
index = ~(isinf(a) | isinf(b)) & (a < b);
[logZhat(index),meanConst(index),varConst(index)] =...
    neither_is_inf(a(index),b(index),mu(index),lowerB(index),upperB(index));


%%

index = ~(isinf(a) & isinf(b) | (a > b)); 

logZhatOUT(index) = logZhat(index);
ZhatOUT(index) = exp(logZhat(index));
muHatOUT(index) = mu(index)+meanConst(index).*sqrt(sigma(index)/(2*pi));
sigmaHatOUT(index) = sigma(index) + varConst(index).*sqrt(sigma(index)/(2*pi))+mu(index).^2-muHatOUT(index).^2;




end

%% Case 1: Both limits are infinite
function [ logZhatOUT, ZhatOUT , muHatOUT , sigmaHatOUT ] =...
    both_limits_inf(a,b,mu,sigma)
    logZhatOUT = zeros(size(a),'like',a);
    ZhatOUT = logZhatOUT;%zeros(size(a),'like',a);
    muHatOUT = logZhatOUT;%zeros(size(a),'like',a);
    sigmaHatOUT = logZhatOUT;%zeros(size(a),'like',a);
    % Same signs
    idx = sign(a) == sign(b);
    logZhatOUT(idx) = -inf;
    ZhatOUT(idx) = 0;
    muHatOUT(idx) = a(idx);
    sigmaHatOUT(idx) = 0;
    
    %% Different signs
    idx = ~idx;
    logZhatOUT(idx) = 0;
    ZhatOUT(idx) = 1;
    muHatOUT(idx) = mu(idx);
    sigmaHatOUT(idx) = sigma(idx);
end

%% Case 3: If only a is -infinite
function [ logZhat, meanConst , varConst ] =...
    a_is_inf(b,mu,upperB)
    logZhat = zeros(size(b),'like',b);
    %% if b is greather than 26
    idx = b > 26;
    logZhat(idx) = log1p(-exp(log(0.5)+log(erfcx(b(idx)))-b(idx).^2));
    
    %% else b < 26
    idx = ~idx;
    logZhat(idx) = log(0.5)+log(erfcx(-b(idx)))-b(idx).^2;
    
    %% the mean/var calculations are insensitive to these calculations, as
    % we do not deal in the log space. 
    meanConst = -2./erfcx(-b);
    varConst = -2./erfcx(-b).*(upperB + mu);
    
end

%% Case 4: If only b is infinite 
function [ logZhat, meanConst , varConst] =...
    b_is_inf(a,mu,lowerB)
    logZhat = zeros(size(a),'like',a);
    % if a is less than -26
    idx = a < -26;
    logZhat(idx) = log1p(-exp(log(0.5)+log(erfcx(-a(idx)))-a(idx).^2));
    % else a > -26
    idx = ~idx;
    logZhat(idx) = log(0.5)+log(erfcx(a(idx)))-a(idx).^2;
    
    %% the mean/var calculations are insensitive to these calculations, as
    % we do not deal in the log space. 
    meanConst = 2./erfcx(a);
    varConst = 2./erfcx(a).*(lowerB + mu);

end

%% Case 5: Neither a or b is inf and a < b
function [ logZhat, meanConst , varConst ] =...
    neither_is_inf(a,b,mu,lowerB,upperB)
    logZhat = zeros(size(a),'like',a);
    meanConst = logZhat;
    varConst = logZhat;
    %% then the signs are different, which means b>a (upper>lower by definition), and b>=0, a<=0.
    % but we want to take the bigger one (larger magnitude) and make it positive, as that
    % is the numerically stable end of this tail.
    idx = abs(b) >= abs(a);
    if any(idx(:))
        idx = idx & a >= -26;
        % do things normally
%         nums = 1:numel(a);
%         for i = nums(idx)
%            log(0.5) - a(i).^2 + log( erfcx(a(i)) - exp(-(b(i).^2 - a(i).^2)).*erfcx(b(i))); 
%         end
        logZhat(idx) = log(0.5) - a(idx).^2 + log( erfcx(a(idx))...
                       - exp(-(b(idx).^2 - a(idx).^2)).*erfcx(b(idx)) );

        % now the mean and var
        meanConst(idx) = 2*(1./((erfcx(a(idx))...
                         -exp(a(idx).^2-b(idx).^2).*erfcx(b(idx))))...
                         -1./((exp(b(idx).^2-a(idx).^2).*erfcx(a(idx))-erfcx(b(idx)))));
        varConst(idx) = 2*((lowerB(idx)+mu(idx))./((erfcx(a(idx))...
                        -exp(a(idx).^2-b(idx).^2).*erfcx(b(idx))))...
                        -(upperB(idx)+mu(idx))./((exp(b(idx).^2-a(idx).^2)...
                        .*erfcx(a(idx)) - erfcx(b(idx)))));
        % Otherwise: a is too small and the calculation will be unstable, so
        % we just put in something very close to 2 instead.
        % Again this uses the relationship
        % erfc(a) = 2 - erfc(-a). Since a<0 and b>0, this
        % case makes sense.  This just says 2 - the right
        % tail - the left tail.
        idx = abs(b) >= abs(a) & (a < -26);
        logZhat(idx) = log(0.5) + log( 2 - exp(-b(idx).^2).*erfcx(b(idx)) - exp(-a(idx).^2).*erfcx(-a(idx)) );

        % now the mean and var
        meanConst(idx) = 2*(1./((erfcx(a(idx)) - exp(a(idx).^2-b(idx).^2).*erfcx(b(idx)))) - 1./(exp(b(idx).^2)*2 - erfcx(b(idx))));
        varConst(idx) = 2*((lowerB(idx)+mu(idx))./((erfcx(a(idx)) - exp(a(idx).^2-b(idx).^2).*erfcx(b(idx)))) - (upperB(idx)+mu(idx))./(exp(b(idx).^2)*2 - erfcx(b(idx))));
    end
    %% abs(a) is bigger than abs(b), so we reverse the calculation...
    idx = abs(b) < abs(a);
    if any(idx(:))
    idx = idx & (b <= 26);
        % do things normally but mirrored across 0
        logZhat(idx) = log(0.5) - b(idx).^2 + log( erfcx(-b(idx)) - exp(-(a(idx).^2 - b(idx).^2)).*erfcx(-a(idx)));

        % now the mean and var
        meanConst(idx) = -2*(1./((erfcx(-a(idx)) - exp(a(idx).^2-b(idx).^2).*erfcx(-b(idx)))) - 1./((exp(b(idx).^2-a(idx).^2).*erfcx(-a(idx)) - erfcx(-b(idx)))));
        varConst(idx) = -2*((lowerB(idx)+mu(idx))./((erfcx(-a(idx)) - exp(a(idx).^2-b(idx).^2).*erfcx(-b(idx)))) - (upperB(idx)+mu(idx))./((exp(b(idx).^2-a(idx).^2).*erfcx(-a(idx)) - erfcx(-b(idx)))));

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
        %%
        
        %%
        

        % now the mean and var
        meanConst(idx) = -2*(1./(erfcx(-a(idx)) - exp(a(idx).^2)*2) - 1./(exp(b(idx).^2-a(idx).^2).*erfcx(-a(idx)) - erfcx(-b(idx))));
        varConst(idx) = -2*((lowerB(idx) + mu(idx))./(erfcx(-a(idx)) - exp(a(idx).^2)*2) - (upperB(idx) + mu(idx))./(exp(b(idx).^2-a(idx).^2).*erfcx(-a(idx)) - erfcx(-b(idx))));

    end
    
end