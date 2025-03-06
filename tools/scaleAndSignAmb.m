function [A,scale] = scaleAndSignAmb(A,idx_scale)

if nargin < 2
    idx_scale = 1;
end

scale = ones(1,size(A{1},2));

for n = setdiff(1:length(A),idx_scale)
    % scale = scale.*sqrt(var(A{n},[],1));%mean(A{n}.^2, 1));
    this_scale = sqrt(sum(A{n}.^2,1));
    % this_scale = abs(mean(A{n},1));
    % this_scale = sum(A{n},1);
    % this_scale = max(A{n},[],1);
    scale = scale.*this_scale;
    A{n} = bsxfun(@rdivide,A{n},this_scale);%mean(A{n}.^2, 1)));
end
A{idx_scale} = bsxfun(@times,A{idx_scale},scale);


scale_sign = ones(1,size(A{1},2));

for n = setdiff(1:length(A),idx_scale)
    scale_sign = scale_sign.*sign(sum(A{n}, 1));
    A{n} = bsxfun(@rdivide,A{n},sign(sum(A{n}, 1)));
end
A{idx_scale} = bsxfun(@times,A{idx_scale},scale_sign);