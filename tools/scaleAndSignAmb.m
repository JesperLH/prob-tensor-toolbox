function [A,scale] = scaleAndSignAmb(A)

scale = ones(1,size(A{1},2));

for n = 2:length(A)
    scale = scale.*sqrt(var(A{n},[],1));%mean(A{n}.^2, 1));
    A{n} = bsxfun(@rdivide,A{n},sqrt(var(A{n},[],1)));%mean(A{n}.^2, 1)));
end
A{1} = bsxfun(@times,A{1},scale);


scale_sign = ones(1,size(A{1},2));

for n = 2:length(A)
    scale_sign = scale_sign.*sign(sum(A{n}, 1));
    A{n} = bsxfun(@rdivide,A{n},sign(sum(A{n}, 1)));
end
A{1} = bsxfun(@times,A{1},scale_sign);