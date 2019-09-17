function A = scaleAndSignAmb(A);

scale = ones(1,size(A{1},2));

for n = 2:length(A);
    scale = scale.*sqrt(sum(A{n}.^2, 1));
    A{n} = bsxfun(@rdivide,A{n},sqrt(sum(A{n}.^2, 1)));
end
A{1} = bsxfun(@times,A{1},scale);


scale = ones(1,size(A{1},2));

for n = 2:length(A);
    scale = scale.*sign(sum(A{n}, 1));
    A{n} = bsxfun(@rdivide,A{n},sign(sum(A{n}, 1)));
end
A{1} = bsxfun(@times,A{1},scale);