function getRandomMissingIndex(dims,frac_missing)
%%
dims = [5,5,5];
frac_missing = 0.9;
R = true(dims);

idx = randperm(numel(R));

R(idx(1:floor(numel(R)*frac_missing))) = 0;

for i = 1:length(dims)
    sum(matricizing(R,i),1)
end



end