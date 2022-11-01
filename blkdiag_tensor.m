function G = blkdiag_tensor(cores, core_sizes)
if nargin < 2
    core_sizes_ = cellfun(@size, cores,'UniformOutput',false);
    assert(all(cellfun(@(t) length(core_sizes_{1})==length(t),core_sizes_)),...
        ['Number of modes difference between cores. Maybe some modes have length 1,' ...
        'try explicitly providing the core_sizes argument']);
    core_sizes = cat(1,core_sizes_{:});
end

%%
for ic = 1:length(cores)
    cores{ic}(:) = (1:numel(cores{ic}))*(-1)^ic;
end
% i1 = [1,1,1,2,2,2]
% i2 = [1,2,3,1,2,3]
% i3 = [1,1,1,1,1,1]
%%
n_modes = length(core_sizes(1,:));
G = zeros(sum(core_sizes,1));

for ic = 1:length(cores)
    idx = cell(n_modes,1);
    [ij] = find(cores{ic});
    [idx{:}] = ind2sub(size(cores{ic}), ij);
    for im = 1:n_modes
        if ic>1
                idx{im} = idx{im} + sum(core_sizes(1:ic-1,im));
        end
    end

    idx_lin = sub2ind(sum(core_sizes,1), idx{:});
    assert(length(unique(idx_lin)) == numel(cores{ic}), 'Core %i',ic);
    G(idx_lin) = cores{ic}(:);

end
1+1;
end