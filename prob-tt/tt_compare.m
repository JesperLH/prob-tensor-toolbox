function [avg_tot, D_tot] = tt_compare(G,H)
%%TT_COMPARE compares to tensor train representations

assert(length(G) == length(H), 'Error! Tensor Trains are of different length')

D_tot = 0;
avg_tot = 0;
for m = 1:length(G)
    if m < length(G)
        % Unfold tensor
        G{m} = permute(G{m},[3,1,2]); G{m} = G{m}(:,:)';
        H{m} = permute(H{m},[3,1,2]); H{m} = H{m}(:,:)';
    end
    [avg_corr,match_idx] = greedy_component_match(abs(corr(G{m},H{m})));
    
    D = size(G{m},2);
    avg_tot = avg_tot + avg_corr*D;
    D_tot = D_tot+D;
end
avg_tot = avg_tot/D_tot;
1+1;

end