function [avg_opt, opt_set, opt_measure] = optimal_component_match(measures)
% Match
% measures: is a D x D matrix, where the measure is between the estimated
% components (rows) and true components (columns)

D = size(measures,2);
assert(all(size(measures)==D),'The input was not a square matrix!')
assert(all(measures(:)>0),'A measure was negative, this is not supported.')

% Beware of memory issues, since all the permutations and their measures is stored
s = sprintf(strcat('All permutations for D > 11 is to memory intensive. \nYour ',...
    ' choice (D=%i) would require %6.2f GB of memory'),D,...
    2*(factorial(D)*D*8/1024^3)); % One double is 8 bytes
assert(D<=11,s);

psets = perms(1:D); % All permutations
p_aucs = zeros(length(psets),D);
for i = 1:length(psets)
   % linear indexing to get the right measures
   p_aucs(i,:) = measures(psets(i,:)+ones(1,D).*(D*(0:D-1)));
end

[~, opt_idx] = max(sum(p_aucs,2));
opt_set = psets(opt_idx,:)';
opt_measure=p_aucs(opt_idx,:)';
avg_opt = mean(opt_measure);


end