function [avg_opt, opt_set, opt_measure] = optimal_component_match(measures)
% Match
% measures: is a D_1 x D_2 matrix (D_2>=D_1), where the measure is between the estimated
% components (rows) and true components (columns)

[D1,D2] = size(measures);
%assert(all(size(measures)==D),'The input was not a square matrix!')
assert(all(measures(:)>0),'A measure was negative, this is not supported.')

% Beware of memory issues, since all the permutations and their measures is stored
s = sprintf(strcat('All permutations for D > 11 is to memory intensive. \nYour ',...
    ' choice (D1=%i,D2=%i) would require %6.2f GB of memory'),D1,D2,...
    2*(factorial(D1)*D2*8/1024^3)); % One double is 8 bytes
assert(D1<=11,s);
assert(2*(factorial(D1)*D2*8/1024^3) < 14, s); % max 14 GB of space..


psets = perms(1:D1); % All permutations
p_aucs = zeros(length(psets),D1);
for i = 1:length(psets)
   % linear indexing to get the right measures
   p_aucs(i,:) = measures(psets(i,:)+ones(1,D1).*(D1*(0:D1-1)));
end

[~, opt_idx] = max(sum(p_aucs,2));
opt_set = psets(opt_idx,:)';
opt_measure=p_aucs(opt_idx,:)';
avg_opt = mean(opt_measure);


end