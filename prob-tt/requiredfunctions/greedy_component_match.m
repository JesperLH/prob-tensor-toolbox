function [avg_measure, idx, matched_measure] = greedy_component_match(measures)
% Match
% measures: is a D x D matrix, where the measure is between the estimated
% components (rows) and true components (columns)

D = size(measures,2);
assert(all(size(measures)==D),'The input was not a square matrix!')
assert(all(measures(:)>0),'A measure was negative, this is not supported.')

idx = nan(D,1);
matched_measure = nan(D,1);
for d = 1:D
    % Find the value and column index of the maximum value
    [max_meas,tmatch] = max(max(measures,[],1));
    % Find the estimated components which produced this match
    [~,test] = max(measures(:,tmatch)); 
    
    % save it
    idx(test) = tmatch;
    matched_measure(test) = max_meas;
    
    % Set this component as matched
    measures(:,tmatch) = 0;
    measures(test,:) = 0;
end
avg_measure = mean(matched_measure);

end