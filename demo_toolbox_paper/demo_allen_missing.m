function [X_train, X_test] = demo_allen_missing(X,missing_fraction)

if nargin < 1
%     load('../Data/Genetics/AllenBrainAtlas/processed_allen_data.mat');
end
X(isnan(X(:)))=0;
[X_mvd, train_idx, test_idx] = getMinimumViableDataset(X);
%%
total_obs_elements = sum(X(:)~=0);
current_obs_elements = sum(~isnan(X_mvd(:)));

obs_per_sample = size(X,1); %Genes

%%
requires_obs_elements = total_obs_elements*(1-missing_fraction);
additional_obs = requires_obs_elements-current_obs_elements;

new_samples = floor(additional_obs/obs_per_sample);
[X_train, train_idx, test_idx] = sampleNewAreas(X,X_mvd,new_samples);
X_test = nan(size(X_train));
X_test(test_idx(:)) = X(test_idx(:));

end



function [X_mvd, train_idx, test_idx] = getMinimumViableDataset(X)
[n_genes, n_areas, n_subj] = size(X);
observed_area = squeeze(sum(X~=0,1)~=0) > 0;
fprintf('Number of observed areas is %i (of %i areas)\n',sum(observed_area(:)),numel(observed_area))

%% Minimum Viable Dataset (MVD)
% - Each subject at least once
% - Each area at least once
X_mvd = nan(size(X));

%prior_prob = sum(observed_area,1)/sum(observed_area(:));
if all(n_areas == sum(observed_area,1))
    prior_prob = ones(1,n_areas)/n_areas;
else
    prior_prob = n_areas-sum(observed_area,1); 
    prior_prob = prior_prob/sum(prior_prob);
end

% For each area, sample a subject.
for i = 1:n_areas
    present_area = find(observed_area(i,:));
%     idx_sub = randi([1,length(present_area)],1);
%     person = present_area(idx_sub);
    if length(present_area) > 1
        person = datasample(present_area,1,'weights',prior_prob(present_area)/sum(prior_prob(present_area)));
    else
        person = present_area;
    end

    X_mvd(:, i, person) = X(:, i, person);
end

observed_area_Y = squeeze(nansum(X_mvd,1)) > 0;

fprintf('Number of observed areas is %i (of %i areas)\n',sum(observed_area_Y(:)),numel(observed_area))

%% Create train_idx and test_idx
train_idx = false(size(X));
test_idx = false(size(X));
for i = 1:size(train_idx,2)
    for j = 1:size(train_idx,3)
        
        if observed_area_Y(i,j) == 1
            train_idx(:,i,j) = true;
        elseif observed_area(i,j) == 1
            test_idx(:,i,j) = true;
        end
    end
end

% train_idx = (X_mvd>0);
% test_idx = (X>0) & ~train_idx;
end

function [X, train_idx, test_idx] = sampleNewAreas(X_full, X, num_samples)

[n_genes, n_areas, n_subj] = size(X);

%observed_area = squeeze(sum(X_full,1)) > 0;
observed_area = squeeze(nansum(X_full~=0,1)) > 0;
%observed_area_Y = squeeze(sum(X,1)) > 0;
observed_area_Y = squeeze(nansum(X,1)~=0) > 0;

possible_samples = observed_area & ~observed_area_Y;
possible_sample_idx = find(possible_samples);
assert(length(possible_sample_idx)>=num_samples, 'Not enough samples.')
sample_idx = randperm(length(possible_sample_idx));
sample_idx = possible_sample_idx(sample_idx(1:num_samples));
for i = 1:num_samples
    [i_area,j_subj] = ind2sub([n_areas, n_subj],sample_idx(i));
    assert(possible_samples(i_area, j_subj) == 1, 'Error! Trying to sample a non-existing sample')
    X(:,i_area,j_subj) = X_full(:,i_area,j_subj);
end
observed_area_Y = squeeze(sum(X,1)) > 0;

fprintf('Number of observed areas is %i (of %i areas)\n',sum(observed_area_Y(:)),numel(observed_area))

% Create training and test indices
%train_idx = (X>0);
%test_idx = (X_full>0) & ~train_idx;
train_idx = false(size(X));
test_idx = false(size(X));
for i = 1:size(train_idx,2)
    for j = 1:size(train_idx,3)
        
        if observed_area_Y(i,j) == 1
            train_idx(:,i,j) = true;
        elseif observed_area(i,j) == 1
            test_idx(:,i,j) = true;
        end
    end
end


end