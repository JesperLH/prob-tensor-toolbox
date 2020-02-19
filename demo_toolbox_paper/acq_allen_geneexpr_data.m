% Acquiring the Allen Human Brain Data
% Download data from
% http://human.brain-map.org/static/download

% Downloaded data location
dl_loc = './';

% Save processed data to
save_loc_and_name = '../Data/processed_data_allen';

% Folder prefix and filename.
name_prefix = strcat(dl_loc,'normalized_microarray_donor');
name_surfix = '/MicroarrayExpression.csv';

% Subject id
subject_id = [9861,10021,12876,14380,15496,15697];

%% Load the data for each subject.
microarray = cell(6,1);
names = cell(6,1);
ids = cell(6,1);
sample_coordinates = cell(6,1);
n_genes = 58692;
n_areas = 414;
n_subs = 6;
count = zeros(13008,n_subs); %Ids run from 1 to 100
for sub = 1:n_subs
    tic
    fprintf('Subject %i loading... ',sub)
    filename = sprintf('%s%i%s',name_prefix,subject_id(sub),name_surfix);
    microarray{sub} = csvread(filename);
    sampleannot = readtable(strcat(sprintf('%s%i%s',name_prefix,subject_id(sub),'/'),'SampleAnnot.csv'));
    
    names{sub} = sampleannot.structure_name;
    ids{sub} = sampleannot.structure_id;   
    sample_coordinates{sub} = [sampleannot.mni_x,sampleannot.mni_y,sampleannot.mni_z,...
                               sampleannot.mri_voxel_x , sampleannot.mri_voxel_y, sampleannot.mri_voxel_z];
    toc
end

%% Order and scale the data
X = zeros(n_genes,13008,n_subs);
for sub = 1:n_subs
    tic
    for sample = 1:size(microarray{sub},2)-1
       X(:,ids{sub}(sample),sub) = X(:,ids{sub}(sample),sub)+microarray{sub}(:,sample+1);
       count(ids{sub}(sample),sub) = count(ids{sub}(sample),sub)+1;
    end
    scaleby = count(:,sub)';
    scaleby(scaleby==0) = eps;
    X(:,:,sub) = bsxfun(@times,X(:,:,sub),1./scaleby);
    toc
end
% Keep only areas that are observed in at least one subject
idx_keep = sum(nansum(X,3),1) > 0;
X=X(:,idx_keep,:);

%% Save area coordinates.
column_to_id = find(idx_keep);
coord_dist = nan(6,length(column_to_id),n_subs);
coord_names = {'mri_x','mri_y','mri_z','mri_voxel_x','mri_voxel_y','mri_voxel_z'};
for sub = 1:n_subs
    tic
    for sample = 1:size(sample_coordinates{sub},1)
       c_idx = find(column_to_id == ids{sub}(sample));
       coord_dist(:,c_idx,sub) = sample_coordinates{sub}(sample,:);
    end

    toc
end

%%
all_ids = ids{1}; for i = 2:6; all_ids = [all_ids;ids{i}]; end; length(unique(all_ids)) %#ok<AGROW>
all_areas = {}; for i = 1:6; all_areas = {all_areas{:},names{i}{:}}; end; length(unique(all_areas)) %#ok<CCAT>

[uni_areas, idx_a] = unique(all_areas);
uni_ids = all_ids(idx_a);
[sorted_ids,idx_sort] = sort(uni_ids);
sorted_areas = uni_areas(idx_sort);
features = sorted_areas;

%%
probes = readtable(strcat(sprintf('%s%i%s',name_prefix,subject_id(1),'/'),'Probes.csv'));
genes = probes.gene_symbol;

%%
save(save_loc_and_name,'X','features',...
    'genes','coord_dist','coord_names','-v7.3')