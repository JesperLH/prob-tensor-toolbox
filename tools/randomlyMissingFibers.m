function [X_train, X_test] =...
    randomlyMissingFibers(data, missing_fraction, i_mode_missing)
% RANDOMLYMISSINGFIBERS Splits the data into a training and test set by
% randomly removing fibers. The size of the testset is determined by
% "missing_fraction" [0,1]. The fiber is drawn uniformly at random for each
% mode (unless "i_mode_missing" is stated) and indices in the remaining
% modes are also drawn uniformly.

X_train = data;
X_test = nan(size(data));
n_modes = ndims(data);

n_present_elements = sum(~isnan(data(:)));
n_elements_to_remove = floor(n_present_elements*missing_fraction);

idx_present_fibers = cell(n_modes,1);
for i = 1:n_modes
    idx_present_fibers{i} = find(sum(~isnan(matricizing(data,i)),2)>0);
end

%% Remove fibers at random
num_failed_tries = 0;
while n_elements_to_remove > 0
    
    % Determine mode the fiber comes from
    if nargin < 3 || isempty(i_mode_missing)
        i_mode = randi(n_modes);
    else
        % always remove fibers from this (or these) mode(s).
        idx_rand = randperm(length(i_mode_missing));
        i_mode = i_mode_missing(idx_rand(1)); 
    end
    
    % Get random indices for the remaining modes
    idx = cell(n_modes,1);
    for i=1:n_modes
        if i==i_mode
            idx{i} = 1:size(data,i_mode);
        else
            idx{i} = randi(size(data,i));
        end
    end
    
    % Remove the data (and test if removal is valid.
    X_train(idx{:}) = nan;
    if ~any(all(isnan(matricizing(X_train,i_mode)),2))
        X_test(idx{:}) = data(idx{:});
        n_elements_to_remove = n_elements_to_remove-sum(~isnan(data(idx{:})));
        num_failed_tries = 0;
    else
        % Removed an entire slice or more, put it back and try again
        X_train(idx{:}) = data(idx{:});
        num_failed_tries = num_failed_tries+1;
    end
    if num_failed_tries > 100
        warning('Cound not generated valid missing pattern! Returning empty arrays')
        X_train=[];
        X_test=[];
        return
    end
%     assert(num_failed_tries < 100, 'Could not generated a vaild missing pattern.')
end

end