function [muA, covA] = computeUnfoldedMoments(first_moment, second_moment)

% For one mode
if ~iscell(first_moment) && ismatrix(first_moment)
    [muA, covA] = computeUnfoldedMoments({first_moment}, {second_moment});
    muA = muA{1};
    covA = covA{1};
    
% For multiple modes
else
    muA = cell(1,length(first_moment));
    covA = cell(1,length(first_moment));

    for i = 1:length(first_moment)
        [N,D] = size(first_moment{i});

        muA{i} = zeros(N,D^2);

        for d = 1:D
            %muA{i}(:,D*(d-1)+(1:D)) = first_moment{i}(:,1:end).*first_moment{i}(:,d);  
            muA{i}(:,D*(d-1)+(1:D)) = bsxfun(@times,first_moment{i}(:,1:end),first_moment{i}(:,d));
        end
        
        if ~isempty(second_moment{1})

            if ndims(second_moment{i}) == 3
                covA{i} = second_moment{i}(:,:);
            elseif ismatrix(second_moment{i})
                if size(second_moment{i},1) == size(second_moment{i},2) % I.e. full covariance
                    covA{i} = second_moment{i}(:)'; % Row vector
                else
                    covA{i} = zeros(N,D^2);
                    covA{i}(:,D*((1:D)-1)+(1:D)) = second_moment{i}; % Only diagonal is nonzero
                end
            else
                error('No missing, why are we even here?')
            end
        else
            covA{i} = 0;
            
        end

    end
    
end
end