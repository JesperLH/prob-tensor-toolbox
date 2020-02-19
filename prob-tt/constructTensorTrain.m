function X = constructTensorTrain(G)
%% CONSTRUCTTENSORTRAIN takes a tensor train G and multiplies it into the 
% full data array.

X = squeeze(G{1});
for i=2:length(G)
    X = ntimes(X, G{i}, ndims(X), 1);
end

end