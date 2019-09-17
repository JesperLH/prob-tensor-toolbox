%% Calculates Normalized Mutual Information (NMI)
% score of A and performs a permutation test across runs
%
% 
function [nmi,nmi_perm]=nmiPermutationTest(A,numPermutations)
% A is a q x d x runs matrix
% q is number of components
% d is the dimensionality (or subject times voxel)
% runs are the number of different solutions which are being compared
% numPermutations is the number of permutations to perform
%%%
%% TODO: 
%   * Allow evaluation metric to be passed
%   * Asses test statistic
%   * Parallize for-loop
if nargin < 2 % Only calculate NMI
    numPermutations = 0;
end


[~,d,runs] = size(A);
n_combi = sum(1:(runs-1)); %number of combinations
nmi = nan(n_combi,1);
nmi_perm = nan(n_combi*numPermutations,1);

placeNMI = 1;
placeNMIperm = 1;

for i = 1:runs
    for j = i+1:runs
        nmi(placeNMI) = gather(calcNMI(A(:,:,i),A(:,:,j)));
        placeNMI = placeNMI +1;
        for k = 1:numPermutations
            nmi_perm(placeNMIperm) = gather(calcNMI(A(:,:,i),A(:,randperm(d),j)));
            placeNMIperm = placeNMIperm+1;
        end
    end
end