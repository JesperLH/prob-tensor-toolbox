clear; clc
%% Generate data
% The sum over D is 1
% "runs" indicate how many instances of A was estimated.
%
V = 231; D = 3; runs = 2;
hasMutualInformation = true;
poverlap = 0.30; % Approximate overlap between runs

A = rand(D,V,runs).^3+0.0001;
%The first dimension must sum to 1 for each D and runs to be a probability
%density distribution. 
A = bsxfun(@rdivide,A,sum(A,1)); 

%Simulate overlap
if hasMutualInformation
    overlap=zeros(runs,1); %#ok<UNRCH>
    overlap(1) = 1;
    for r = 2:runs
        overlap(r)=(rand*(1-poverlap)+poverlap);
        A(:,1:floor(V*overlap(r)),r) = A(:,1:floor(V*overlap(r)),1);
    end
end

%% Mutual information
fprintf(['The mutual information (MI) is the amount of shared information',...
         ' between two matrices\n\n'])
fprintf('MI(A_1,A_1) = %1.6f, MI(A_1,A_2) = %1.6f\n',...
        calcMI(A(:,:,1),A(:,:,1)),calcMI(A(:,:,1),A(:,:,2)))

fprintf(['\nWhile the mutual (shared) information is higher between A_1 and itself ',...
         'than between A_1 and A_2,\nit is not possible to directly relate to '...
         'the amount of information shared vs. unshared between A_1 and A_2 .\n\n']);

%% Normalized mutual information
fprintf(['However, for the normalized MI the amount of shared information',...
         ' from the matrix A_1 to itself is 1 (i.e. 100%%).\nWhich then gives'...
         ' a contrast to the NMI between A_1 and A_2 .\n\n'])
fprintf('NMI(A_1,A_1) = %1.6f, NMI(A_1,A_2) = %1.6f\n\n',...
        calcNMI(A(:,:,1),A(:,:,1)),calcNMI(A(:,:,1),A(:,:,2)))
    
%% NMI permutation test
fprintf('--- Performing permutation test ... '); tic;

numberOfPermutations = 600;
[nmi,nmi_perm]=nmiPermutationTest(A,numberOfPermutations);
toc;

fprintf(['Average NMI over all pairwise combinations of runs : \t%1.6f',...
         '\nAverage NMI over all permutations (%i permutations for each',...
         ' pairwise combination) : %1.6f\n\n'],mean(nmi),numberOfPermutations,...
         mean(nmi_perm))
     
figure
hist(nmi_perm, 20); hold on
xlabel('Normalized Mutual Information')
ylabel('Count')
title(sprintf('Histogram of NMI values for %i permutations',numberOfPermutations))

if hasMutualInformation
    fprintf(['The pairwise combinations have higher average NMI than the ',...
             'permutations, as mutual information is present.\n'])
else
    fprintf(['The pairwise combinations have similar average NMI to the ',...
             'permutations average NMI, as mutual information is absent.\n'])
end