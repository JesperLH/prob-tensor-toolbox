% Demo Amino Acid
clear
addpath(genpath('./myclasses'))
addpath(genpath('./tools'))
addpath('./myMatlab/')
addpath('../thirdparty-matlab/nway331/') % Download from http://www.models.life.ku.dk/nwaytoolbox

% Load data
% Amino
load ../Data/Amino_Acid_fluo/amino.mat; D = 8; max_iter = 50;

% Sugar
%load ../Data/sugar_Process/data.mat; D = 10; max_iter = 500;

X = reshape(X,DimX);
name_modes = {'Samples', 'Emission', 'Excitation'};
X = X/sqrt(var(X(:)));

% Specify constraints
constr = {'normal ard', 'normal ard', 'normal ard'};

% Note, one can also enforce non-negativity, but the ARD process doesn't
% work quite as well as in the unconstrained case. This is an issue with
% the column-wise updating of non-negative factors.
% constr = {'nonneg ard', 'nonneg ard', 'nonneg ard'};
% X=addTensorNoise(X,5);

%% Run
[A,AtA,lambda, cost, model] = VB_CP_ALS(X,D,constr,'maxiter',max_iter,...
    'inference','variational','model_tau',true);

%% Visualize solution
for i = 1:1
    Asc = scaleAndSignAmb(A);
    
    figure('Position',[100,400,800,400])
    [i_ard] = min(find(my_contains(constr,'ard')));
    if isempty(i_ard)
        idx_min = 1:D;
    else
        [~, idx_min] = sort(lambda{i_ard},'ascend');
    end

    for n = 1:ndims(X)
        subplot(2,ndims(X)+1,n);
        plot(Asc{n}(:,idx_min));
        title(sprintf('Mode-%i, %s',n,name_modes{n}))

        subplot(2,ndims(X)+1,n+ndims(X)+1);
        plot(Asc{n}(:,idx_min(1:3)));
    end
    subplot(2,ndims(X)+1,n+1);
    if ~isempty(i_ard)
        bar((lambda{i_ard}(:,idx_min))); ylim([0,300])
    end
end
%%
figure('Position',[1000,400,800,400])
for n = 1:ndims(X)
    subplot(1,ndims(X),n);
    %muA = model.factors{n}.getExpFirstMoment();
    CI = model.factors{n}.getCredibilityInterval([0.025,0.975]);
    
    
    for d = idx_min(1:3)
        plotCI(1:size(A{n},1),A{n}(:,d),squeeze(CI(:,d,:)))
        hold on
    end
    title(sprintf('Mode-%i, %s',n,name_modes{n}))
end

