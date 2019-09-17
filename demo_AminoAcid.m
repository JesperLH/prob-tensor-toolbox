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

% Specify constraint
constr = {'nonneg ard','nonneg constr', 'nonneg constr'};

%% Run
[A,A2,lambda, cost, model] = VB_CP_ALS(X,D,constr,'maxiter',max_iter,...
    'inference','variational','model_tau',true);

%%
for i = 1:1
    A = scaleAndSignAmb(A);
    
    figure
    [~, i_ard] = min(find(my_contains(constr,'ard')));
    if isempty(i_ard)
        idx_min = 1:D;
    else
        [~, idx_min] = sort(lambda{i_ard},'ascend');
    end

    for n = 1:ndims(X)
        subplot(2,ndims(X)+1,n);
        plot(A{n}(:,idx_min));
        title(sprintf('Mode-%i, %s',n,name_modes{n}))

        subplot(2,ndims(X)+1,n+ndims(X)+1);
        plot(A{n}(:,idx_min(1:3)));
    end
    subplot(2,ndims(X)+1,n+1);
    if ~isempty(i_ard)
        bar((lambda{i_ard}(:,idx_min))); ylim([0,300])
    end
end
%%
figure
for n = 1:ndims(X)
    subplot(1,ndims(X),n);
    CI = model.factors{n}.getCredibilityInterval([0.025,0.975]);
    A = model.factors{n}.getExpFirstMoment();

    for d = idx_min
        plotCI(1:size(A,1),A(:,d),squeeze(CI(:,d,:)))
        hold on
    end

end
% 
% for n = 1:ndims(X)
%     CI = model.factors{n}.getCredibilityInterval();
%     subplot(2,ndims(X)+1,n);
%     B =  squeeze(CI(:,idx_min,3));
%     B_95 = CI(:,idx_min,[2,4]);
%     
%     plot(B);
%     title(sprintf('Mode-%i, %s',n,name_modes{n}))
% 
%     subplot(2,ndims(X)+1,n+ndims(X)+1);
%     for j = 2:2%size(B,2)
%         errorbar(1:size(B,1), B(:,j), squeeze(B_95(:,j,1)),squeeze(B_95(:,j,2)))
%     end
%         
%     
% end
% subplot(2,ndims(X)+1,n+1);
% bar((lambda{i_ard}))

return
%%
A = model.factors{2}.getExpFirstMoment();
CI = model.factors{2}.getCredibilityInterval([0.025,0.975]);
x = A(:,1);
ci = squeeze(CI(:,1,:));
figure; 
subplot 221, plot(x)
subplot 222, plot(1:length(x),x,'k', 'LineWidth',2); hold on
             plot(1:length(x),ci(:,1),'r');
             plot(1:length(x),ci(:,3),'b'); hold off
subplot 223, plotCI(1:length(x),x,ci(:,[1,3]))

