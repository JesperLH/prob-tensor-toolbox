%% Demonstration of CP with Heteroscedastic noise on the Amino Acid data
% - Requires the Amino Acid dataset (available from http://www.models.life.ku.dk/Amino_Acid_fluo)
setup_paths;

clear
rng(0,'twister')

% Load data
% Amino Acid Dataset
load ../Data/Amino_Acid_fluo/amino.mat; dataset='aminoacid';
X = reshape(X,DimX);
name_modes = {'Samples', 'Emission (nm)', 'Excitation (nm)'};
xaxis_modes = {1:5, linspace(250,450,size(X,2)), linspace(250,300,size(X,3))};

X = X/sqrt(var(X(:)));
D = 10; max_iter = 500;
noise_scheme = [1,1,1]; % Heteroscedastic noise on each mode.
% noise_scheme = [0,1,0]; % Heteroscedastic noise on the second mode.

% Specify constraint
constr = {'nonneg ard','nonneg constr', 'nonneg constr'};

%% Run
A_best = []; model_best=[]; lambda_best = []; max_elbo = -inf;
for i=1:5
    [A,~,lambda, cost, model] = VB_CP_ALS(X,D,constr,'maxiter',max_iter,...
        'inference','variational','model_tau',true,'noise',noise_scheme);
    if cost(end)>max_elbo
        A_best=A; model_best=model; lambda_best = lambda; max_elbo = cost(end);
    end
end
A=A_best; model=model_best; lambda = lambda_best;

%% Plot
save_folder = './demo_toolbox_paper/img/';
mkdir(save_folder)

figure('Position',[50,300,1200,400])
font_size=13;
for i = 1:1
%     [A,scale] = scaleAndSignAmb(A_best);
    A = A_best;
    
    [~, i_ard] = min(find(my_contains(constr,'ard')));
    if isempty(i_ard)
        idx_min = 1:D; idx_sort=1:D;
    else
        [~, idx_sort] = sort(lambda{i_ard},'ascend');
        idx_min =idx_sort(lambda{i_ard}(idx_sort)<1000); % Only active components
    end

    for n = 1:ndims(X)
        subplot(2,ndims(X)+1,n);
        if ~isempty(xaxis_modes)
            x = xaxis_modes{n};
        else
            x = 1:size(X,n);
        end
            
        plot(x,A{n}(:,idx_min),'LineWidth',1.5);
%         title(sprintf('Mode %i: %s',n,name_modes{n}))
        title(sprintf('Mode %i',n))
        %xlabel(sprintf('%s',name_modes{n}))
        xticks([])
        if n==1
            ylabel('Latent Score')
        end
        axis tight
        set(gca,'FontSize',font_size)
        
        % Noise
        subplot(2,ndims(X)+1,n+ndims(X)+1);
        plot(x,sqrt(1./model.noise{n}.getExpFirstMoment()),'LineWidth',1.5);
        xlabel(sprintf('%s',name_modes{n}))
        if n==1
            ylabel('Noise s.d.')
        end
        axis tight
        set(gca,'Position',get(gca,'Position') + [0.01,0.17,-0.01,-0.05])
        set(gca,'FontSize',font_size)
        
    end
    subplot(2,ndims(X)+1,n+1);
    if ~isempty(i_ard)
        bar(sqrt(1./lambda{i_ard}(:,idx_sort))); axis tight
        title('Component s.d.')
        xlabel('Components')
        ylabel('s.d.')
        set(gca,'FontSize',font_size)
    end
end

save_currentfig(sprintf('%s%s-hetero_cp_D%i',save_folder,dataset,D))
