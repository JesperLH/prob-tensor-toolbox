%% Demonstration of CP
% - Requires the Amino Acid dataset (available from
% http://www.models.life.ku.dk/Amino_Acid_fluo) and the Sugar Process data
% (from http://www.models.life.ku.dk/Sugar_Process).

setup_paths;
clear; close all;
rng(0,'twister')

for i_sc = 1:4
    
    rng(0,'twister')
    % Load data
    % Amino
    if i_sc == 1
        load ../Data/Amino_Acid_fluo/amino.mat; D = 10; max_iter = 100; name_dataset='aminoacid';
    elseif i_sc == 2
        load ../Data/Amino_Acid_fluo/amino.mat; D = 100; max_iter = 100; name_dataset='aminoacid';
        
    elseif i_sc == 3
        % Sugar
        load ../Data/sugar_Process/data.mat; D = 10; max_iter = 500; name_dataset='sugar';
    elseif i_sc == 4
        load ../Data/sugar_Process/data.mat; D = 100; max_iter = 500; name_dataset='sugar';
        
    end
    
    X = reshape(X,DimX);
    name_modes = {'Samples', 'Emission', 'Excitation'};
    X = X/sqrt(var(X(:)));
    
    % Specify constraint
    constr = {'normal ard','normal ard', 'normal ard'};
    %% Run
    A_best = []; model_best=[]; lambda_best = []; max_elbo = -inf;
    for i=1:5
        [A,A2,lambda, cost, model] = VB_CP_ALS(X,D,constr,'maxiter',max_iter,...
            'inference','variational','model_tau',false,'model_lambda',true);
        if cost(end)>max_elbo
            A_best=A; model_best=model; lambda_best = lambda; max_elbo = cost(end);
        end
    end
    A=A_best; model=model_best; lambda = lambda_best;
    
    %% Illustrate Results
    save_folder = 'demo_toolbox_paper/img/';
    mkdir(save_folder);
    
    figure('Position',[50,300,1200,200])
    font_size=13;
    for i = 1:1
        [A,comp_scale] = scaleAndSignAmb(A_best);
        %     A = A_best;
        
        [~, i_ard] = min(find(my_contains(constr,'ard')));
        if isempty(i_ard)
            idx_min = 1:D;
        else
            [~, idx_min] = sort(lambda{i_ard},'ascend');
        end
        
        for n = 1:ndims(X)
            subplot(1,ndims(X)+1,n);
            plot(A{n}(:,idx_min),'LineWidth',1.5);
%             title(sprintf('Mode %i: %s',n,name_modes{n}));
            title(sprintf('Mode %i',n));
            xlabel(sprintf('%s',name_modes{n}))
            if n==1
                ylabel('Latent activation')
            end
            axis tight
            set(gca,'FontSize',font_size)
        end
        subplot(1,ndims(X)+1,n+1);
        if ~isempty(i_ard)
            bar(sqrt(1./lambda{i_ard}(:,idx_min))); axis tight
            title('ARD')
            xlabel('Components')
            ylabel('s.d')
            set(gca,'FontSize',font_size)
        end
    end
    
    save_currentfig(sprintf('%s%s_cp_D%i',save_folder,name_dataset,D))
    
end