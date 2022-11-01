%% Demo Probabilistic PARAFAC2
addpath('../parafac2-multiple-varying-modes/'); % nmodel_pf2
addpath('../auto-gcgc/thirdparty-toolboxes/'); % parafac2_gcgc
addpath('../thirdparty-matlab/DataSim/Data/')
clear variables
s_scenario = '3way';
s_scenario = '4way';
% s_scenario = '4way-array';
% s_scenario = '5way';
% s_scenario = 'amino';
% s_scenario = 'amino-array';

D=4;
addnoise = true;
disp_plot=true;

if contains(s_scenario, 'amino','IgnoreCase',true)
    load('../Data/Amino_Acid_fluo/amino.mat');
    Y = permute(reshape(X,DimX),[2,3,1]);
    K=size(Y,3);
    clear X;
    for k = K:-1:1
        X{k} = Y(:,:,k);
    end
    n_obs = [size(X{1},1), length(X)];
else
    K=2;
    if strcmpi(s_scenario, '3way')
        max_n_obs = 15;
        max_m_obs = 10;
        n_fixedmodes = 1;
    elseif contains(s_scenario, '4way','IgnoreCase',true)
        max_n_obs = 15;
        max_m_obs = 10;
        n_fixedmodes = 2;
    elseif strcmpi(s_scenario, '5way')
        max_n_obs = 15;
        max_m_obs = 10;
        n_fixedmodes = 3;

    else
        error('No data loaded!')
    end

    
    n_obs = [max_n_obs:-1:(max_n_obs-n_fixedmodes+1),K];
    m_obs = max_m_obs-K+1:max_m_obs;
end


% Constraints
pf2_constraint = [];
pf2_constraint = repmat({'normal ard'},1,length(n_obs)+1);
% pf2_constraint = repmat({'nonneg const'},1,length(n_obs)+1);
% pf2_constraint = repmat({'nonneg expo const'},1,length(n_obs)+1);
% pf2_constraint = repmat({'normal const'},1,length(n_obs)+1);
%     X = cellfun(@abs, X, 'UniformOutput',false);
%     
%     pf2_constraint{1} = 'orthogonal';
% pf2_constraint{1} = 'normal constr';
%     pf2_constraint{1} = 'normal wishart';
%     pf2_constraint{1} = 'nonneg expo ard';
% pf2_constraint{1} = 'nonneg expo constr';

% Generate data
if ~exist('X','var')
    [X,A,P] = generateDataParafac2(n_obs,m_obs,D,pf2_constraint);
end


% Turn into array form
if contains(s_scenario, 'array','IgnoreCase',true)
    X=cat(ndims(X{1})+1,X{:});
end

%%
% [A,H,C,P,fit,AddiOutput]=parafac2_gcgc(X,D,[1,1]);
if addnoise
    try 
        X=addTensorNoise(X,10);
    catch
        K = length(X);
        for k = 1:K
            X{k} = addTensorNoise(X{k},10);
        end
    end
end

try
    X=X/sqrt(var(X(:)));
catch
    X = cellfun(@(t) t/sqrt(var(t(:))), X,'UniformOutput',false);
end

%% Run PARAFAC2
[EA,Pk,elbo] = pt_PARAFAC2(X,D,pf2_constraint,'maxiter',100,'init_method','');%,'inference','sampling');
% [EA,Pk,elbo] = pt_PARAFAC2(X,D,pf2_constraint,'maxiter',100);

if disp_plot
    %
    figure('Position',[100,500,600,400])
    subplot(2,2,1); plot(EA{1}); 
    subplot(2,2,2); plot(EA{3});% title('Concentrations')
    subplot(2,2,3); plot(Pk{1});
    subplot(2,2,4); plot(Pk{2});

    if exist('A','var')
        figure('Position',[700,500,600,400])
        subplot(2,2,1); plot(A{1}); 
        subplot(2,2,2); plot(A{3}); %title('Concentrations')
        subplot(2,2,3); plot(P{1});
        subplot(2,2,4); plot(P{2});
        suptitle('Ground truth')
    end
end





