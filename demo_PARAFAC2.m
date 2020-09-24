%% Demo Probabilistic PARAFAC2
addpath('../parafac2-multiple-varying-modes/'); % nmodel_pf2
addpath('../auto-gcgc/thirdparty-toolboxes/'); % parafac2_gcgc
clear variables
s_scenario = '3way';
s_scenario = '4way';
% s_scenario = '4way-array';
% s_scenario = '5way';
% s_scenario = 'amino';
% s_scenario = 'amino-array';

D=5;
addnoise = false;

if strcmpi(s_scenario, '3way')
    [X,A,P] = generateDataParafac2([10,3],[7,6,5],D);
    
elseif strcmpi(s_scenario, '4way')
    [X,A,P] = generateDataParafac2([10,9,3],[7,6,5],D);
elseif strcmpi(s_scenario, '4way-array')
    [X,A,P] = generateDataParafac2([10,9,3],[7,7,7],D);
    X=cat(ndims(X{1})+1,X{:});
    
elseif strcmpi(s_scenario, '5way')
    [X,A,P] = generateDataParafac2([10,9,8,3],[7,6,5],D);
    
elseif strcmpi(s_scenario, 'amino')
    load('../Data/Amino_Acid_fluo/amino.mat');
    Y = permute(reshape(X,DimX),[2,3,1]);
    K=size(Y,3);
    clear X;
    for k = K:-1:1
        X{k} = Y(:,:,k);
    end
elseif strcmpi(s_scenario, 'amino-array')
    load('../Data/Amino_Acid_fluo/amino.mat');
    X = permute(reshape(X,DimX),[2,3,1]);
    
else
    error('No data loaded!')
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


%% Run PARAFAC2
pf2_constraint = []
if iscell(X)
     pf2_constraint = repmat({'normal ard'},1,ndims(X{1})+1);
    pf2_constraint = repmat({'nonneg const'},1,ndims(X{1})+1);
%     X = cellfun(@abs, X, 'UniformOutput',false);
%     
%     pf2_constraint{1} = 'orthogonal';
%     pf2_constraint{1} = 'normal ard';
%     pf2_constraint{1} = 'normal wishart';
%     pf2_constraint{1} = 'nonneg expo ard';
%     pf2_constraint{1} = 'nonneg expo sparse';
end

[EA,Pk,elbo] = pt_PARAFAC2(X,D,pf2_constraint,'maxiter',20);

%
figure('Position',[100,500,600,400])
subplot(2,2,1); plot(EA{1}); 
subplot(2,2,2); plot(EA{3}); title('Concentrations')
subplot(2,2,3); plot(Pk{1});
subplot(2,2,4); plot(Pk{2});

if exist('A','var')
    figure('Position',[700,500,600,400])
    subplot(2,2,1); plot(A{1}); 
    subplot(2,2,2); plot(A{3}); title('Concentrations')
    subplot(2,2,3); plot(P{1});
    subplot(2,2,4); plot(P{2});
    suptitle('Ground truth')
end





