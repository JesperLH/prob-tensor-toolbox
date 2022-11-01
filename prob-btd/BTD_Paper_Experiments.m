%% BTD Paper Experiments
function BTD_Paper_Experiments(s_analysis, save_loc)
setup_paths;

if nargin < 1
    s_analysis = 'synthetic';
end

if nargin < 2
    save_loc = 'BTDExperiments';
end
if ~isfolder(save_loc), mkdir(save_loc); end


n_repeats = 10;

%%
[Xcomb, Gtrue, Utrue, Xnoiseless] = get_btd_data(s_analysis);


%% Fit BTD
% [EG, EU, elbo] = pt_Tucker_BTD(Xcomb,Dbtd); % known struct
% - TODOs -
%   * Sort blocks? (well not when specified..)
%   * Sort within block by superdiagonal

%% TEMP LOCATION !! Calling tensor lab - Check it even works for non (Lr,Lr,1)
% Vi skal selv have en inital tensor, lad os gøre som vi vore btd kode. Nb,
% strukturen af U_init afgør model_range
% U_init = {'tensorlab btd structure'} 
%addpath('../thirdparty-matlab/tensorlab_2016-03-28/')
% % Alternativt, genere fra tensorlab
% U_init = btd_rnd([size(Xcomb), model_range{im} + 'but as {[core 1], [core 2], ..}'])
% U_init = btd_rnd([19,18,17],{[3,4,1], [2,1,3], [3,3,3]});
% T = noisy(btdgen(U_init),0);
% options.Display = 5;     % Show convergence progress every 5 iterations.
% options.MaxIter = 200;   % Set stop criteria.
% options.TolFun  = 1e-12;
% options.TolX    = 1e-12;
% [Uhat,output] = btd_nls(T,U_init,options);
% 
% % NB. Levens lab shows that BTD sufferes from degeneracy - which the
% % Bayesian formulation should not! 
% semilogy(0:output.iterations,sqrt(2*output.fval));
% xlabel('iteration');
% ylabel('frob(btdres(T,U))');
% grid on;

%% CPD <- BTD -> Tucker
Nx = ndims(Xcomb);
model_range = arrayfun(@(t) repmat(ones(1,Nx)*t,12/t,1), [1,2,3,4,6,12], 'UniformOutput', false);
% model_range{end+1} = Dbtd;
% model_range{1} = [10,2,3; 2,2,2;1,1,5];
% model_range(2:end) = [];

estimate_model = cell(length(model_range),2);
performance_models = nan(length(model_range),2,n_repeats);

for im = 1:length(model_range)
    fprintf('Fitting %d cores with %d components each\n',size(model_range{im},1), model_range{im}(1))
%     success = false;
    best_elbo = -inf;
    for irep = 1:n_repeats
          
%         if ~success
            try
                [EG, EU, elbo] = pt_Tucker_BTD(Xcomb,model_range{im}); 
                performance_models(im,1,irep) = elbo(end);
                if strcmpi(s_analysis,'synthetic')
                    Xresidual = Xnoiseless-nmodel(EU, EG);
                    performance_models(im,2,irep) = sqrt(mean(Xresidual(:).^2));
                end
%                 success = true;
            catch
                warning('Model failed :-( for %i', im)
                elbo=nan;
            end
%         end
            if elbo(end) > best_elbo
                estimate_model(im,:) = {EG, EU};
            end
    end

    if strcmpi(s_analysis,'eeg')
        % Reshape time-freq to and image
        estimate_model{im,2}{3} = reshape(estimate_model{im,2}{3},61,72,size(estimate_model{im,2}{3},2));
    end

end



save(fullfile(save_loc,sprintf('estimated-models-%s',s_analysis)),...
    'estimate_model','performance_models','model_range', 's_analysis',...
    'Xcomb','Gtrue','Utrue','Xnoiseless')
%%
%
% figure
% subplot(1,2,1)
% elboend = nan(size(estimate_model,1),1);
% for i = find(~cellfun(@isempty, estimate_model(:,end)))'
%     hold all
%     semilogy(estimate_model{i,end})
%     elboend(i) = estimate_model{i,end}(end);
% end
% legend()
% subplot(1,2,2)
% bar(elboend)
%%
% figure
% im=3;
% plotBlockTermDecomposition(Res{im,1}, Res{im,2})


%% Show best decomposition
for im = 1:length(model_range)

    if ~isempty(estimate_model{im,1})
            if strcmpi(s_analysis,'synthetic')
                plotting_style = 'hinton';
            elseif strcmpi(s_analysis,'flowinj')
                plotting_style = {'hinton','plot','plot'};
            elseif strcmpi(s_analysis,'eeg')
                plotting_style = {'hinton','topo','image'};
            else
                plotting_style = 'plot';
            end
            model_perf(1) = max(squeeze(performance_models(im,1,:))); % ELBO
            str_model{1} = 'ELBO';
            if ~isempty(Xnoiseless)
                model_perf(2) = min(squeeze(performance_models(im,2,:))); % Reconstruction err
                str_model{2} = '\epsilon';
            end
            f  = figure('Position',[50,50,1600,1200]);        
            plotBlockTermDecomposition(estimate_model{im,1}, estimate_model{im,2},...
                str_model,model_perf,plotting_style)
            save_currentfig(save_loc,sprintf('%s-%icores-%isize',s_analysis, size(model_range{im},1),model_range{im}(1)))
            close(f)
    end
end

%% Some sort of prediction, maybe only for EEG
% here an example for amino flour ... which is a bad example
% Res{2,2}{1}*(Res{2,2}{1}\y)
% y
% corr(y,Res{2,2}{1}*(Res{2,2}{1}\y))

% keyboard

end

function [X, G_true, U_true, Xnoiseless] = get_btd_data(s)
% Get synthetic, flowinj, or eeg data
G_true = [];
U_true = [];
Xnoiseless = [];
s = lower(s);
switch s
    case 'synthetic'
        %% Simulated data
        N = [40,40,40];
        sim_constr = [7,7,7]; %  orthogonal
        % Dbtd = [2,3,4;3,2,4; 3,3,1; 2,2,1];
        f = @(t) repmat(ones(1,length(N))*t,12/t,1);
        Dbtd = f(3);
        Xcomb = 0;
        Ucomb = cell(length(N),size(Dbtd,1));
        Gcomb = cell(size(Dbtd,1),1);
        for ib = size(Dbtd,1):-1:1
            [X, U, G] = generateTuckerData(N,Dbtd(ib,:),sim_constr); 
            Xcomb = Xcomb +X;
            Ucomb(:,ib) = U;
            Gcomb{ib} = G;
        end
        % Do it differently, so U is orthogonal
        % XX = nmodel({cat(2,Ucomb{1,:}),cat(2,Ucomb{2,:}),cat(2,Ucomb{3,:})}, blkdiag_tensor(Gcomb,Dbtd));
        U = {orth(cat(2,Ucomb{1,:})),orth(cat(2,Ucomb{2,:})),orth(cat(2,Ucomb{3,:}))};
        Xnoiseless = nmodel(U, blkdiag_tensor(Gcomb,Dbtd));

        bla = cumsum([0,0,0;Dbtd])';
        for in = 1:size(Dbtd,2)
            for ib = 1:size(Dbtd,1)
%                 1+bla(in,ib):bla(in+1,ib)
                Ucomb{in,ib} = U{in}(:,1+bla(in,ib):bla(in,ib+1));
            end
        end
        
        % noise? also noise full or noise blockwise
        X = addTensorNoise(Xnoiseless,20);
%         s_analysis = 'synthetic'
        G_true = Gcomb;
        U_true = Ucomb;

        % indicator = @(t,k) [zeros(1,floor(length(t)/6)*(k-1)), ones(1,floor(length(t)/6)), zeros(1,length(t)-floor(length(t)/6)*k)]
        % T = sin(x.*(1:12)')'; T=T./sqrt(sum(T.^2,1)); T'*T;

    case 'flowinj'
        %% Flow
        load('../Data/life_ku/Flow-Injection\fia.mat','X','DimX')
        X = reshape(X,DimX);
    
    case 'eeg'
        %% EEG
        s_analysis = 'eeg'
        filenames = dir('../Data/Tutorialdataset2/*.mat'); filenames = {filenames.name};
        Xcomb = cell(length(filenames),1);
        chanlocs_all = cell(length(filenames),1);
        for i = 1:length(filenames)
            load(sprintf('../Data/Tutorialdataset2/%s',filenames{i}),'WT','chanlocs');
            Xcomb{i} = WT(:,:); % Unfold time x freq to time-freq
            chanlocs_all{i} = chanlocs;
        end
        Xcomb = permute(abs(cat(3,Xcomb{:})), [3,1,2]);
        Xcomb = Xcomb-mean(Xcomb,2); % Substract channel mean
%         Xcomb(:,:,500:end) = []; % TEMP!!!
        X = Xcomb; 
        % 
        % % TOPOPLOTS EEG
        addpath('../thirdparty-matlab/eeglab2021.1/');
        evalc('eeglab;'); close all;
end
end
