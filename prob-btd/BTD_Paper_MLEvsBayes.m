% Compare MLE BTD and Bayesian BTD
addpath('../thirdparty-matlab/tensorlab_2016-03-28/')

N = [40,40,40];
n_repeats=100;
noiselevels = [-20:2.5:40]; %[10,20,30]; % In dB20:5:30%

for bool_too_many_components = [true,false]
    btd_model_order = {[3,3,3], [3,3,3], [3,3,3],[3,3,3]};
% bool_too_many_components = false;
if bool_too_many_components
%     btd_model_order = cellfun(@(t) t+1, btd_model_order, 'UniformOutput', false);
btd_model_order = cellfun(@(t) t*2, btd_model_order, 'UniformOutput', false);
end

%%
label_methods = {'BTD-NLS', 'BTD-minf', 'pBTD'};
frob_err_fit = nan(length(noiselevels),3,n_repeats);
frob_err_noiseless = nan(length(noiselevels),3,n_repeats);
time_elapsed = nan(size(frob_err_noiseless));

for i_rep = 1:n_repeats
% f = @(t) repmat(ones(1,length(N))*t,12/t,1);
% btd_model_order = mat2cell(f(3),4,[1,1,1]);
U_true = btd_rnd(N,btd_model_order);
Xnoiseless = btdgen(U_true);



for i_noise = 1:length(noiselevels)
    %X = noisy(Xnoiseless,noiselevels(i_noise));
    X = addTensorNoise(Xnoiseless,noiselevels(i_noise));
%     X=X/max(X(:));

    %% Tensorlab BTD
    options.Display = 5;     % Show convergence progress every 5 iterations.
    options.MaxIter = 100;   % Set stop criteria.
    options.TolFun  = 1e-12;
    options.TolX    = 1e-12;
    U_init = btd_rnd(N,btd_model_order);
    if i_rep==1
        % Only run MLE BTD-NLS onces, as it takes 20-30 times as long as
        % BTD-minf and gives similar results.
        t0 = tic;
        [Uhat_nls,output] = btd_nls(X,U_init,options);
        time_elapsed(i_noise,1,i_rep) = toc(t0);
        
        frob_err_fit(i_noise,1,i_rep) = frobbtdres(X,Uhat_nls)/norm(X(:));
        frob_err_noiseless(i_noise,1,i_rep) = frobbtdres(Xnoiseless,Uhat_nls)/norm(Xnoiseless(:));
    end
    
    t0 = tic;
    [Uhat_minf,output] = btd_minf(X,U_init,options);
    time_elapsed(i_noise,2,i_rep) = toc(t0);
    frob_err_fit(i_noise,2,i_rep) = frobbtdres(X,Uhat_minf)/norm(X(:));
    frob_err_noiseless(i_noise,2,i_rep) = frobbtdres(Xnoiseless,Uhat_minf)/norm(Xnoiseless(:));

    %% Bayes BTD
    Dbtd = cat(1,btd_model_order{:});
    t0 = tic;
    best_elbo = -inf;
%     for irep = 1:5
        [EG, EU, elbo] = pt_Tucker_BTD(X,Dbtd); 
%         if elbo(end)>best_elbo
%             best_elbo=elbo(end);
%             EG=EG_;
%             EU = EU_;
%         end
%     end
    time_elapsed(i_noise,3,i_rep) = toc(t0);
    
    % Convert to tensorlab format
    Uhat_vb = cell(1,size(Dbtd,1));
    bla = cumsum([0,0,0;Dbtd])';      
    for ib = 1:size(Dbtd,1)
        for in = 1:size(Dbtd,2)
            Uhat_vb{ib}{in} = EU{in}(:,1+bla(in,ib):bla(in,ib+1));
        end
        Uhat_vb{ib}{end+1} = EG(1+bla(1,ib):bla(1,ib+1),...
                                1+bla(2,ib):bla(2,ib+1),...
                                1+bla(3,ib):bla(3,ib+1));
    end
    frob_err_fit(i_noise,3,i_rep) = frobbtdres(X,Uhat_vb)/norm(X(:));
    frob_err_noiseless(i_noise,3,i_rep) = frobbtdres(Xnoiseless,Uhat_vb)/norm(Xnoiseless(:));
    
    %%
    
    


end
end
%%
%{'black', 'orange', 'sky blue', 'bluish green', 'Yellow', 'blue', 'vermilion', 'reddish purple'}
list_safe_color = [0,0,0; 230, 159, 0; 86, 180, 233; 0, 158, 115; 240, 228, 66; 0, 114, 178; 213, 94, 0; 204, 121, 167]/255;
colormethods = list_safe_color([1,2,3],:);
colormethodsshade = list_safe_color([1,2,3],:)/2;
figure('Position',[50,50,400,300]); 
% subplot(1,5,1:2)
% plot(noiselevels,frob_err,'Linewidth',2)
% ylabel('Reconstruction error')
% xlabel('SNR dB')

% subplot(1,5,4:5)
% plot(noiselevels,frob_err_rel,'Linewidth',2)
% plot(noiselevels,nanmean(frob_err_rel,3),'Linewidth',2)
plot(noiselevels,nanmean(frob_err_noiseless(:,1,:),3),'Linewidth',2)
for i_met = 2:3
%     boxplot(squeeze(frob_err_noiseless(:,i_met,:))',noiselevels,'Colors',[1,0,i_met==2])
    
    mu = mean(squeeze(frob_err_noiseless(:,i_met,:)),2);
    sd = sqrt(var(squeeze(frob_err_noiseless(:,i_met,:)),[],2));
    plotCI(noiselevels,mu,mu + [-1,1].*sd(:),colormethods(i_met,:),colormethodsshade(i_met,:))
end
%

ylim([0,1.2])
ylabel('Relative error (noiseless data)')
xlabel('SNR dB')
%legend(label_methods,'Position',[0.42,0.32,0.14,0.3],'Orientation','vertical','FontSize',10)
legend(label_methods,'Location','northeast','fontsize',12)
%
if bool_too_many_components
    save_currentfig('BTDExperiments','synthetic-btd-mle-vs-bayes-too-many-D.png')
else
    save_currentfig('BTDExperiments','synthetic-btd-mle-vs-bayes.png')
end

figure; plot(nanmean(time_elapsed,3)); legend(label_methods); ylabel('seconds'); 
end
