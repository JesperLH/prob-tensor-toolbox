% Overview / Reproducable script for BTD paper

%% 1) Synthetic MLE vs Bayes BTD

BTD_Paper_MLEvsBayes(false); % Right number of components
BTD_Paper_MLEvsBayes(true); % Double the number of components.

%% 2) Real data
BTD_Paper_Experiments('flowinj');
BTD_Paper_Experiments('eeg');

%% 2) Synthetic Bayes BTD
model_CD={'(12,1)','(6,2)','(4,3)','(3,4)','(2,6)', '(1,12)'};
for s_model = model_CD
    BTD_Paper_Experiments(sprintf('synthetic-BTD%s',s_model{1}));
end

% Create table
res_elbo = zeros(length(model_CD));
res_residual = zeros(length(model_CD));

% Gather results
for i = 1:length(model_CD)
    load(sprintf('BTDExperiments/estimated-models-synthetic-BTD%s',model_CD{i}),'performance_models','Xnoiseless')
    1+1;
    [res_elbo(i,:),imax] = max(squeeze(performance_models(:,1,:)),[],2);
    resi = squeeze(performance_models(:,1,:));
    for j = 1:length(model_CD)
        res_residual(i,j) = performance_models(j,2,imax(j));
    end
    res_residual(i,:) = res_residual(i,:)/sqrt(mean(Xnoiseless(:).^2));
end
res_elbo = res_elbo./abs(max(res_elbo,[],2)); % Make it relative to the maximum value
%%
s = '';

for irow = 0:length(model_CD)*2
    if irow == 0
        s = sprintf('%s %%----------- Scaled ELBO -------\n (C,D)',s);
    elseif irow == length(model_CD)+1
        s = sprintf('%s %%----------- Relative Residuals -------\n',s);
    end
    if irow ~=0
        s = sprintf('%s BTD-%s',s,model_CD{irow-((irow>length(model_CD))*length(model_CD))});
    end
    for icol = 1:length(model_CD)
        if irow == 0 
            s = sprintf('%s & %s',s,model_CD{icol});
        elseif irow <= length(model_CD) % elbo
            s = sprintf('%s & %6.4f',s,res_elbo(irow,icol));
        else % residual
            s = sprintf('%s & %6.4f',s,res_residual(irow-length(model_CD),icol));
        end
        
    end
    
    s= sprintf('%s\\\\\n',s);
   

end
disp(s)



