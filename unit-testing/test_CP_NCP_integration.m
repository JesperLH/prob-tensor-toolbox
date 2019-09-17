runs=30;
%% Model passer med data generering (NCP)
for i = 1:runs
    X = generateTensorData([50,40,30],5,[6,6,6]); X = addTensorNoise(X,5); X = X/sqrt(nanvar(X(:)));
    constr = {'ard','ard','ard'};
    idx=randperm(numel(X));
    X(idx(1:floor(numel(X)*0.1)))=nan; clear idx
    VB_CP_ALS(X,5,constr,'model_tau',true, 'model_lambda', true);
    fprintf('Run %i of %i\n\n',i,runs)
    pause(1)
end


%% Model passer med data generering (Wishart)
for i = 1:runs
    X = generateTensorData([50,40,30],5,[6,6,6]); X = addTensorNoise(X,5); X = X/sqrt(nanvar(X(:)));
    constr = {'wishart','wishart','wishart'};
    idx=randperm(numel(X));
    X(idx(1:floor(numel(X)*0.0001)))=nan; clear idx
    VB_CP_ALS(X,4,constr,'model_tau',true, 'model_lambda', true);
    fprintf('Run %i of %i\n\n',i,runs)
    pause(1)
end

%%
for i = 1:runs
    X = generateTensorData([50,40,30],5,[6,6,3]); X = addTensorNoise(X,5);% X = X/sqrt(nanvar(X(:)));
    constr = {'wishart','wishart','ard'};
    idx=randperm(numel(X));
    X(idx(1:floor(numel(X)*0.0001)))=nan; %clear idx
    VB_CP_ALS(X,4,constr,'model_tau',true, 'model_lambda', true);
    fprintf('Run %i of %i\n\n',i,runs)
    pause(1)
end