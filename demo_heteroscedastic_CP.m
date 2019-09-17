%% Demo CP with heteroscedastic noise
n_obs = [50,40,30,20];
constr = {'normal wishart','nonneg ard','nonneg ard','normal wishart'};
noise_scheme = ones(1,length(n_obs));
snr_db = 0;
create_missing = false;


% Generate data
X=generateTensorData(n_obs,5,[6,6,3,6]); 
[X_noisy,~,~,~,noise_precision]=addTensorNoise(X,snr_db, noise_scheme);


if create_missing
    % Set 10% of the values to missing
    idx = randperm(numel(X));
    X_noisy(idx(1:round(numel(X)*0.1))) = nan;
end

%%
[A,A2,lambda, elbo, model]=VB_CP_ALS(X_noisy,5,constr(1:ndims(X_noisy))...
     ,'model_tau', true, 'fixed_tau', 15,...
     'model_lambda', true, 'fixed_lambda',10,...
     'maxiter',100, 'noise',[1,1,1,1], 'conv_crit',1e-7);

%%
figure('units','normalized','position',[0.3,0.4,0.6,0.4])
for idx = 1:length(n_obs)
    if ~isempty(model.noise{idx})
        Ni =n_obs(idx);
        subplot(1, length(n_obs),idx)
        hold on
        plot(1:Ni,noise_precision{idx}'/median(noise_precision{idx}'),'r','linewidth',1)
        plot(1:Ni, model.noise{idx}.getExpFirstMoment()'/median(model.noise{idx}.getExpFirstMoment()'),'b')
        title(sprintf('corr(sim,est) = %3.2f',corr(model.noise{idx}.getExpFirstMoment(),noise_precision{idx})))
        hold off
        legend({'Simulated noise', 'Estimated noise'})
    end
end