%% Toolbox DEMO: Tensor Completion
D_est = 10;
miss_fractions = [0.1:0.1:0.9,0.92:0.02:0.98];
run_no = 1;

run_chemometrics = true;
run_eeg = true;
run_allen = true;

%% Example 1: Chemometrics
if run_chemometrics
    load('../Data/life_ku/Sugar_Process/data.mat')%../Data/sugar_Process/data.mat')
    X = reshape(X, DimX); X=X/sqrt(var(X(:)));
    t0n=tic;
    [train_err_sugar, test_err_sugar] = demo_missing_analysis(...
        X,D_est,miss_fractions);

    time_sec_sugar = toc(t0n)
    save(sprintf('sugar-res-%i',run_no),'train_err_sugar','test_err_sugar','miss_fractions','time_sec_sugar') % ca. 
end
%% Example 2: Electroencephalogram (EEG) data
% - 
if run_eeg
    load('../Data/Facescramble-Matlabform/facescramble-Sub07.mat')
    data = data(:,:,1:idx_condition(2)); % Only the first condition
    data = data/sqrt(var(data(:)));

    t0n=tic;
    [train_err_eeg, test_err_eeg] = demo_missing_analysis(...
        data,D_est,miss_fractions,'eegdata');

    time_sec_eeg = toc(t0n)
    save(sprintf('eeg-res-%i',run_no),'train_err_eeg','test_err_eeg','miss_fractions','time_sec_eeg') % ca.
end

%% Example 3: Human Allen Data
if run_allen
    load('../Data/Genetics/AllenBrainAtlas/processed_allen_data.mat');
    %X=X(1:500:end,:,:);
    X(X(:)==0)=nan; % Missing values
    X=X/sqrt(nanvar(X(:)));
    t0n=tic;
    [train_err_allen, test_err_allen] = demo_missing_analysis(...
        X,D_est,miss_fractions,'allendata');

    time_sec_allen = toc(t0n)
    save(sprintf('allen-res-%i',run_no),'train_err_allen','test_err_allen','miss_fractions','time_sec_allen') % ca.
end


%%
        
