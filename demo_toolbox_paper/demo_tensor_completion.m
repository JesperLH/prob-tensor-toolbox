%% Toolbox DEMO: Tensor Completion
D_est = 10;
miss_fractions = [0.1:0.1:0.9,0.92:0.02:0.98];
%% Example 1: Chemometrics
load('../Data/life_ku/Sugar_Process/data.mat')%../Data/sugar_Process/data.mat')
X = reshape(X, DimX);
t0n=tic;
[train_err_sugar, test_err_sugar] = demo_missing_analysis(...
    X,D_est,miss_fractions);

time_sec_sugar = toc(t0n)
save('sugar-res','train_err_sugar','test_err_sugar','miss_fractions','time_sec_sugar') % ca. 

%% Example 2: Electroencephalogram (EEG) data
% - 
load('../Data/Facescramble-Matlabform/facescramble-Sub07.mat')
data = data(:,:,1:idx_condition(2)); % Only the first condition

t0n=tic;
[train_err_eeg, test_err_eeg] = demo_missing_analysis(...
    data,D_est,miss_fractions,'eegdata');

time_sec_eeg = toc(t0n)
save('eeg-res','train_err_eeg','test_err_eeg','miss_fractions','time_sec_eeg') % ca.

%% Example 3: Human Allen Data
clear data
load('../Data/Genetics/AllenBrainAtlas/processed_allen_data.mat');
X(X(:)==0)=nan; % Missing values
t0n=tic;
[train_err_allen, test_err_allen] = demo_missing_analysis(...
    X,D_est,miss_fractions,'allendata');

time_sec_allen = toc(t0n)
save('allen-res','train_err_allen','test_err_allen','miss_fractions','time_sec_allen') % ca.


