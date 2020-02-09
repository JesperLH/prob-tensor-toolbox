%% Toolbox DEMO: Tensor Completion

miss_fractions = [0:0.1:0.9,0.92:0.02:0.98];
%% Example 1: Chemometrics
load('../Data/sugar_Process/data.mat')
X = reshape(X, DimX);

[train_err, test_err] = demo_missing_analysis(...
    X,9,miss_fractions);

%% Example 2: Electroencephalogram (EEG) data
% - 


%% Example 3: Human Allen Data