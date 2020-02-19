%% Toolbox DEMO: Tensor Completion
% Note. This demo requires the following datasets:
% Example 1: Sugar Process Data (from http://www.models.life.ku.dk/Sugar_Process)
% Example 2: EEG data from Subject 7 from the Wakeman dataset (see script acq_allen_geneexpr_data.m )
% Example 3: The Allen Human Brain data (see script acq_wakeman_eeg_data.m )
load_sugar = '../Data/life_ku/Sugar_Process/data.mat';
load_eeg = '../Data/Facescramble-Matlabform/facescramble-Sub07.mat';
load_allen = '../Data/Genetics/AllenBrainAtlas/processed_allen_data.mat';
    
setup_paths;

D_est = 10;
miss_fractions = [0.1:0.1:0.9,0.92:0.02:0.98];
run_no = 1;

run_chemometrics = true;
run_eeg = true;
run_allen = true;

save_folder = './demo_toolbox_paper/';

%% Example 1: Chemometrics
if run_chemometrics
    load(load_sugar)
    X = reshape(X, DimX); X=X/sqrt(var(X(:)));
    t0n=tic;
    [train_err_sugar, test_err_sugar] = run_missing_analysis(...
        X,D_est,miss_fractions);

    time_sec_sugar = toc(t0n)
    save(sprintf('%ssugar-res-%i',save_folder,run_no),'train_err_sugar','test_err_sugar','miss_fractions','time_sec_sugar') % ca. 
end
%% Example 2: Electroencephalogram (EEG) data
% - 
if run_eeg
    load(load_eeg)
    data = data(:,:,1:idx_condition(2)); % Only the first condition
    data = data/sqrt(var(data(:)));

    t0n=tic;
    [train_err_eeg, test_err_eeg] = run_missing_analysis(...
        data,D_est,miss_fractions,'eegdata');

    time_sec_eeg = toc(t0n)
    save(sprintf('%seeg-res-%i',save_folder,run_no),'train_err_eeg','test_err_eeg','miss_fractions','time_sec_eeg') % ca.
end

%% Example 3: Human Allen Data
if run_allen
    load(load_allen);
    X=X(1:500:end,:,:);
    X(X(:)==0)=nan; % Missing values
    X=X/sqrt(var(X(:)));
    t0n=tic;
    [train_err_allen, test_err_allen] = run_missing_analysis(...
        X,D_est,miss_fractions,'allendata');

    time_sec_allen = toc(t0n)
    save(sprintf('%sallen-res-%i',save_folder,run_no),'train_err_allen','test_err_allen','miss_fractions','time_sec_allen') % ca.
end


%% Visualize results

miss_schemes = {'Missing Elements', 'Missing Fibers'};
for i=[1,2,3]
    
    if i == 1
        f_list = dir('./demo_toolbox_paper/sugar*'); f_list = {f_list.name};
        load(sprintf('./demo_toolbox_paper/%s',f_list{1}))
        train_err = zeros([length(f_list), size(train_err_sugar)]);
        test_err = zeros(size(train_err));
        for i_f = 1:length(f_list)
            load(sprintf('./demo_toolbox_paper/%s',f_list{i_f}));
            train_err(i_f,:) = train_err_sugar(:);
            test_err(i_f,:) = test_err_sugar(:);
        end
        s_title = 'Sugar';
        
        
    elseif i == 2
        f_list = dir('./demo_toolbox_paper/eeg*'); f_list = {f_list.name};
        load(sprintf('./demo_toolbox_paper/%s',f_list{1}))
        train_err = zeros([length(f_list), size(train_err_eeg)]);
        test_err = zeros(size(train_err));
        for i_f = 1:length(f_list)
            load(sprintf('./demo_toolbox_paper/%s',f_list{i_f}));
            train_err(i_f,:) = train_err_eeg(:);
            test_err(i_f,:) = test_err_eeg(:);
        end
        s_title = 'EEG';
        
    elseif i == 3
        f_list = dir('./demo_toolbox_paper/allen*'); f_list = {f_list.name};
        load(sprintf('./demo_toolbox_paper/%s',f_list{1}))
        train_err = zeros([length(f_list), size(train_err_allen)]);
        test_err = zeros(size(train_err));
        for i_f = 1:length(f_list)
            load(sprintf('./demo_toolbox_paper/%s',f_list{i_f}));
            train_err(i_f,:) = train_err_allen(:);
            test_err(i_f,:) = test_err_allen(:);
        end
        s_title = 'Allen';
        miss_fractions= (1-0.56)+0.56.*(miss_fractions);
    end
    train_err = squeeze(nanmedian(train_err,1));
    test_err = squeeze(nanmedian(test_err,1));
    
%     
    for i_sc = 1:length(miss_schemes)
        figure('Position',[10,10,600,400])
        l_colors = [1,0,0;0,0,1];
        hold on
        for i_mod=1:2% models
            plot(miss_fractions*100, train_err(:,i_sc,i_mod,1),...
                '-','Color',l_colors(i_mod,:),'LineWidth',2)
            plot(miss_fractions*100, test_err(:,i_sc,i_mod,1),...
                '--','Color',l_colors(i_mod,:),'LineWidth',2)
        end
        y = [train_err(:,i_sc,:,1), test_err(:,i_sc,:,1)];
        ylim([nanmin(y(:)),min(nanmax(y(:)),2)])
        xlim([0,100])
        set(gca,'FontSize',16)
        xlabel('Percentage Missing')
        ylabel('RMSE')

        title(sprintf('%s: %s',s_title,miss_schemes{i_sc}))
        save_currentfig('./demo_toolbox_paper/img/',...
            strrep(sprintf('%s-%s',s_title,miss_schemes{i_sc}),' ','-'));
    end
    
    if i==3 
        legend({'VB', 'VB (test)', 'Gibbs', 'Gibbs (test)'},'Location','northoutside','Orientation','horizontal')
        save_currentfig('./demo_toolbox_paper/img/','legend-horizontal');
    end
end
        
        
        
