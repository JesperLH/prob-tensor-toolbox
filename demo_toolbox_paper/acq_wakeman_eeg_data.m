%% Acquiring and processing of the Wakeman Dataset
% A through description of the data, pre-processing, and analysis is given
% in the SPM 12 Manual, Chapter 42 "Multimodal, Multisubject datafusion".
% This "script" simply describes how to get from the available data and
% original scripts to the EEG data used in the Tensor Completion
% experiment.

%% Acquiring the Wakeman Dataset
% The data is available through
% "ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson/wakemandg_hensonrn/"
% and the original scripts are available at
% "ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson/SPMScripts/"


%% Processing steps
% Each subject was processed as follows: 
% 1) A band pass filter (FIR) [1,35]Hz with filter order $\mathtt{ceil}(f \cdot 1.65)\cdot$
%    where $f$ is the sampling frequency, both directions was applied to each channel. 
% 2) The signal was resampled to 200 Hz. 
% 3) Trials were epoched considering the interval [-100,800]ms. 
% 4) Baseline estimated in the interval [-100,0]ms and removed from each trial. 
% 5) Rereferencing to average EEG/channel. 
% 6) The conditions Famous, Unfamiliar, and Scrambled were extracted.  
% 7) Detection and removal of artifacts (eye blinks) by thresholding EOC. 
% 8) Combine planar gradiometers. 
% 9) Artifact detection via averaging over trials.

% This preprocessing scheme is same as the one described in Chapter 42.3 of
% the SPM12 Manual (https://www.fil.ion.ucl.ac.uk/spm/ ), except for the
% addition of step 1) used to remove drift and high frequency activity.  
%
% This additional step is done by modifying the original "master_script.m"
% where "jobfile = {fullfile(scrpth,'batch_preproc_meeg_convert_job.m')};"
% is replaced with jobfile = {fullfile(scrpth,'new_batch_preproc_meeg_convert_job.m')};

% The file "new_batch_preproc_meeg_convert_job" contains the following instructions for SPM
%-------------------------- start file ---------------------------------------------
matlabbatch{1}.spm.meeg.convert.dataset = '<UNDEFINED>';
matlabbatch{1}.spm.meeg.convert.mode.epoched.trlfile = '<UNDEFINED>';
matlabbatch{1}.spm.meeg.convert.channels{1}.type = 'EEG';
matlabbatch{1}.spm.meeg.convert.channels{2}.type = 'MEGMAG';
matlabbatch{1}.spm.meeg.convert.channels{3}.type = 'MEGPLANAR';
matlabbatch{1}.spm.meeg.convert.outfile = '';
matlabbatch{1}.spm.meeg.convert.eventpadding = 0;
matlabbatch{1}.spm.meeg.convert.blocksize = 3276800;
matlabbatch{1}.spm.meeg.convert.checkboundary = 1;
matlabbatch{1}.spm.meeg.convert.saveorigheader = 0;
matlabbatch{1}.spm.meeg.convert.inputformat = 'autodetect';
matlabbatch{2}.spm.meeg.preproc.prepare.D(1) = cfg_dep('Conversion: Converted Datafile', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{2}.spm.meeg.preproc.prepare.task{1}.settype.channels{1}.chan = 'EEG061';
matlabbatch{2}.spm.meeg.preproc.prepare.task{1}.settype.channels{2}.chan = 'EEG062';
matlabbatch{2}.spm.meeg.preproc.prepare.task{1}.settype.newtype = 'EOG';
matlabbatch{2}.spm.meeg.preproc.prepare.task{2}.settype.channels{1}.chan = 'EEG063';
matlabbatch{2}.spm.meeg.preproc.prepare.task{2}.settype.newtype = 'ECG';
matlabbatch{2}.spm.meeg.preproc.prepare.task{3}.settype.channels{1}.chan = 'EEG064';
matlabbatch{2}.spm.meeg.preproc.prepare.task{3}.settype.newtype = 'Other';
matlabbatch{2}.spm.meeg.preproc.prepare.task{4}.setbadchan.channels{1}.chanfile = '<UNDEFINED>';
matlabbatch{2}.spm.meeg.preproc.prepare.task{4}.setbadchan.status = 1;
matlabbatch{3}.spm.meeg.preproc.filter.D(1) = cfg_dep('Prepare: Prepared Datafile', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{3}.spm.meeg.preproc.filter.type = 'fir';
matlabbatch{3}.spm.meeg.preproc.filter.band = 'bandpass';
matlabbatch{3}.spm.meeg.preproc.filter.freq = [1 35];
matlabbatch{3}.spm.meeg.preproc.filter.dir = 'twopass';
matlabbatch{3}.spm.meeg.preproc.filter.order = ceil(1100*3.3/2)*2;
matlabbatch{3}.spm.meeg.preproc.filter.prefix = 'f';
matlabbatch{4}.spm.meeg.preproc.downsample.D(1) = cfg_dep('Filter: Filtered Datafile', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{4}.spm.meeg.preproc.downsample.fsample_new = 200;
matlabbatch{4}.spm.meeg.preproc.downsample.method = 'resample';
matlabbatch{4}.spm.meeg.preproc.downsample.prefix = 'd';
matlabbatch{5}.spm.meeg.preproc.bc.D(1) = cfg_dep('Downsampling: Downsampled Datafile', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{5}.spm.meeg.preproc.bc.timewin = [-100 0];
matlabbatch{5}.spm.meeg.preproc.bc.prefix = 'b';
matlabbatch{6}.spm.meeg.other.delete.D(1) = cfg_dep('Prepare: Prepared Datafile', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{7}.spm.meeg.other.delete.D(1) = cfg_dep('Filter: Filtered Datafile', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{8}.spm.meeg.other.delete.D(1) = cfg_dep('Downsampling: Downsampled Datafile', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));

%--------------------------------- end file--------------------------------------

%% The following function can then be used to load data from each subject "i_subject"
function [X_eeg, trials, eeg_chanpos, eeg_label] = load_wakemann(i_subject, folder_loc)

file_name = 'PapMcbdfspmeeg_run_01_sss';

%     Load SPM proccessed header file
load(sprintf('%sSub%02i/MEEG/%s.mat',folder_loc,i_subject,file_name));

% Change file and folder reference
D.path = sprintf('%sSub%02i/MEEG/',folder_loc,i_subject,file_name);
D.data.fname = sprintf('%sSub%02i/MEEG/%s.dat',folder_loc,i_subject,file_name);

%% Get EEG and MEEG data
%X = D.data(:,:,:);
trials.bad = boolean([D.trials.bad])';
trials.label = {D.trials.label}';

remove_bad_trials=false;
if remove_bad_trials
    idx_keep_trial = find(~trials.bad);
    trials.label = trials.label(~trials.bad);
    trials.bad(trials.bad) = [];
else
    idx_keep_trial = 1:length(trials.label);
end

% Get EEG data
all_chan_labels = {D.channels.label}';
idx_eeg = find(contains(all_chan_labels,'EEG'));
X_eeg = double(D.data(idx_eeg,:,idx_keep_trial));

% Remove bad channels
bad_channels = [D.channels.bad]';
bad_channels = bad_channels(idx_eeg);


D.sensors; % eeg and meeg
eeg_label = D.sensors.eeg.label;
eeg_chanpos = D.sensors.eeg.chanpos;

%% Sort channels, so they are in ascending order
[~, idx_sort] = sort(arrayfun(@(s) str2num(strrep(s{1},'EEG','')), eeg_label));


% Apply sorting index
X_eeg = double(X_eeg(idx_sort,:,:));
eeg_chanpos = eeg_chanpos(idx_sort,:);
eeg_label = eeg_label(idx_sort);


% Remove HEOG, VEOG, ECG, and freefloating electrode channels
idx_bad_ch = [61,62,63,64];
X_eeg(idx_bad_ch,:,:) = [];
eeg_chanpos(idx_bad_ch,:) = [];
eeg_label([61,62,63,64]) = [];

end












