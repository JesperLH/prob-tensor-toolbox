function [amari,spatial_corr,fig_handle] = histAmariCorr(Atrue,Aest,method_name,varargin)

% Get input parameters
paramNames = {'bins','fontsize','position','save'};
defaults   = {20,12,[0,0,0.12,.5], ''};

[bins, f_size, position, save_folder]...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});

D = size(Aest,2);
colors = distinguishable_colors(D,[1,1,1; 0,0,0]);
figure('units','Normalized','Position',position);

% Calculate histogram bin-centers and counts
counts = nan(bins,D);
centers = nan(bins,D);
for d = 1:D
    [counts(:,d),centers(:,d)]= hist(Aest(:,d),bins);
end

%Standardize to proportions
counts = bsxfun(@rdivide,counts,sum(counts)); %Get proportion

%Plot histograms
xlim = [min(centers(:)),max(centers(:))];
ylim = [0,max(counts(:))];
for d = 1:D
    subplot(D,1,d);
    bar(centers(:,d),counts(:,d),'facecolor',colors(d,:));
    %title(sprintf('Comp. %i',d),'Fontsize',f_size+2);
    set(gca,'fontsize',f_size);
    
    axis([xlim,ylim*1.05])
    text(xlim(1)*0.98,ylim(2)*0.97,...
        sprintf('k=%2.2f',kurtosis(Aest(:,d))-3),'fontsize',f_size);
    if d == ceil(D/2);
        ylabel('Proportions');
    end
    
    if d ~= D
        set(gca,'Xticklabel',[]);
    end
end
set(gca,'Xticklabel',[]);
%xlabel('Values')

% Match
amari = amariDist(Atrue,Aest,D);
spatial_corr = calcMatchedCorrelation(Atrue,Aest);

fig_handle = gcf;

if ~isempty(save_folder)
    tightfig();
    print(sprintf('%s%s%s%s',save_folder,'eps/',method_name,'_hist'),'-depsc');
    print(sprintf('%s%s%s%s',save_folder,'eps/',method_name,'_hist'),'-dpng');
    fout = fopen([save_folder method_name '_distance.txt'],'w');
    fprintf(fout,'Amari: %6.4e \tSpatial corr.: %6.4e',amari,spatial_corr);
    fclose(fout);
else
    suptitle(method_name);
end

end