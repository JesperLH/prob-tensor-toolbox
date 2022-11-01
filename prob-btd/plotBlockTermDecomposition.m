function plotBlockTermDecomposition(gest,aest,str_perform, val_perform, plotstyles)
%% PLOTBLOCKTERMDECOMPOSITION plots the core array and factors from a 3-way block term decomposition.
%
% Input:
%   gest:       The estimated core array
%   aset:       The estimated factor matrices
%   gtruth:     The ground truth core array or [] if unknown.
%   gtruth:     The ground truth factor matrices or [] if unknown.
%   plotstyles: A cell array with the plotstyles for each mode

if nargin < 3
    gtrue = [];
    atrue = [];
    plotstyles = repmat({'plot'},1,ndims(gest));
end
if isstring(plotstyles) || ischar(plotstyles)
    plotstyles = repmat({plotstyles},1,ndims(gest));
end

assert(ndims(gest)<=3,'Function only works up to 3-way data')
if size(aest{1},2) ~= 12
    warning('This function has been developed and tested for D=12 only - visualization might look strange')
end


nrow=6; ncol=8;
i_core = [1:4,9:12,17:20];
% i_factors = [3,2,4];
i_factors = {[25:28,33:36,41:44], [5:8,13:16,21:24],[29:32,37:40,45:48]};

%% Plot the core array
% subplot(nrow,ncol,i_core)
% if ~strcmpi(plotstyles,'hinton')
    subplot(nrow,ncol,i_core)
    plotTCorefast(gest);
    pos_core = get(gca,'Position');
%     set(gca,'Position',[0.05,0.32,0.33+0.22,0.22+0.22])
    set(gca,'Position',[0.17,0.6,0.22+0,0.22+0])

% end
%% Also plot performance here
if length(val_perform) == 2
    text(-0.05,1.05-0.05*length(val_perform),sprintf('%s:  %6.4e \n',str_perform{1},val_perform(1),str_perform{2},val_perform(2)),'Units','normalized','FontSize',24)
else
    text(-0.05,1.05-0.05*length(val_perform),sprintf('%s:  %6.4e \n',str_perform{1},val_perform(1)),'Units','normalized','FontSize',24)
end

%% Plot factor matrics
for ifac = 1:length(aest)
%     subplot(nrow,ncol,i_factors(ifac))
    subplot(nrow,ncol,i_factors{ifac})
    
    if strcmpi(plotstyles{ifac},'plot')
        plot(aest{ifac},'Linewidth',2);
        axis off;
    elseif strcmpi(plotstyles{ifac},'hinton')
        hintonw(aest{ifac}'); %axis off; 
%         axis off
        set(gca,'color',[1 1 1]);
        set(gcf,'color',[1 1 1]);
        set(gca,'ydir','normal');
        set(gca,'xdir','normal');
        [S,R] = size(aest{ifac}');
        plot([0 R R 0 0]+0.5,[0 0 S S 0]+0.5,'k');
        axis off;
    elseif strcmpi(plotstyles{ifac},'topo')
        load('./btd_eeg_chanlocs.mat','eeg_chanlocs');
        B = aest{ifac};
        c_axis_lim = [min(B(:)), max(B(:))];
        for f = 1:size(B,2)
            if any(B(:,f))
                subplot(nrow,ncol,i_factors{ifac}(f));
                topoplot(B(:,f),eeg_chanlocs); caxis(c_axis_lim);
                title(sprintf('%i',f),'FontSize',14)
            end
        end
        %colorbar('Position',[0.25,0.05,0.5,0.05],'Orientation','Horizontal')
        colorbar('Position',[0.92,0.55,0.03,0.4],'FontSize',13) % This works for factor 2
        %colorbar('eastoutside')
        set(gca,'color',[1 1 1]);
        set(gcf,'color',[1 1 1]);
        1+1;
    elseif strcmpi(plotstyles{ifac},'image')
        B = aest{ifac};
        c_axis_lim = [min(B(:)), max(B(:))];
        for f = 1:size(B,3)
            if any(B(:,f))
                subplot(nrow,ncol,i_factors{ifac}(f));
                imagesc(B(:,:,f)); caxis(c_axis_lim);
                title(sprintf('%i',f),'FontSize',14)
            end
            axis off
        end
        colorbar('Position',[0.92,0.1,0.03,0.4],'Fontsize',13)
    end
    
    pos = get(gca,'Position');
    if ifac==1 % buttom            
        set(gca,'View',[90,90]);
        %set(gca,'Position',[0.25,0.05,0.12,0.35]);
        set(gca,'Position',pos-[0,0.06,0,0]);
    elseif ifac==2 % right
        set(gca,'View',[0,90]);
        %set(gca,'Position',[0.2,0.5-0.12/2,0.35,0.12]);
    elseif ifac==3 % third mode
        if ~strcmpi(plotstyles{ifac},'image')
            set(gca,'View',[45,90])
        end
        %set(gca,'Position',[0.25,0.65,0.15,0.35]);
%         set(gca,'Position',[0.45,0.15,0.4,0.4]);
%         set(gca,'Position',[0.5,0.1,0.4,0.4]);
        set(gca,'Position',pos-[0,0.06,0,0])
    end
    
    if ~strcmpi(plotstyles{ifac},'topo') && ~strcmpi(plotstyles{ifac},'image')
        title(sprintf('Latent Factors Mode %i',ifac),'FontSize',18)
    end
end

% for ifac = 1:length(aest)
%     subplot(nrow,ncol,i_factors(ifac))
% %     imagesc(aest{ifac})
%     pos = get(gca,'Position');
%     
%     if strcmpi(plotstyles,'plot')
%         plot(aest{ifac});
% %         yticklabels([]); xticklabels([])
%         if ifac==1 % buttom            
%             set(gca,'View',[90,90]);
%             %set(gca,'Position',[0.25,0.05,0.12,0.35]);
%         elseif ifac==2 % right
%             set(gca,'View',[0,90]);
%             %set(gca,'Position',[0.2,0.5-0.12/2,0.35,0.12]);
%         else % top
%             set(gca,'View',[45,90])
%             %set(gca,'Position',[0.25,0.65,0.15,0.35]);
%             set(gca,'Position',[0.45,0.15,0.4,0.4]);
%         end
%         axis off; % Removes both background and box lining - TODO: Include box again? or draw rectangle?
% 
%     elseif strcmpi(plotstyles,'hinton')
%         hintonw(aest{ifac}'); %axis off; 
% %         axis off
%         set(gca,'color',[1 1 1]);
%         set(gcf,'color',[1 1 1]);
%         set(gca,'ydir','normal');
%         set(gca,'xdir','normal');
%         [S,R] = size(aest{ifac}');
%         plot([0 R R 0 0]+0.5,[0 0 S S 0]+0.5,'k');
%         if ifac==1 % buttom
%     %         plotTCorefast(permute(aest{ifac},[3,2,1]));
%     %         set(gca,'Position',pos+[-0.01,0.17,0,0])
%             
%             set(gca,'View',[90,90]);
% %             set(gca,'Position',[0.22,0.05,0.12,0.35]);
%         elseif ifac==2 % right
%     %         plotTCorefast(aest{ifac}');
%     %         set(gca,'Position',pos-[0.35,-0.02,0,0])
%             set(gca,'View',[0,90]);
% %             set(gca,'Position',[0.5,0.5-0.12/2,0.35,0.2]);
%         else % top
%     %         plotTCorefast(aest{ifac});
%     %         set(gca,'Position',pos-[-0.02,0.20,0,0])
%             set(gca,'View',[45,90]);
%             set(gca,'Position',[0.5,0.1,0.4,0.4]);
%         end
%         axis off
%     end
%     title(sprintf('Latent Factor Mode %i',ifac),'FontSize',15)
% end
1+1;
%%

%%
% if strcmpi(plotstyles,'hinton')
%     subplot(nrow,ncol,i_core)
%     plotTCorefast(gest);
%     pos_core = get(gca,'Position');
%     %set(gca,'Position',[0.05,0.35,0.33+0.18,0.22+0.18])
%     set(gca,'Position',[0.17,0.6,0.22+0,0.22+0])
% % end
%%
1+1;
% box off


%% Calculate and plot performance?


