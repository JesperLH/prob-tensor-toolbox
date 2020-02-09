function visualizeDenoising(X_raw, X_recon, label_sources, show_diff)

N = ndims(X_raw); % Modes
M = length(X_recon);


nrow=M+1;
ncol=size(X_raw,1)*(1+show_diff);


for r = 1:nrow
    
    for i_samp = 1:ncol/(1+show_diff)
        subplot(nrow,ncol,i_samp + (r-1)*ncol)
        if r == 1
            surf(squeeze(X_raw(i_samp,:,:)));
            title(sprintf('Sample %i',i_samp));
        else
            surf(squeeze(X_recon{r-1}(i_samp,:,:)));
        end
        axis tight
        grid off
        set(gca,'color','none')
        set(gca,'YTick',[], 'XTick', [])%, 'ZTick',[])
        set(gca,'view', [-37.5, 30])
        colormap parula
        shading flat% interp
               
        if i_samp==1
            zlabel(strcat(label_sources{r},{'  '}),...
                'Rotation',0,'Fontsize',12,'HorizontalAlignment','right')
        end
        
        if r > 1
            pos = get(gca,'Position');
            set(gca,'Position',pos+[0,0.1*(r-1),0,-0.04]);
        end
        
        % Display residual error
        if show_diff && r>1
            subplot(nrow,ncol,i_samp + (r-1)*ncol+ncol/2)
            surf(squeeze(X_raw(i_samp,:,:))...
                -squeeze(X_recon{r-1}(i_samp,:,:)))
            if r==2
                title(sprintf('Residual\n Sample %i',i_samp))
            end
        end
        
        axis tight
        grid off
        set(gca,'color','none')
        set(gca,'YTick',[], 'XTick', [])%, 'ZTick',[])
        set(gca,'view', [-37.5, 30])
        colormap parula
        shading flat% interp
        
        if r > 1
            pos = get(gca,'Position');
            set(gca,'Position',pos+[0,0.1*(r-1),0,-0.04]);
        end
    end
    %         v = get(gca,'view');
    %         xh = get(gca,'XLabel'); % Handle of the x label
    %         set(xh, 'Units', 'Normalized')
    %         set(xh, 'Position',[0.57, 0.03,0],'Rotation',v(2)+5)
    %
    %         yh = get(gca,'YLabel'); % Handle of the y label
    %         set(yh, 'Units', 'Normalized')
    %         set(yh, 'Position',[0.35,0.03, 0],'Rotation',v(1)-9)
    
end

end