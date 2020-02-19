function visualizeDenoising(X_raw, X_recon, label_sources, show_diff)

N = ndims(X_raw); % Modes
M = length(X_recon);


nrow=1+M*(1+show_diff);
ncol=size(X_raw,1);


for r = 1:M+1
    
    for i_samp = 1:ncol
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
        zt = get(gca,'ZTick');
        set(gca,'ZTick',[zt(1),median(zt([1,end])),zt(end)])
%         if r > 1
            pos = get(gca,'Position');
            set(gca,'Position',pos+[0,0.045*(r-1),0,-0.01 ]);
%         end
        
    end
end
for rw = 1:M
    r = rw+M+1;
    if show_diff
        for i_samp = 1:ncol
            
            
            % Display residual error
            if show_diff && r>1
                subplot(nrow,ncol,i_samp + (r-1)*ncol)
                surf(squeeze(X_raw(i_samp,:,:))...
                    -squeeze(X_recon{rw}(i_samp,:,:)))
            end
            
            axis tight
            grid off
            set(gca,'color','none')
%             set(gca,'YTick',[], 'XTick', [])%, 'ZTick',[])
            set(gca,'view', [-37.5, 30])
        
            colormap parula
            shading flat% interp
            
            if rw == 1 && i_samp==3
                title('Sample Specific Reconstruction Error')
            end
            
            
            if r > 1
                pos = get(gca,'Position');
                set(gca,'Position',pos.*[1,1,1,1]+[0,0.04*(r-1)-0.05,0,0]);
%                 set(gca,'Position',pos+[0,0.04*(r-1)-0.05,0,0]);
            end
            if i_samp==1
                zlabel(strcat(label_sources{rw+1},{'  '}),...
                    'Rotation',0,'Fontsize',12,'HorizontalAlignment','right')
            end
            
            if rw == M
                if all(size(X_raw) == [5,201,61])
                    xlabel('Excitation','Fontsize',10)
                    ylabel('Emission','Fontsize',10)
                    set(gca,'XTick',[1,31 ...,61
                        ],'XTickLabel',[250,275])%,300])
                    set(gca,'YTick',[1,101 ...,201
                        ],'YTicklabel',[250,350])%,450])
                    zt = get(gca,'ZTick');
                    set(gca,'ZTick',[zt(1),median(zt([1,end])),zt(end)])
                end

                v = get(gca,'view');

                xh = get(gca,'XLabel'); % Handle of the x label
                set(xh, 'Units', 'Normalized')
                pos = get(xh, 'Position');
%                 set(xh, 'Position',[0.55, -0.06,0],'Rotation',v(2)-20)
                set(xh, 'Position',[0.58, -0.3,0],'Rotation',v(2)-23)
                yh = get(gca,'YLabel'); % Handle of the y label
                set(yh, 'Units', 'Normalized')
%                 set(yh, 'Position',[0.45,-0.12, 0],'Rotation',v(1)+17)
                 set(yh, 'Position',[0.4,-0.35, 0],'Rotation',v(1)+23)
                1+1;
            else
                set(gca,'YTick',[], 'XTick', [])
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
    %     annotation(gcf,'textbox',[0.225, 0.55, 0.6, 0.03],'String',{'Reconstruction Error' })
end

end