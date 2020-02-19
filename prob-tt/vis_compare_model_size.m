
f_cp = @(D,N,I) N.*D.*I;
f_tucker = @(D,N,I) D.^N + N.*D.*I;
f_btd = @(D,N,I,B) B.*(D./B).^N + B.*N.*(D./B).*I;
f_tt = @(D,N,I) D.*I.*2 + (N-2).*D.*I.*D;



figure, 
l_obs = [10,100,1000];
l_modes = 3:6
n_blocks = 2;
rangeD = 1:50;
y_max = f_tucker(rangeD(end),l_modes(end),l_obs(end));
y_max = 10^ceil(log10(y_max));
for i_obs = 1:length(l_obs)
    for i_modes = 1:length(l_modes)

    n_obs = l_obs(i_obs);
    n_modes = l_modes(i_modes);
    
    
    subplot(length(l_obs),length(l_modes), (i_obs-1)*length(l_modes) + i_modes)
    hold on
    lw=1.5;
    plot(rangeD,f_cp(rangeD,n_modes,n_obs),':k','LineWidth',lw)
	plot(rangeD,f_tucker(rangeD,n_modes,n_obs),'-k','LineWidth',lw)
	plot(rangeD,f_btd(rangeD,n_modes,n_obs,n_blocks),'--k','LineWidth',lw)
    plot(rangeD,f_tt(rangeD,n_modes,n_obs),'-.k','LineWidth',lw)
    set(gca,'YScale','log','YTick',10.^(0:2:ceil(log10(y_max))))
    ylim([0,y_max])
    set(gca,'XTick',[min(rangeD),floor(median(rangeD)),max(rangeD)])
    xlim([min(rangeD),max(rangeD)])
%     semilogy(rangeD,f_cp(rangeD,n_modes,n_obs),...
%                  rangeD,f_tucker(rangeD,n_modes,n_obs),...
%                  rangeD,f_btd(rangeD,n_modes,n_obs,n_blocks),...
%                  rangeD,f_tt(rangeD,n_modes,n_obs))
    if i_obs == 1
        title(sprintf('%i modes',l_modes(i_modes)))
    end

    if i_obs == length(l_obs)
        xlabel('Components (D)')
    else
        set(gca,'XTick',[])
    end
    
    if i_modes == 1
        ylabel(sprintf('#Parameters\n (I=%i)',l_obs(i_obs)))
    else
        set(gca,'YTick',[])
    end
    
%     set(gca,'Position',get(gca,'Position')-[0.05,0,0.01,0])
    
    
    end
end
legend({'CP','Tucker','BTD','TT'},'FontSize',12)