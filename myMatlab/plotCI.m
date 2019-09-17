function plotCI(x,mu,ci,plot_color, fill_color)

if size(x,2) == 1
    x = x';
end

if nargin < 4
    plot_color = 'k';
    fill_color = 'r';
end

x_plot =[x, fliplr(x)];
y_plot=[ci(:,1)', flipud(ci(:,2))'];

hold on
plot(x, mu, 'color', plot_color, 'linewidth', 1)
fill(x_plot, y_plot, 1,'facecolor', fill_color, 'edgecolor', 'none', 'facealpha', 0.4,'HandleVisibility','off');
hold off

end