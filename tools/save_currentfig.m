function save_currentfig(location, filename)

    if nargin < 2
        filename='';
    end

    %savefig(strcat(location,filename,'.fig'))
    print(fullfile(location,filename),'-dpng')
    %print(fullfile(location,filename),'-depsc')

end