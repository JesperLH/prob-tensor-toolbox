function save_currentfig(location, filename)

    if nargin < 2
        filename='';
    end

    %savefig(strcat(location,filename,'.fig'))
    print(strcat(location,filename),'-dpng')
    print(strcat(location,filename),'-depsc')

end