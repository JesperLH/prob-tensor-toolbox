function truefalse = my_contains(s, pattern, varargin)
%MY_CONTAINS tries to duplicate the functionality of Matlab2016b function
%"contains".
%   If this code is run on Matlab2016b or newer, the function defaults to
%   "contains". If an older version of Matlab is used, this function
%   provides backwards compatibility.

ver = version('-release');
% Call Matlab function contains, if it exists
if str2num(ver(1:end-1)) > 2016 || ...
    (str2num(ver(1:end-1)) == 2016 && strcmp(ver(end),'b'))
    
    truefalse = contains(s, pattern, varargin{:});
else
    
    ignoreCase = ~isempty(varargin) && strcmp(varargin{1},'IgnoreCase') && varargin{2};
    if ignoreCase
        pattern = lower(pattern);
    end
    
    % Attempt to duplicate functionality
    if iscell(s)
        if ignoreCase
            for j = 1:length(s)
                s{j} = lower(s{j});
            end
        end
        
        truefalse = false(length(s),1);
        for j = 1:length(s)
            truefalse(j) = ~isempty(strfind(s{j}, pattern));
        end
    else
        if ignoreCase
            truefalse = ~isempty(strfind(lower(s), pattern));
        else
            truefalse = ~isempty(strfind(s, pattern));
        end
    end
end

