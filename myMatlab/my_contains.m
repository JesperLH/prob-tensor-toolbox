function truefalse = my_contains(s, pattern, varargin)
%MY_CONTAINS tries to duplicate the functionality of Matlab2016b function
%"contains".
%   If this code is run on Matlab2016b or newer, the function defaults to
%   "contains". If an older version of Matlab is used, this function
%   provides backwards compatibility.
%
% The function compares a single string or cell or strings to a single or
% cell of string patterns. It returns a boolean vector, that is true when
% any pattern is exists in the string.

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
    
    % Attempt to duplicate functionality.
    if iscell(s)
        if ignoreCase
            for j = 1:length(s)
                s{j} = lower(s{j});
            end
        end
        
        truefalse = false(1,length(s));
        for j = 1:length(s)
            try
                truefalse(j) = ~isempty(strfind(s{j}, pattern));
            catch
                tmpbool = false;
                for p = 1:length(pattern)
                      tmpbool= tmpbool || ~isempty(strfind(s{j}, pattern(p)));
                end
                truefalse(j) = tmpbool;
            end
        end
    else
        if ignoreCase
            s=lower(s);
        end
        
        try
            truefalse = ~isempty(strfind(s, pattern));
        catch
            tmpbool = false;
            for p = 1:length(pattern)
                  tmpbool= tmpbool || ~isempty(strfind(s, pattern(p)));
            end
            truefalse=tmpbool;
        end
    end
end

