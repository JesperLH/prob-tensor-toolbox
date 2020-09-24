%% Calculate the element-wise difference between inputs
%
%function [abs_diff, rel_diff] = db_diff(A,B,str_name)
function db_diff(A,B,str_name)
    abs_diff = norm(A(:)-B(:));
    rel_diff = abs_diff/norm(A(:));
    fprintf('%12s:\tabs_diff: %4.6e\trel_diff: %4.6e\n',str_name,abs_diff,rel_diff);
    if rel_diff > 1e-16
%         fprintf('A large difference was observed, do you which to continue? (press F5)\n')
%         keyboard
    end
end
