function result = run_precommit_tests(s_program_id)
%% Pre-commit tests
% Only checks that the functions can be called, not if they find the
% correct answer or are numerically stable.

%setup_paths;
addpath('./')
addpath(genpath('../tools'))
addpath('./myMatlab')
addpath(genpath('./myclasses/'))
addpath('../thirdparty-matlab/nway331/')
addpath('../thirdparty-matlab/trandn/')

if nargin < 1
    save_loc = './unit-testing/summary_of_tests.md';
else
    disp(s_program_id)
    save_loc = sprintf('./unit-testing/summary_of_tests_R%s.md',version('-release'));
end
fID = fopen(save_loc,'w');
fprintf(fID, '# Unit test status\n');

% Creates parallel pool
% parfor i = 1:10
%    i*2; 
% end


t0 = tic;
tCpu = cputime;
% If no coverage report is desired, then parallel is faster
run_parallel = true;
% else
% Run unit tests: Non-negative CP, t-norm, expo and infinity (e.g. NCP)
result_ncp = runtests('unit-testing/test_functionality_NCP_and_infinity.m','UseParallel',run_parallel);
% Run unit tests: Multivariate Normal CP (eg. CP)
result_cp = runtests('unit-testing/test_functionality_CP.m','UseParallel',run_parallel);
% Run unit tests: Integration of NCP and CP factors
result_inte = runtests('unit-testing/test_functionality_CP_and_NCP_integration.m','UseParallel',run_parallel);
% Run unit tests: Integration of Orthogonal factors and NCP and CP factors
% (No missing, and only homoscedastic noise)
result_orth = runtests('unit-testing/test_functionality_Orthogonal_integration.m','UseParallel',run_parallel);

result = [result_ncp(:); result_cp(:); result_inte(:); result_orth(:)];
fprintf(fID, 'Real time was %6.4f sec.\n', toc(t0) );
fprintf(fID, 'CPU time was %6.4f sec.\n\n', cputime-tCpu);

%%
idx_passed = [result.Passed]; 
if sum(idx_passed) == length(idx_passed)
    fprintf(fID, 'Everything is able to run (all %i cases)\n\n', sum(idx_passed));
else
    fprintf(fID, ['Some testcases did not complete (%i of %i cases), these should ', ...
             'be investigated before committing.\n\n'],length(idx_passed)-sum(idx_passed), length(idx_passed));
end

failed_tests = {result.Name}; failed_tests = failed_tests([result.Failed]);
for i = 1:length(failed_tests)
    fprintf(fID, [failed_tests{i},'\n']);
end

fprintf(fID, datestr(now,'mmmm dd, yyyy HH:MM:SS'));
fclose(fID);
fprintf('\n')
%% Show file output in console
if strcmpi(computer('arch'), 'win64')
    command = sprintf('type %s',save_loc);
    status = dos(command, '-echo');
elseif strcmpi(computer('arch'), 'glnxa64')
    command = sprintf('cat %s',save_loc);
    status = unix(command, '-echo');
end
%%!cat unit-testing/summary_of_tests.md
% keyboard
%%
return

%% Display details of failed cases
failed_results = result(~idx_passed);

for i = 1:length(failed_results)
    fprintf(failed_results(i).Name);
    fprintf('\n')
    %fprintf(failed_results(i).Details.DiagnosticRecord);
    disp(failed_results(i).Details.DiagnosticRecord.Exception)
    fprintf('\n')
    for j = 1:length(failed_results(i).Details.DiagnosticRecord.Stack)
        disp(failed_results(i).Details.DiagnosticRecord.Stack(j))
    end
    %fprintf(
    fprintf('--------------------------------------------------------\n\n')
end
end
    