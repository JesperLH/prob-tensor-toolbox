function [X, A, P] = generateDataParafac2(Nobs,Mobs,Components, constraints_A)

assert(Nobs(end)==length(Mobs), 'Number of observations in the last mode, must match length of the variable mode.')
assert(min(Mobs(:))>=Components, 'The number of components cannot be higher than the smallest dimension in the varying mode.')

if nargin < 4
    constraints_A = repmat({'normal'},1,length(Nobs)+2);
end

A = cell(length(Nobs)+1,1);
P = cell(length(Mobs),1);

for n = 1:length(Nobs)-1
    
    A{n} = mvnrnd(zeros(Nobs(n),Components),rand(1,Components)*10+1);
    if contains(constraints_A{n},{'non','negative'},'IgnoreCase',true')
        A{n} = abs(A{n});
    elseif contains(constraints_A{n},{'orthogonal'},'IgnoreCase',true')
       A{n} = orth(A{n}); 
    end
    % TODO: more constraints
end
A{end-1} = rand(Components, Components)+0.1;
A{end} = rand(Nobs(end),Components)*2+0.1;

for m=1:length(Mobs)
    P{m} = orth(randn(Mobs(m),Components));
end


X = nmodel_pf2(A(1:end-2), A(end-1), A{end}, P);



end