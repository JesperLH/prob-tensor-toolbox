function [X,FACT,G]=generateTuckerData(Ns,Ds,fact_constr, core_constr)
%Ns = [6, 5, 4, 3];
%D = 2;
%constr = [0,1,2,3,5];
%   constr 5: sparse with 30% sparse
%   constr 3: unbounded in the upper limit (NO ARD)
%   constr 2: bounded with infinity norm 1
%   constr 1: sparse and unbounded in the upper limit (element-wise ARD)
%   constr 0: unbounded in the upper limit + column ARD
%X = zeros(Ns);
FACT = cell(length(Ns),1);


i = 1;
maxrepeats = 10;
repeats = 0;
while i <= length(Ns)
    
    if fact_constr(i) == 0
        %
        FACT{i} = randn(Ns(i),Ds(i)).^2;
    elseif fact_constr(i) == 1
        FACT{i} = abs((randn(Ns(i),Ds(i)).^3));
    elseif fact_constr(i) == 2
        FACT{i} = randn(Ns(i),Ds(i));
        FACT{i} = FACT{i}/max(abs(FACT{i}(:)));
    elseif fact_constr(i) == 3 ||  fact_constr(i) == 4
        %FACT{i} = bsxfun(@times,rand(Ns(i),D).^3,sqrt(sharedARDvar));
        sharedARDvar = (1:Ds(i))*2+1;
        FACT{i} = abs(mvnrnd(zeros(Ds(i),1),sharedARDvar,Ns(i)));
    elseif fact_constr(i) == 5
        
        FACT{i} = randn(Ns(i),Ds(i)).^2;
        sparse_pattern = ones(Ns(i),Ds(i));
        for d = 1:Ds(i)
            idx = randperm(Ns(i));
            sparsity_level = 0.8;
            numel_sparse = floor(Ns(i)*sparsity_level);
            sparse_pattern(idx(1:numel_sparse),d) = 0;
        end
        FACT{i} = FACT{i}.*sparse_pattern;
    elseif fact_constr(i) == 6
        FACT{i} = mvnrnd(zeros(Ds(i),1),rand(1,Ds(i))*0.5+0.5,Ns(i));
    elseif fact_constr(i) == 7
        FACT{i} = orth(mvnrnd(zeros(Ds(i),1),rand(1,Ds(i))*0.5+0.5,Ns(i)));
    else
        
        error('Constraint unspecified');
    end
    
    
    % Checks if the random factor has rank D, otherwise a new is generated.
    if rank(FACT{i}) == Ds(i)
        i = i+1;
        repeats = 0;
    else
        repeats = repeats+1;
        if repeats > maxrepeats
            error('Could not generate the desired random factor')
        end
    end
   
    
end


%% Generate core
G = randn(Ds).^3;
X = nmodel(FACT, G);

G = G/sqrt(var(X(:))) + randn(size(G));
X = nmodel(FACT, G);


end
