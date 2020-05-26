function [factors, priors, shares_prior ] = initializePriorDistributions(...
                                        N, D, missing, ... 
                                        constraints,...
                                        inference_scheme,lambda_a0,lambda_b0)
%INITIALIZEPRIORDISTRIBUTIONS initializes the factor and (hyper)prior
%distributions for a CP model according to the constraints and inference
%schemes. 
%   The inputs are:
%       'N'         A vector with the number of observations in each mode
%       'D'         Scalar, the number of components
%       'missing'   Are missing values present yes (true) or no (false)
%       'constraints'         
%       'inference_scheme'      variational or sampling         

if nargin <1
   input_factor_distr = {'normal','nonnegative','nonneg expo','infin'};
   input_prior_distr = {'ard', 'ard', 'ard indi', 'sparse'};
   N = randi([5,20],length(input_factor_distr));
   D=3;
   missing=true;
   inference_scheme = nan;
   constraints = strcat(input_factor_distr',{' '},input_prior_distr');
    
end

if length(D) < length(N)
    D = repmat(D,1,length(N));
end

%%
for i = 1:length(constraints)
    temp = strsplit(constraints{i});
    if any(my_contains(temp,'indi'))
        input_factor_distr{i} = strjoin(temp(1:end-2));
        input_prior_distr{i} = strjoin(temp(end-1:end));
    else
        input_factor_distr{i} = strjoin(temp(1:end-1));
        input_prior_distr{i} = temp{end};
    end
    
    if my_contains(input_prior_distr{i}, 'infinity','IgnoreCase',true)
        input_prior_distr{i} = '';
        input_factor_distr{i} = 'infinity';
    elseif my_contains(input_prior_distr{i}, 'orth','IgnoreCase',true)
        input_prior_distr{i} = '';
        input_factor_distr{i} = 'orth';
    end
    
end

%%
                                   
assert(length(input_factor_distr) == length(input_prior_distr),... 
    'The number of factor and prior distributions must be the same.')
Nx = length(input_factor_distr);


%% Validate Input 
if all(my_contains(input_factor_distr, 'orth','IgnoreCase',true))
    warning(['For Candecomp/PARAFAC (CP): Orthogonalizing all modes may lead to only zero factors. ',...
    'Try scaling the data "X/(var(X(:))^(1/ndims(X)))" or modeling a normal ',...
    'distribution on one of the modes.'])
%     pause(5)
end

valid_factor_distr = {'normal','nonneg',...
                      'infi', 'orth'...
%                       'orthogonal', 'unimodal',...
                      };
valid_factor_distr_long = {'normal','nonnegative','nonnegative exponential',...
                      'infinity'...
                      'orthogonal', ... 'unimodal',...
                      };
valid_prior_normal = {'const', 'scale', 'ard', 'wish'};
valid_prior_nonneg = {'const', 'scale', 'ard', 'sparse'};

% Remove special character and trim whitespace
input_factor_distr = strtrim(strrep(input_factor_distr, '-', ''));
input_prior_distr = strtrim(strrep(input_prior_distr, '-', ''));

% Check the specifed distribution is valid
valid_input = false(length(input_factor_distr), length(valid_factor_distr));
for i = 1:length(valid_factor_distr)
   valid_input(:,i) =  my_contains(input_factor_distr, valid_factor_distr{i});
end

% Specified distribution is ambiguous or invalid..
idx_invalid = [find(sum(valid_input,2)>1)', find(sum(valid_input,2)==0)'];
if ~isempty(idx_invalid)
    s_err = '';
    for i = idx_invalid
        s_err = sprintf('%s\tFactor distribution on mode %i was "%s"\n', s_err, i, input_factor_distr{i});
    end
    s_err = sprintf('%s\nTry one of the following valid distributions instead, \n\t%s',s_err, string(strjoin(valid_factor_distr_long,'\n\t')));
    error('ERROR: Specified distribution was ambiguous or invalid, see\n%s\n', s_err)
end

%% Now validate the prior distributions..
valid_input = false(length(input_factor_distr), 1);
for i = 1:length(input_factor_distr)
    
    if my_contains(input_factor_distr{i}, 'normal')
        valid_prior = valid_prior_normal;
    elseif my_contains(input_factor_distr{i}, 'nonneg')
        valid_prior = valid_prior_nonneg;
    elseif my_contains(input_factor_distr{i}, 'infi')
        valid_prior = [];
    elseif my_contains(input_factor_distr{i}, 'orth')
        valid_prior = [];
    else
        error('This should never occur.')
    end
    
    if ~isempty(valid_prior)
        t = false(length(valid_prior),1);
        for j = 1:length(valid_prior)
            t(j) = my_contains(input_prior_distr{i}, valid_prior{j});
        end
        valid_input(i) = sum(t) == 1;
    elseif my_contains(input_factor_distr{i},{'infi', 'orth'},'IgnoreCase', true)
        valid_input(i) = isempty(input_prior_distr{i});
    end
    
end

%If any distributions are invalid, throw error
idx_invalid = find(valid_input'==0);
if ~isempty(idx_invalid)
    s_err = '';
    for i = idx_invalid
        s_err = sprintf('%s\tDistribution on mode %i was "%s"\n', s_err, i, input_prior_distr{i});
    end
    s_err = sprintf('%s\nSee "help" or "doc" for details on prior specification.\n',s_err);
    error('ERROR: Specified prior distribution was ambiguous or invalid, see\n%s\n', s_err)
end

%% Initialize Prior Distribution
prior_initialized = false(Nx,1);
idx_individual_prior = my_contains(input_prior_distr,{'indi','const'},'IgnoreCase',true);
factors = cell(Nx,1);
priors = cell(Nx,1);

shares_prior = zeros(Nx,1);
for i = 1:Nx
    if ~prior_initialized(i) && ~my_contains(input_factor_distr{i}, {'infinity', 'orth'}, 'IgnoreCase', true)
        
        % Find prior distribution
        if my_contains(input_prior_distr{i}, 'const', 'IgnoreCase', true)
            prior_distr = 'constant';
        elseif my_contains(input_prior_distr{i}, 'scale', 'IgnoreCase', true)
            prior_distr = 'scale';
        elseif my_contains(input_prior_distr{i}, 'ard', 'IgnoreCase', true)
            prior_distr = 'ard';
        elseif my_contains(input_prior_distr{i}, 'sparse', 'IgnoreCase', true) ...
                && my_contains(input_factor_distr{i}, 'nonneg', 'IgnoreCase', true)
            prior_distr = 'sparse';
        elseif my_contains(input_prior_distr{i}, 'wishart', 'IgnoreCase', true) ...
                && my_contains(input_factor_distr{i}, 'normal', 'IgnoreCase', true)
            prior_distr = 'wishart';
        end
        
        %Find factor distribution
        if my_contains(input_factor_distr{i}, 'nonneg', 'IgnoreCase', true)
            if my_contains(input_factor_distr{i}, 'expo', 'IgnoreCase', true)
                factor_distr = 'exponential';
            else
                factor_distr = 'truncated normal';
            end
        else
            factor_distr = 'normal';
        end
        
        % Determine how many "observations" have this prior.
        idx_shared = find(my_contains(input_prior_distr, prior_distr) & ~idx_individual_prior);
        if isempty(idx_shared) ||length(idx_shared) == 1
            Ni = N(i);
            Neffective = [];
        else
            distr_const = 1./(1+~my_contains(input_factor_distr(idx_shared),'expo'));
            Ni = sum(N(idx_shared)); %#ok<FNDSB>
            Neffective = N(idx_shared)*distr_const';
            if strcmpi(prior_distr, 'sparse') 
               Ni = N(i); 
               assert(all(N(i) == N(idx_shared)), 'Shared sparsity, but the number of observations differ!')
            end
        end
        if length(idx_shared)>1
            shares_prior(idx_shared) = max(shares_prior)+1;
        end
        
        %% Actually initialize the prior-object.
        if any(strcmpi(prior_distr,{'scale', 'ard', 'sparse'}))
            %lambda_a0=[]%1e-3
            %lambda_b0=[]%1e-3
            priors{i} = GammaHyperParameter(factor_distr, prior_distr,...
                [Ni,D(i)],lambda_a0,lambda_b0,inference_scheme{i,2}, Neffective);
        elseif strcmpi(prior_distr,'constant')
            constr_value=1;
            priors{i} = HyperParameterConstant([N(i), D(i)], constr_value);
            
        elseif strcmpi(prior_distr,'wishart')
            priors{i} = WishartHyperParameter([Ni,D(i)],inference_scheme{i,2});
        else
            error('Unknown prior distribution (%s).', prior_distr)
        end
        
        % Link (handle) the shared priors to the same prior (object)
        if length(idx_shared)>1
            for j = idx_shared(2:end)
                priors{j} = handle(priors{i});
                prior_initialized(j) = true;
            end
        end
    else
        %Pass, this prior is shared and has already been initialized
    end
end

%% Initialize Factor Distribution
for i = 1:length(input_factor_distr)
    %Find factor distribution
    if my_contains(input_factor_distr{i}, 'nonneg', 'IgnoreCase', true)
        if my_contains(input_factor_distr{i}, 'expo', 'IgnoreCase', true)
            factor_distr = 'exponential';
        else
            factor_distr = 'truncated normal';
        end
        
    elseif my_contains(input_factor_distr{i},'infi', 'IgnoreCase', true)
        factor_distr = 'infinity';
        
	elseif my_contains(input_factor_distr{i},'orth', 'IgnoreCase', true)
        factor_distr = 'orthogonal';
        
    elseif my_contains(input_factor_distr{i},'normal', 'IgnoreCase', true)
        factor_distr = 'multivariate normal';
        
    else
        error('Unknown or unsupported factor distribution (%s)', factor_distr);
    end
    
    %% Actually initialize factor
    if strcmp(factor_distr, 'exponential')
        factors{i} = ExponentialFactorMatrix([N(i),D(i)], handle(priors{i}),...
            missing, inference_scheme{i,1});
        factors{i}.initialize(@rand);
        
    elseif strcmp(factor_distr, 'truncated normal')
        factors{i} = TruncatedNormalFactorMatrix([N(i),D(i)], handle(priors{i}),...
            missing, inference_scheme{i,1});
        factors{i}.initialize(@rand);
        
    elseif strcmp(factor_distr, 'multivariate normal')
        factors{i} = MultivariateNormalFactorMatrix(...
            [N(i),D(i)], handle(priors{i}),...
            missing, inference_scheme{i,1});
        factors{i}.initialize(@randn);
        
    elseif strcmp(factor_distr, 'infinity')
        factors{i} = UniformFactorMatrix([N(i), D(i)], 0, 1,...
            missing, inference_scheme{i,1});
        factors{i}.initialize(@(arg) rand(arg)*0.5+0.25);
        
        priors{i} = factors{i}.hyperparameter;
        
    elseif strcmp(factor_distr, 'orthogonal')
        
        assert(missing==false,'Missing values are not supported when using orthogonal factors.')
        assert(N(i)>= D(i), 'When modeling orthogonality in mode i, then the observations in that mode (N_i=%i) must be greater than or equal to the rank (D=%i).', N(i), D)
        
        factors{i} = OrthogonalFactorMatrix([N(i), D(i)], inference_scheme{i,1});
        factors{i}.initialize(@(arg) orth(randn(arg)));
        
        priors{i} = factors{i}.hyperparameter;
        
    end
    
    
end
assert(any(~cellfun(@isempty, factors)), 'Error! Some factors where not initialized')
assert(any(~cellfun(@isempty, priors)), 'Error! Some priors where not initialized')
1+1;    
end

