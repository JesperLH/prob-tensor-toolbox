function samples = sample_matrix_vonMises_Fisher(M,num_sweeps)
% Based on
% Hoff (2009): Simulation of the Matrix Bingham-von Mises-Fisher Distribution, With Applications to Multivariate and Relational Data
% https://rdrr.io/cran/rstiefel/

if nargin < 2
    num_sweeps = 100;
end
% M_old = randn(size(M));
% M = rmf_matrix(M_old);
[UU,~,VV] = svd(M,'econ');
rscol = max(2,min(round(log(size(M,1))),size(M,2)));

samples = zeros([size(M),num_sweeps]);
samples(:,:,1) = UU*VV'; % MAP estimate;

for i_samp = 1:num_sweeps-1
    [U,S,V] = svd(M,'econ');
    H = U * S;
    Y = samples(:,:,i_samp) * V;
    [m,R] = size(H);
    
    for iter = 1:round(R/rscol)
        r =  randperm(R,rscol); % Randomly select columns
        r_not = 1:R; r_not(r) = [];
        
        N = null(Y(:,r_not)');
        
        y = rmf_matrix(N' * H(:,r));
        Y(:,r) = N*y;
    end
    samples(:,:,i_samp+1) = Y * V';
    
end

end
function O = rmf_matrix(M)
% Rejection sampler..
if size(M,2) == 1
    O = rmf_vector(M);
else
    [U,S,V] = svd(M);
    H = U*S;
    [m,R] = size(H);
    cmet = false;
    rej = 0;
    while ~cmet
        U = zeros(m,R);
        U(:,1) = rmf_vector(H(:,1));
        lr = 0; % likelihood ratio?
        for j = 2:R
            N = null(U(:,1:j-1)');
            x=rmf_vector(N'*H(:,j));
            U(:,j) = N * x;
            if all(j<=size(S)) && S(j,j) > 0
                xn = sqrt(sum((N' * H(:,j)).^2)); 
                xd = sqrt(sum(H(:,j).^2));
                lbr = log(besseli(xn, 0.5 * (m-j-1)))...
                    - log(besseli(xd, 0.5 * (m-j-1)));
                if isnan(lbr)
                    lbr = 0.5 * (log(xd)-log(xn));
                end
                lr = lr + lbr + (xn-xd) + 0.5*(m-j-1)*(log(xd)-log(xn));
            end
        end
        cmet = log(rand(1)) < lr;
        rej = rej + (1-1*cmet);
        
    end
    O = U * V';
end
end

function u = rmf_vector(kmu)

kap = sqrt(sum(kmu.^2));
mu = kmu/kap;
m = length(mu);
if kap == 0
    u = randn(length(kmu),1);
    u = u/sqrt(sum(u.^2));
elseif kap > 0
    if m == 1
        u = (-1).^binornd(1,1./(1+exp(2*kap*mu)));
    elseif m>1
        W = rW(kap,m, 1);
        V = randn(m-1,1);
        V = V/sqrt(sum(V.^2));
        x = [sqrt((1-W^2)) * V; W];
        u = [null(mu'), mu] * x;
    end
end

end

function w = rW(kap,m, w)
% described in Wood(1994): Simulation of the von Mises Fisher distribution
b = (-2*kap + sqrt( 4 * kap^2 + (m-1)^2))/(m-1);
x0 = (1-b)/(1+b);
c = kap*x0 + (m-1) * log(1-x0^2);
%Z=0; U=0;
done = false;
while ~done
    Z = betarnd( (m-1)/2, (m-1)/2);
    w = (1-(1+b)*Z)/(1-(1-b)*Z);
    U = rand(1);
    if (kap*w + (m-1)*log(1-x0*w) -c) > log(U)
        done = true;
    end
end


end

% function O = NullC(X)
% [Q,R] = qr(X); % 0 => Econ
% x_rank = rank(Q);
% if x_rank == 0
%     set = 1:size(X,2);
% else
%     set = x_rank+1:size(X,2);
% %     set = x_rank:size(X,2);
% end
% O = Q(:,set)*R(set,:);
% end