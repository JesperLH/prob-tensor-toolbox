%% RV coefficient
% Emperical comparison between the RV coefficent and the avg. correlation
% of matched components. 
%Data
A = randn(10000,10);
B = randn(10000,10);
% U is a rotational matrix
BtB=B'*B; [U,L] = eig(BtB);
% Rotation of A
a=coeffRV(A,A);
b=coeffRV(A,A*U);
% The diagonal of the correlation, gives the column by column correlation
c=diag(corr(A,A));
d=diag(corr(A,A*U));

fprintf('\nWhile the two solutions are from the same subspace, their interpretation can be very different.\n')
fprintf('(A,A) : \t RV=%6.4f \t avg.corr=%6.4f\n',a,mean(c(:)))
fprintf('(A,A*U) : \t RV=%6.4f \t avg.corr=%6.4f\n',b,mean(d(:)))

%
fprintf('\nAvg. correlation is higher if the components are matched.\n')
fprintf('(A,A) : \t RV=%6.4f \t avg.corr=%6.4f \t (matched) avg.corr=%6.4f\n',...
    a,mean(c(:)),calcMatchedCorrelation(A,A))
fprintf('(A,A*U) : \t RV=%6.4f \t avg.corr=%6.4f \t (matched) avg.corr=%6.4f\n',...
    b,mean(d(:)),calcMatchedCorrelation(A,(A*U)))