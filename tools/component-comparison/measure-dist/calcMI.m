function MI=calcMI(Z1,Z2)

% Function to calculate mutual information between two assignment matrices
% Z1 and Z2
%
% Usage:
%  MI=calcMI(Z1,Z2)
%
% Input:
%   Z1  D1 x N Assignment matrix
%   Z2  D2 x N Assignment matrix
%   
% Output:
%   MI  Mutual information between Z1 and Z2.
%
% Written by Morten Mørup

P=Z1*Z2';
PXY=P/sum(sum(P));
PXPY=sum(PXY,2)*sum(PXY,1);
ind=find(PXY>0);
MI=sum(PXY(ind).*log(PXY(ind)./PXPY(ind)));