%% Calculate matched correlations between components
% Note V is features, D is the number of components and B is the number of
% subjects.
% Inputs:
%   'A_true'    :  A matrix of true components of size V x D (posibly, x B)
%   'A_est'     :  A matrix of estimated components of size V x D (posibly, x B)
%
% Outputs:
%   'meanCorr'  :  Average correlation of matched components
%   'mci'       :  Correlation between true components and matched
%                  components. A matrix of size B x D .
%   'domfrac'   :  Fraction of subjects for which the dominating sequence
%                  is identical (max repetion/subjects=dominating fraction)
%   'domseq'    :  Indexes for the matched components (highest corr. first) 
%   'fracunmatch': Fraction of subjects for which the globally matched
%                  components is not equal to the locally matched component.
function [meanCorr, mci, domfrac, domseq, fracunmatch]=calcMatchedCorrelation(A_true,A_est)
%%
% If one of the arrays has less components, then padd with zeros
noc_true = size(A_true,2);
noc_est = size(A_est,2);
if noc_true < noc_est
    padsize = zeros(ndims(A_true),1); padsize(2) = noc_est-noc_true;
    A_true = padarray(A_true,padsize,'post');
elseif noc_true > noc_est
    padsize = zeros(ndims(A_est),1); padsize(2) = noc_true-noc_est;
    A_est = padarray(A_est,padsize,'post');
end


[~,noc,B] = size(A_est);
mci=zeros(B,noc); %zeros(M,L)
matching=zeros(B,noc); %zeros(M,L)

for i=1:B % for each subject
    if size(A_true,3) == B
        C=corr(A_true(:,:,i),A_est(:,:,i));
    else
        C=corr(A_true,A_est(:,:,i));
    end
    %r=zeros(size(Si.ic,1),1);
    oc=1:noc;
    ec=1:noc;
    for j=1:noc % for each component
        [rj ridx]=max(abs(C)); % find max value in each col and their row index
        [rj cidx]=max(rj); % find max col index and highest correlation
        ridx=ridx(cidx);  % find max row index
        mci(i,oc(ridx))=rj; % mci = correlation of each component for each subject
        C(ridx,:)=[]; % remove row
        C(:,cidx)=[]; % remove col
        matching(i,oc(ridx))=ec(cidx); % components for each subject
        oc(ridx)=[];
        ec(cidx)=[];
    end
end

um=unique(matching,'rows'); % returns all unique rows
for i=1:size(um,1)
    % Estimate how many times each unique row is present
    nm(i)=sum(sum(abs(bsxfun(@minus,matching,um(i,:))),2)==0); %sum(sum(abs(matching - unique row),col) == 0)
end
[moc, midx]=max(nm); % find first max repetiton. moc = value, midx = index.
domfrac=moc/B; % max repetion/subjects = dominating fraction
domseq=um(midx,:); %  the dominating sequence
%disp(sprintf('mean correlation %.3f, dominating sequence %sat %.1f%%',...
%    mean(mci(:)),sprintf('%i ',um(midx,:)),domfrac*100.));
% M - sum((matching - domseq) ==0) (how many times is each component the
% same as the component in domseq). 30 comoponents total - how many are
% matched to the domseq. so fraction of unmatched components.
fracunmatch=(B-sum(bsxfun(@minus,matching,um(midx,:))==0))/B;
%disp(sprintf('occurance of non-matching over components:%s',...
%    sprintf(' %.1f%%',fracunmatch*100)));
meanCorr = nanmean(mci(:)); % mean correlation