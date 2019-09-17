function [Rkrkr,IND] = premultiplication(R,kr,D)
h2ind = [1];
for i = 2:D
    h2ind(i) = h2ind(i-1)+ D-i+2;
end

IND = nan(D,D-1,'like',kr);
IND(1,:) = 2:D;
for i = 2:D
    if i == D
        IND(D,:) = IND(1:(D-1),D-1)';
    else
        IND(i,1:(i-1)) = IND(1:(i-1),i-1)';
        IND(i,i:end) = h2ind(i)+1:h2ind(i+1)-1;
    end
end

% Premultiplication
Rkrkr = zeros(size(R,1),D*(D+1)/2,'like',kr);
for i = 1:D
    if i == D
        Rkrkr(:,end) = R*bsxfun(@times,kr(:,D),kr(:,D));
    else
        if size(R,1) < size(kr,2)-i+1 % Which dimension is larges
            Rkrkr(:,h2ind(i):h2ind(i+1)-1) = bsxfun(@times,R,kr(:,i)')*kr(:,i:end);
        else
            Rkrkr(:,h2ind(i):h2ind(i+1)-1) = R*bsxfun(@times,kr(:,i:end),kr(:,i));
        end
    end
end


end