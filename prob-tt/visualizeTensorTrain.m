function visualizeTensorTrain(U, labels_mode)
% VISUALIZETENSORTRAIN provides a basic visualization for Tensor Train
% models. 
% INPUT:
% U:    Cell array with each tensor train cart
% labels_mode:  Labels for each data mode

M = length(U);
D = cellfun(@(x) size(x,1),U);
Dmax = max(D);
U{1} = squeeze(U{1})';

for m = 1:M

    subplot(Dmax,M,m)
    plot(squeeze(U{m}(:,:,1)'))
    axis tight
    title(sprintf('Mode %i',m))
    
    if ~(m==1 || m==M)
        
        for d = 2:D(m+1)
           set(gca,'XTick',[])
           subplot(Dmax,M,m+(d-1)*M)
           plot(U{m}(:,:,d)')
           axis tight

        end
    end
    xlabel(labels_mode{m})
    
end


end