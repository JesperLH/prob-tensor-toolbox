function visualizeDecomposition(A,G,labels_mode)
% VISUALIZEDECOMPOSITION provides a basic visualization tool for CP and
% Tucker models.
% INPUT:
% A:    Cell with factor matrices
% G:    Core array (empty => CP model)
% labels_mode:  A cell array containing strings with labels on each mode

%%
M = length(A);
if isempty(G)
    D=size(A{1},2);
    nrow = 1;
else
    D = size(G);
    nrow = 2;
    min_vert=inf;
end

for m = 1:M
    
    for j=1:nrow
        subplot(nrow,M,m + (j-1)*M)
        if j==1 && nrow==2 % Visualize unfolded core
            imagesc(matricizing(G,m));
            colormap parula;
%             if m==M
%                 colorbar('Location','southoutside')
%             end
            axis equal; axis tight
            pos = get(gca,'Position');
            min_vert = min(pos(2),min_vert);
            ylabel(sprintf('D_{%i}',m))%,'Rotation',0);
            set(gca,'XTick',[])
        else % Visualize m-mode factor matrix.
            plot(squeeze(A{m}))
            axis tight
            if m==1
                ylabel('Loading')
            end
        end
        
        if j==1
            title(sprintf('Mode %i',m))
            if  nrow==2 && m==M
             cb=colorbar('Location','southoutside',...
                 'Position',[0.15,min_vert*0.98,0.70,0.025]);
            end
        end
    end
    
    if ~isempty(labels_mode)
        xlabel(labels_mode{m})
    end
    
end

end
