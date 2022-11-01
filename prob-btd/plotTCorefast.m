function [E,V,L]=plotTCorefast(Core)

N=size(Core);
if length(N)==2
    N(3)=1;
end
Lt=[2 6; 2 4; 4 5; 5 6; 6 7; 7 1; 5 1; 1 3; 4 3; 3 8; 8 2; 8 7];

% Generate Bounding Box
C=N/2;
Box(1,:)=C+N/2;
Box(2,:)=C-N/2;
Box(3,:)=C+[N(1)/2 N(2)/2 -N(3)/2];
Box(4,:)=C+[N(1)/2 -N(2)/2 -N(3)/2];
Box(5,:)=C+[N(1)/2 -N(2)/2 N(3)/2];
Box(6,:)=C+[-N(1)/2 -N(2)/2 N(3)/2];
Box(7,:)=C+[-N(1)/2 N(2)/2 N(3)/2];
Box(8,:)=C+[-N(1)/2 N(2)/2 -N(3)/2];            
for k=1:size(Lt,1)
    hold on;
    if any(k == [1,4,5])
        plot3(Box(Lt(k,:),1)',Box(Lt(k,:),3)',Box(Lt(k,:),2)','--k','linewidth',1);
    else
        plot3(Box(Lt(k,:),1)',Box(Lt(k,:),3)',Box(Lt(k,:),2)','-k','linewidth',2);
    end
end

% Generate Tucker Core Cubes
Core=Core/max(abs(Core(:)));
q=0;
qq=0;
V=zeros(8*prod(N),3);
E=zeros(12*prod(N),3);
L=zeros(12*prod(N),2);
Col=[];
Et=[2 4 6; 4 5 6; 5 4 3; 3 5 1; 5 6 7; 5 7 1; 7 1 3; 7 8 3; 6 2 8; 7 6 8; 2 4 8; 4 3 8];
for i=1:N(1)
    i;
    for j=1:N(2)
        for k=1:N(3)
            C=[i-0.5 j-0.5 k-0.5];
            t=0.5*abs(Core(i,j,k));
            Corner(1,:)=C+t;
            Corner(2,:)=C-t;
            Corner(3,:)=C+[t t -t];
            Corner(4,:)=C+[t -t -t];
            Corner(5,:)=C+[t -t t];
            Corner(6,:)=C+[-t -t t];
            Corner(7,:)=C+[-t t t];
            Corner(8,:)=C+[-t t -t];            
            V=[V; Corner];            
            V(q+1:q+8,:)=Corner;
            E(qq+1:qq+12,:)=Et+q;
            L(qq+1:qq+12,:)=Lt+q;
            if Core(i,j,k)>=0
               Col(qq+1:qq+12)=ones(size(Et,1),1); 
            else
               Col(qq+1:qq+12)=zeros(size(Et,1),1);
            end
            q=q+8; 
            qq=qq+12;
        end
    end
end


% Plot Tucker Core Cubes
hold on;
for k=1:size(L,1)
    plot3(V(L(k,:),1)',V(L(k,:),3)',V(L(k,:),2)','-k','linewidth',2);
end

trisurf(E,V(:,1),V(:,3),V(:,2),Col,'EdgeColor',[0.5 0.5 0.5],'FaceAlpha',0.5,'EdgeAlpha',0.5);
colormap(gray);
%text(N(1)/2,-0.5,-0.5,'l','FontWeight','bold','FontSize',12)
%text(-0.5,N(2)/2,-0.1,'m','FontWeight','bold','FontSize',12)
%text(-0.5,N(2)+0.5,N(3)/2,'n','FontWeight','bold','FontSize',12)

set(gca,'CameraViewAngle',[7.38228]);
%set(gca,'CameraViewAngle',0);
%set(gca,'CameraPosition',[36.6166*5 11.0062*5 20.8639*5]);
%set(gca,'CameraPosition',[36*5,-10*5,-20*5]);
%set(gca,'CameraPosition', [75 50 45])
set(gca,'CameraPosition', [45 -45 40])
%set(gca,'CameraUpVector', [1,1,1])
% 37 -52 57
axis equal;
axis off;
colormap(gray);
caxis([0 1]);

% set(gca,'YDir','normal')
% set(gca,'XDir','reverse')
% set(gca,'ZDir','reverse')