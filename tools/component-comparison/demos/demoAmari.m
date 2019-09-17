%% Demostration of Amari distance measure
% Shows how to find the Amari distance between two matrices $X\in \mathbb{R}^{d \times N}$ and $Y\in \mathbb{R}^{d \times N}$ 

N = 10; % Number of observations
d = 3; % Dimensionality
tries = 1000;
bins = 20;

%% Generate data and calculate Amari distance
distXY = zeros(tries,1); 
distYX = zeros(tries,1);
for i = 1:length(distXY)
    
    X = rand(d,N);
    Y = rand(d,N);
    distXY(i) = amariDist(X,Y,d); 
    distYX(i) = amariDist(Y,X,d); 
end;

%% Illustrate the distances measured
xmax = max(max(distXY(:)),max(distYX(:)));
xmin = min(min(distXY(:)),min(distYX(:)));
figure; 
subplot 311; histogram(distXY,bins,'BinLimits',[xmin,xmax]); title('Amari(X,Y)');
ylabel('Occurence');%xlabel('Distance'); 
subplot 312; histogram(distYX,bins,'BinLimits',[xmin,xmax]);  title('Amari(Y,X)');
ylabel('Occurence');%xlabel('Distance'); 
subplot 313; histogram((distXY+distYX)/2,bins,'BinLimits',[xmin,xmax]); title('( Amari(X,Y)+Amari(Y,X) )/2');
xlabel('Distance'); ylabel('Occurence')

suptitle('Amari distance measure is asymmetric')
set(gcf,'defaultaxesfontsize',18)