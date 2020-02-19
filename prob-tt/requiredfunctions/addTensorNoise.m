function [X_noisy, actual_snr_dB, avg_noise, P_noise] = addTensorNoise(X,snr_dB)

%Calculate power of the noisefree signal for SNR 
%Xmu = bsxfun(@minus,X,mean(X,2));
%P_sig =  sum(Xmu(:).^2)/numel(X); %Power of signal
P_sig = sum(X(:).^2)/numel(X);

%Theoretical noise amplitude
P_noise = P_sig*10^(-snr_dB/10);

%Generate noise    
noise_unit = randn(size(X));
n = sum(noise_unit(:).^2)/numel(noise_unit);
noise=noise_unit/sqrt(n)*sqrt(P_noise); 
X_noisy = X+noise;

%%Actual noise amplitude
P_noise = sum(noise(:).^2)/(numel(noise));
actual_snr_dB = 10*log10(P_sig)-10*log10(P_noise);

avg_noise = mean(noise(:));
assert(~any(isnan(X_noisy(:))))
assert(~any(isnan(X(:))))
end