function [X_noisy, actual_snr_dB, avg_noise, P_noise, noise_het] = ...
    addTensorNoise(X,snr_dB, noise_modes, noise_type)

if nargin < 4 || isempty(noise_type)
    noise_type = 'rand';
end

if nargin < 3
    noise_modes = [];
    noise_het = [];
    noise_type = [];
end

%Calculate power of the noisefree signal for SNR 
%Xmu = bsxfun(@minus,X,mean(X,2));
%P_sig =  sum(Xmu(:).^2)/numel(X); %Power of signal
P_sig = sum(X(:).^2)/numel(X);

%Theoretical noise amplitude
P_noise = P_sig*10^(-snr_dB/10);

%Generate noise    
noise_unit = randn(size(X)); % Homoscedastic noise
if ~isempty(noise_modes)
    % Generate mode specific heteroscedastic noise
    noise_het = cell(ndims(X),1);
    for i = 1:ndims(X)
        if noise_modes(i)
            if strcmpi(noise_type,'rand')
                noise_het{i} = (rand(size(X,i),1)*0.1+0.05);
            elseif strcmpi(noise_type,'exp')
                noise_het{i} = exprnd(5,size(X,i),1);
            end
        else
            noise_het{i} = ones(size(X,i),1);
        end
    end
    noise_unit = noise_unit.*sqrt(1./nmodel(noise_het));
end

% Scale noise to achieve desired SNR
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