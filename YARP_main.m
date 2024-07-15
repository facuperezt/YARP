% Main script
clc;
clear;
close all;


% Parameters
W = 10 * 1e6; % bandwidth in hertz
M = 16; % number antennas
N = 8; % number sequences
nMC = 100; % number of monte carlo iterations
m = 0:(M - 1); % 
d = 1; % distance between antennas
c = 3e8; % speed of light
f0 = 5e9; % carrier frequency
wavelength = c / f0; % wavelength
L = 63; % length of zadoff-chu
q = 29; % factor of zadoff-chu
var = 1; % variance of awgn

% Initialize Zadoff-Chu sequence
zc = zadoff_chu_sequence(L, q);

%sweep parameters
len_hat = 10;
SNR = -20:5:25; % snr in db
beta_hat = linspace(-pi/3,pi/3,len_hat);
tau_hat = linspace(0,L/3,len_hat); 
eta_hat = linspace(-0.03,0.03,len_hat) *10e3;

%init containers
objectiveFunction = zeros(len_hat,len_hat,len_hat);
Omega_mc_snr = zeros(3,nMC,length(SNR));
Omega_snr = zeros(3,length(SNR));

for iSNR = 1:length(SNR)
    snr = SNR(iSNR);
    A = sqrt(10^(snr / 10)); %signal amplitude

    for iMC = 1:nMC

        %% draw true paramters for each monte carlo simulation 

        phase = 2 * pi * rand; % Phase
        rho = exp(1i * phase);

        beta = -pi / 3 + (pi / 3 + pi / 3) * rand; % Random from [-pi/3, pi/3] %check
        tau = randi(L/3); % Delay
        eta = (randi(60) - 30) *1e-3; % Doppler shift

        alpha = exp(1i * 2 * pi * m * d / wavelength * sin(beta)); % vector alpha for incident angle estimation

        F =  1 / sqrt(L) * exp(-1i * (2 * pi) / L * ((0:(L-1)).' * (0:(L-1)))); % fourier transform matrix for delay

        D_z = diag(F .* zc).'; 
        b_k = exp(-1i * (2 * pi) / L * (0:(L - 1)) * tau);
        c_n = exp(1i * (2 * pi) * eta * (0:(N - 1)));

        Dz_b = D_z .* b_k; 
        Dz_b_c = (Dz_b.' * c_n).';

        [rows, cols] = size(Dz_b_c);
        slices = length(alpha);

        X = zeros(rows, cols, slices);
        
        for k = 1:slices
            X(:,:,k) = alpha(k) * Dz_b_c;
        end

        awgn = normrnd(0,var,size(X)) + 1i * normrnd(0,var,size(X)); %complex gaussian noise

        Y = rho*A*X + awgn; %backscattered signal

        y_vec = reshape(Y,[M*L*N,1,1]); 


        %% sweep over vector of estimated values

        for i = 1:len_hat

            beta = beta_hat(i);

            for j = 1:len_hat

               tau = tau_hat(j);

               for k = 1:len_hat

                    eta = eta_hat(k);

                    alpha = exp(1i * 2 * pi * m * d / wavelength * sin(beta));

                    kl = (0:(L-1)).' * (0:(L-1));

                    F =  1 / sqrt(L) * exp(-1i * (2 * pi) / L * kl);
            
                    D_z = diag(F .* zc).';
                    b_k = exp(-1i * (2 * pi) / L * (0:(L - 1)) * tau);
                    c_n = exp(1i * (2 * pi) * eta * (0:(N - 1)));
            
                    Dz_b = D_z .* b_k;
                    Dz_b_c = (Dz_b.' * c_n).';
            
                    [rows, cols] = size(Dz_b_c);
                    slices = length(alpha);
            
                    h = zeros(rows, cols, slices);
                    
                    for s = 1:slices
                        h(:,:,s) = alpha(s) * Dz_b_c;
                    end

                    h_vec = reshape(h,[M*L*N,1,1]);

       
                    objectiveFunction(i,j,k) = abs(h_vec' * y_vec).^2;
        
                end
            end
        end
    
    [U, I] = max(objectiveFunction(:));

    [ibeta_hat, itau_hat, ieta_hat] = ind2sub(size(objectiveFunction), I);

    if ~(objectiveFunction(ibeta_hat, itau_hat, ieta_hat) == U)
        error('max search failed');
    end

    Omega_mc_snr(1,iMC,iSNR) = abs(beta_hat(ibeta_hat) - beta)^2;
    Omega_mc_snr(2,iMC,iSNR) = abs(tau_hat(itau_hat) - tau)^2;
    Omega_mc_snr(3,iMC,iSNR) = abs(eta_hat(ieta_hat) - eta)^2;

    end
    
    Omega_snr(1,iSNR) = sum(Omega_mc_snr(1,:,iSNR))/nMC;
    Omega_snr(2,iSNR) = sum(Omega_mc_snr(2,:,iSNR))/nMC;
    Omega_snr(3,iSNR) = sum(Omega_mc_snr(3,:,iSNR))/nMC;
end

figure
hold on
plot(SNR,Omega_snr(1,:))
plot(SNR,Omega_snr(2,:))
plot(SNR,Omega_snr(3,:))
hold off

title('MSE of parameters over SNR')
xlabel('OSNR')
ylabel('MSE')
legend('\beta','\tau','\eta')


function zl = zadoff_chu_sequence(L, q)
    % Generate Zadoff-Chu sequence
    l = 0:(L - 1);
    zl = exp(-1i * pi * q * (l .* (l + 1) / L));
end
