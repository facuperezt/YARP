import numpy as np
import matplotlib.pyplot as plt

# Zadoff-Chu sequence class
class zadoff_chu:
    def __init__(self, L, q):
        """
        L = 63
        q = 1 --> needs to be looked up from literature
        """
        self.L = L
        self.q = q  # either 29, 34 and 25

    def sequence(self):
        l = np.arange(0, self.L, 1)
        zl = list()
        for li in l:
            zl.append(np.exp(-1j * np.pi * self.q * (li * (li + 1) / self.L)))
        return zl


if __name__ == '__main__':
    # Parameters
    W = 10 * 1e6  # bandwidth in hertz
    M = 16  # number antennas
    N = 8  # number sequences
    nMC = 100  # number of monte carlo iterations
    m = np.arange(M)
    d = 1  # distance between antennas
    c = 3e8  # speed of light
    f0 = 5e9  # carrier frequency
    wavelength = c / f0  # wavelength
    L = 63  # length of zadoff-chu
    q = 29  # factor of zadoff-chu
    var = 1  # variance of awgn

    # Initialize Zadoff-Chu sequence
    zadoff_chu_seq = zadoff_chu(L, q)
    zc = zadoff_chu_seq.sequence()

    # Sweep parameters
    len_hat = 10
    SNR = np.arange(-20, 30, 5)  # snr in dB
    beta_hat = np.linspace(-np.pi / 3, np.pi / 3, len_hat)
    tau_hat = np.linspace(0, L / 3, len_hat)
    eta_hat = np.linspace(-0.03, 0.03, len_hat) * 10e3

    # Init containers
    objectiveFunction = np.zeros((len_hat, len_hat, len_hat))
    Omega_mc_snr = np.zeros((3, nMC, len(SNR)))
    Omega_snr = np.zeros((3, len(SNR)))

    for iSNR in range(len(SNR)):
        snr = SNR[iSNR]
        A = np.sqrt(10**(snr / 10))  # signal amplitude

        for iMC in range(nMC):

            # Draw true parameters for each monte carlo simulation
            phase = 2 * np.pi * np.random.rand()  # Phase
            rho = np.exp(1j * phase)

            beta = -np.pi / 3 + (np.pi / 3 + np.pi / 3) * np.random.rand()  # Random from [-pi/3, pi/3]
            tau = np.random.randint(L // 3)  # Delay
            eta = (np.random.randint(60) - 30) * 1e-3  # Doppler shift

            alpha = np.exp(1j * 2 * np.pi * m * d / wavelength * np.sin(beta))  # vector alpha for incident angle estimation

            kl = np.outer(np.arange(L), np.arange(L))
            F = 1 / np.sqrt(L) * np.exp(-1j * (2 * np.pi) / L * kl)  # Fourier transform matrix for delay

            D_z = np.diag(F @ zc)
            b_k = np.exp(-1j * (2 * np.pi) / L * np.arange(L) * tau)
            c_n = np.exp(1j * (2 * np.pi) * eta * np.arange(N))

            Dz_b = D_z @ b_k
            Dz_b_c = np.outer(Dz_b, c_n)

            X = np.zeros((M, L, N), dtype=complex)
            for k in range(M):
                X[k, :, :] = alpha[k] * Dz_b_c

            awgn = np.random.normal(0, var, X.shape) + 1j * np.random.normal(0, var, X.shape)  # complex gaussian noise

            Y = rho * A * X + awgn  # backscattered signal

            y_vec = Y.flatten()

            # Sweep over vector of estimated values
            for i in range(len_hat):
                beta = beta_hat[i]

                for j in range(len_hat):
                    tau = tau_hat[j]

                    for k in range(len_hat):
                        eta = eta_hat[k]

                        alpha = np.exp(1j * 2 * np.pi * m * d / wavelength * np.sin(beta))

                        F = 1 / np.sqrt(L) * np.exp(-1j * (2 * np.pi) / L * kl)

                        D_z = np.diag(F @ zc)
                        b_k = np.exp(-1j * (2 * np.pi) / L * np.arange(L) * tau)
                        c_n = np.exp(1j * (2 * np.pi) * eta * np.arange(N))

                        Dz_b = D_z @ b_k
                        Dz_b_c = np.outer(Dz_b, c_n)

                        h = np.zeros((M, L, N), dtype=complex)
                        for s in range(M):
                            h[s, :, :] = alpha[s] * Dz_b_c

                        h_vec = h.flatten()

                        objectiveFunction[i, j, k] = np.abs(np.vdot(h_vec, y_vec))**2

            U = np.max(objectiveFunction)
            I = np.argmax(objectiveFunction)

            ibeta_hat, itau_hat, ieta_hat = np.unravel_index(I, objectiveFunction.shape)

            if not (objectiveFunction[ibeta_hat, itau_hat, ieta_hat] == U):
                raise ValueError('max search failed')

            Omega_mc_snr[0, iMC, iSNR] = np.abs(beta_hat[ibeta_hat] - beta)**2
            Omega_mc_snr[1, iMC, iSNR] = np.abs(tau_hat[itau_hat] - tau)**2
            Omega_mc_snr[2, iMC, iSNR] = np.abs(eta_hat[ieta_hat] - eta)**2

        Omega_snr[0, iSNR] = np.mean(Omega_mc_snr[0, :, iSNR])
        Omega_snr[1, iSNR] = np.mean(Omega_mc_snr[1, :, iSNR])
        Omega_snr[2, iSNR] = np.mean(Omega_mc_snr[2, :, iSNR])

    # Plot results
    plt.figure()
    plt.plot(SNR, Omega_snr[0, :], label=r'$\beta$')
    plt.plot(SNR, Omega_snr[1, :], label=r'$\tau$')
    plt.plot(SNR, Omega_snr[2, :], label=r'$\eta$')
    plt.title('MSE of parameters over SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

