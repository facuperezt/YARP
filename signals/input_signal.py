#%%
from typing import Iterable, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from math import gcd as bltin_gcd

from antennas import Antennas
from target import Target

class ZadoffChuSequence:
    """
    Implements a discrete Zadoff-Chu sequence.
    Requires L and q to be co-primes.
    """
    def __init__(self, L: int, q: int) -> None:
        """
        params:
        L - length of sequence
        q - root
        """
        if not self._is_coprime(L, q):
            raise ValueError("N and q have to be co-primes.")
        self.L = L
        self.q = q

    def __call__(self, l: Optional[int] = None) -> np.ndarray:
        if l is not None:
            return self._formula(l)
        return self._formula(np.arange(self.L))

    @staticmethod
    def _first_prime_factor(n):
        if n & 1 == 0:
            return 2
        d= 3
        while d * d <= n:
            if n % d == 0:
                return d
            d= d + 2
        return n

    @staticmethod
    def _isprime(n: int) -> bool:
        return ZadoffChuSequence._first_prime_factor(n) == n
    
    @staticmethod
    def _is_coprime(n1: int, n2: int) -> bool:
        return bltin_gcd(n1, n2) == 1

    def _formula(self, l: int) -> float:
        return np.exp(-1j*np.pi*self.q*((l*(l+1)) / self.L))


class Signals:
    """
    Handles generating a chipped sequence based of a ZC sequence
    """
    def __init__(self, zc: ZadoffChuSequence, delta: float, resolution: int = 100) -> None:
        """
        params:

        resolution - Points per delta
        """
        self.zc = zc
        self.delta = delta
        self.resolution = resolution

    def one_complex_envelope_cycle(self, normalize_pulse_energy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns one cycle of the complex envelope signal

        z(t) in the project
        """
        x = np.linspace(0, self.zc.L*self.delta, int(self.zc.L*self.resolution))
        out = np.repeat(self.zc(), self.resolution)
        if normalize_pulse_energy:
            out *= (1/np.sqrt(self.delta))  # Divide by sqrt(delta) to make the pulse unit energy
        return x, out
    
    def compare_sequence_to_cycle(self) -> None:
        x, cycle = self.one_complex_envelope_cycle()
        plt.figure()
        plt.plot(self.zc(), label = "Original Sequence")
        plt.plot(np.linspace(0, len(self.zc()), len(x)), cycle, label = "Resulting Cycle")
        plt.legend()
        plt.show()

    def complex_envelope(self, N: int = 0, normalize_pulse_energy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the whole signal 
        Not sure what N is, so its a variable now
        Calling it with N = 0 is the same as calling one_cycle()

        the return variable x holds the time information
        x(t)
        """
        cycle = self.one_complex_envelope_cycle(normalize_pulse_energy)[1]
        complex_envelope = np.tile(cycle, N+1)
        x = np.linspace(0, self.zc.L*self.delta*(N+1), int(self.zc.L*self.resolution*(N+1)))
        return x, complex_envelope
    
    def illuminator_signal(self, amplitude: float, modulating_frequency: float, N: int = 100, normalize_pulse_energy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the actual signal

        s(t) = Re{A x(t) e^(j2pifot)}
        """
        t, complex_envelope = self.complex_envelope(N=N, normalize_pulse_energy=normalize_pulse_energy)
        exp = np.exp(1j*2*np.pi*modulating_frequency*t)
        return t, (amplitude*complex_envelope*exp).real
    
    def _demodulated_backscattered_signal(
            self,
            n_antennas: int,
            dist_antennas: float,
            wavelength: float,
            noise_power: float,
            original_amplitude: float,
            attenuation: float,
            phase_shift: float,
            angle_of_arrival: float,
            delay: float,
            doppler_shift: float,
            N: int,
            normalize_pulse_energy: bool,
    ):
        t, complex_envelope = self.complex_envelope(N=N, normalize_pulse_energy=normalize_pulse_energy)
        rho = attenuation * np.exp(1j*phase_shift)
        tau_shift = np.argmin(np.abs((t-delay) - t[0]))
        t_delayed = t[tau_shift:]
        complex_envelope_delayed = complex_envelope[tau_shift:]
        exp1 = np.exp(1j*2*np.pi*doppler_shift*t_delayed)
        exp2 = np.exp(1j*2*np.pi*np.sin(angle_of_arrival))
        common_result = rho*original_amplitude*complex_envelope_delayed*exp1*exp2
        each_result = []
        for m in range(n_antennas):
            exp3 = np.exp(1j*2*np.pi*m*(dist_antennas/wavelength))
            noise = np.random.multivariate_normal(mean=[0, 0], cov=[[noise_power,0],[0,noise_power]])
            each_result.append(common_result*exp3 + (noise[0] + 1j*noise[1]))
        return np.stack(each_result)

    def backscatter_signal_demodulated(self, antennas: Antennas, target: Target, N: int, normalize_pulse_energy: bool = False):
        return self._demodulated_backscattered_signal(*antennas.params, *target.params, N=N, normalize_pulse_energy=normalize_pulse_energy)

#%%
if __name__ == "__main__":
    # Plots the same Zadoff-Chu sequence from wikipedia, but in discrete time using the unit-energy pulse
    L, q, bandwidth, N = 353, 7, 1e9, 0
    delta = 1/bandwidth
    zc = ZadoffChuSequence(L=L, q=q)
    z = Signals(zc, delta=delta)
    t, x_t = z.complex_envelope(N=N)
    fig, axs = plt.subplots(2, 1, figsize=(20, 5))
    axs[0].plot(t, x_t.real)
    axs[1].plot(t, x_t.imag)
    plt.figure()
    amplitude, f0 = 10, 10e6
    plt.plot(*z.illuminator_signal(amplitude, f0, N=N))
    plt.show()

    target = Target(0.1, 0.2, 0.5*np.pi, 1e-9, 10)
    antennas = Antennas(10, 1, 1e-9, amplitude, 10)
    plt.plot(z.backscatter_signal_demodulated(antennas, target, 100)[:, :100])

# %%
