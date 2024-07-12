#%%
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from math import gcd as bltin_gcd

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


class UnitEnergyChippedSignal:
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

    def one_cycle(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns one cycle of the complex envelope signal

        z(t) in the project
        """
        x = np.linspace(0, self.zc.L*self.delta, int(self.zc.L*self.resolution))
        return x, np.repeat(self.zc(), self.resolution)*(1/np.sqrt(self.delta))  # Divide by sqrt(delta) to make the pulse unit energy
    
    def compare_sequence_to_cycle(self) -> None:
        x, cycle = self.one_cycle()
        plt.figure()
        plt.plot(self.zc(), label = "Original Sequence")
        plt.plot(np.linspace(0, len(self.zc()), len(x)), cycle, label = "Resulting Cycle")
        plt.legend()
        plt.show()

    def signal(self, N: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the whole signal 
        Not sure what N is, so its a variable now
        Calling it with N = 0 is the same as calling one_cycle()

        the return variable x holds the time information
        x(t)
        """
        cycle = self.one_cycle()[1]
        signal = np.tile(cycle, N+1)
        x = np.linspace(0, self.zc.L*self.delta*(N+1), int(self.zc.L*self.resolution*(N+1)))
        return x, signal
#%%
if __name__ == "__main__":
    L, q, delta, N = 61, 4, 0.15, 10
    zc = ZadoffChuSequence(L=L, q=q)
    z = UnitEnergyChippedSignal(zc, delta=delta)
    t, x_t = z.signal(N=N)
    plt.plot(t, x_t)