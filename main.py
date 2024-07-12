#%%
import numpy as np
import matplotlib.pyplot as plt

from signals import Antennas, Target, Signals, ZadoffChuSequence

if __name__ == "__main__":
    L, q, bandwidth, N = 353, 7, 1e9, 0
    delta = 1/bandwidth
    zc = ZadoffChuSequence(L=L, q=q)
    z = Signals(zc, delta=delta)
    target = Target(0.1, 0.2, 0.5*np.pi, 1e-9, 10)
    antennas = Antennas(10, 1, 1e-9, 5, 10)
    plt.plot(z.backscatter_signal_demodulated(antennas, target, 100)[:, :100])
# %%
