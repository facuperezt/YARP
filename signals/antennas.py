class Antennas:
    def __init__(self, n_antennas: int, dist_antennas: float, wavelength: float, original_amplitude: float, noise_power: float):
        """
        Simulates the "Ground truth" for a target and provides the parameters for the backscattered signal
        """
        self.n_antennas = n_antennas
        self.dist_antennas = dist_antennas
        self.wavelength = wavelength
        self.noise_power = noise_power
        self.original_amplitude = original_amplitude

    def __iter__(self):
        return iter((self.n_antennas, self.dist_antennas, self.wavelength, self.noise_power, self.original_amplitude))
    
    def __repr__(self) -> str:
        a = "" \
        f"# Antennas = {self.n_antennas}" \
        f"d = {self.dist_antennas:.2e}rad\n" \
        f"λ = {self.wavelength:.2e}m\n" \
        f"N0 = {self.noise_power:.2e}Hz\n" \
        f"A = {self.original_amplitude:.2e}\n" 

        return a

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def params(self):
        return list(self)