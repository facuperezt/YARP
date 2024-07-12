class Target:
    def __init__(self, attenuation: float, phase_shift: float, angle_of_arrival: float, delay: float, doppler_shift: float):
        """
        Simulates the "Ground truth" for a target and provides the parameters for the backscattered signal
        """
        self.attenuation = attenuation
        self.phase_shift = phase_shift
        self.angle_of_arrival = angle_of_arrival
        self.delay = delay
        self.doppler_shift = doppler_shift

    def __iter__(self):
        return iter((self.attenuation, self.phase_shift, self.angle_of_arrival, self.delay, self.doppler_shift))
    
    def __repr__(self) -> str:
        a = "" \
        f"ϱ = |{self.attenuation:.2f}|*exp(j{self.phase_shift:.2f})\n" \
        f"β = {self.angle_of_arrival:.2f}rad\n" \
        f"τ = {self.delay}s\n" \
        f"ν = {self.doppler_shift}" 

        return a

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def params(self):
        return list(self)