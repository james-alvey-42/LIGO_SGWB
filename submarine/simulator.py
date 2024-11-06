import swyft
import numpy as np


class SubmarineSim(swyft.Simulator):
    def __init__(self):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.freq_grid = np.linspace(0.5, 2.5, 100)
        self.psd = self.PSD(self.freq_grid)
        self.bounds = np.array([[0.5, 2.0], [0.0, 2 * np.pi], [0.1, 1.0]])

    def noise(self):
        return np.random.normal(0.0, np.sqrt(self.psd))

    def PSD(self, f):
        return 1.0 * f**0.0

    def source_prior(self):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def signal(self, theta):
        Amp, f0, T0 = theta
        return np.sin(2 * np.pi * self.freq_grid / T0 + f0) / Amp

    def xi_prior(self):
        return np.random.uniform(0.0, 1.0, 1)

    def get_data(self, noise, signal, xi):
        selection = np.random.uniform(0.0, 1.0) < xi
        if selection:
            return signal + noise
        else:
            return noise

    def build(self, graph):
        noise = graph.node("noise", self.noise)
        theta = graph.node("theta", self.source_prior)
        signal = graph.node("signal", self.signal, theta)
        xi = graph.node("xi", self.xi_prior)
        data = graph.node("data", self.get_data, noise, signal, xi)
