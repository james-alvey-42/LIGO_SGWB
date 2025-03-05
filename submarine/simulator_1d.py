import swyft
import numpy as np


class SineGaussianSim(swyft.Simulator):
    def __init__(self, fs=100, duration=10, sigma=1, bounds=[(10, 20)]):
        super().__init__()
        self.fs = fs
        self.transform_samples = swyft.to_numpy32

        self.duration = duration
        self.sigma = sigma
        self.bounds = np.array(bounds)
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        self.dt = self.t[1] - self.t[0]
        self.freq_grid = np.fft.rfftfreq(len(self.t), self.dt)

    def noise(self):
        noise_real = np.random.normal(0, self.sigma, len(self.freq_grid))
        noise_imag = np.random.normal(0, self.sigma, len(self.freq_grid))
        return noise_real + 1j * noise_imag

    def sine_gaussian_time(self, Amp, f0, t0, tau):
        gaussian_envelope = np.exp(-((self.t - t0) ** 2) / tau ** 2)
        sine_wave = np.sin(2 * np.pi * f0 * (self.t - t0))
        return Amp * sine_wave * gaussian_envelope

    def source_prior(self):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def signal(self, Amp):
        f0 = 10  # Central frequency of the sine-Gaussian
        t0 = 0.3  # Center of the Gaussian envelope
        tau = 0.1  # Width of the Gaussian envelope
        signal = self.sine_gaussian_time(Amp, f0, t0, tau)
        return np.fft.rfft(signal) * self.dt

    def xi_prior(self):
        return np.random.uniform(0.0, 1.0, 1)

    def get_data(self, noise, signal, xi):
        
        data= signal + noise if np.random.uniform(0.0, 1.0) < xi else noise
        
        return data 
    
    def get_real_data(self,data):
        return np.concatenate((data.real, data.imag), axis=-1)

    def build(self, graph):
        noise = graph.node("noise", self.noise)
        Amp = graph.node("Amp", self.source_prior)
        signal = graph.node("signal", self.signal, Amp)
        xi = graph.node("xi", self.xi_prior)
        data = graph.node("data", self.get_data, noise, signal, xi)
        real_data = graph.node("real_data", self.get_real_data,data)
