
#%%
import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
from tqdm import tqdm
# Sine-Gaussian in frequency domain
def sine_gaussian_time(t, Amp, f0, t0, tau):
    gaussian_envelope = np.exp(-((t - t0)**2) / tau**2)
    sine_wave = np.sin(2 * np.pi * f0 * (t - t0))
    return Amp * sine_wave * gaussian_envelope

# Observed strain: signal + noise
def observed_strain(Amp,freq_grid, f0, t0, tau, sigma, xi):
    noise = np.random.normal(0, sigma, len(freq_grid))
    signal = sine_gaussian_time(t, Amp, f0, t0, tau)
    h_f = np.fft.rfft(signal)  # Use np.fft.rfft for consistency with rfftfreq
    return h_f + noise if np.random.uniform(0.0, 1.0) < xi else noise

# Log-Likelihood function
def log_likelihood(Amp, f0, t0, tau, s_obs, freq_grid, sigma):
    signal = sine_gaussian_time(t, Amp, f0, t0, tau)
    h_f = np.fft.rfft(signal) 
    diff = s_obs - h_f
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    delta_f = freq_grid[1] - freq_grid[0]  # Frequency bin width
    return norm - 0.5*4*np.real(delta_f * np.sum(np.conj(diff)*diff/ sigma))

# Prior function
def prior(Amp, f0, t0, tau, bounds):
    A_min, A_max = bounds[0]
    f0_min, f0_max = bounds[1]
    t0_min, t0_max = bounds[2]
    tau_min, tau_max = bounds[3]

    if A_min <= Amp <= A_max and f0_min <= f0 <= f0_max and t0_min <= t0 <= t0_max and tau_min <= tau <= tau_max:
        volume = np.prod([b[1] - b[0] for b in bounds])
        return 1.0 / volume
    return 0.0

# Combined integrand (likelihood * prior)
def integrand(Amp, f0, t0, tau, s_obs, freq_grid, sigma, bounds):
    likelihood_value = np.exp(log_likelihood(Amp, f0, t0, tau, s_obs, freq_grid, sigma))
    prior_value = prior(Amp, f0, t0, tau, bounds)
    return likelihood_value * prior_value

# Compute signal evidence (Z_S)
def compute_ZS(s_obs, freq_grid, sigma, bounds):
    Z_S, _ = nquad(
        integrand,
        ranges=bounds,
        args=(s_obs, freq_grid, sigma, bounds)
    )
    return Z_S

# Compute noise evidence (Z_N)
def compute_ZN(s_obs, freq_grid, sigma):
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    delta_f = freq_grid[1] - freq_grid[0]  # Frequency bin width
    print(norm - 0.5 *4*np.real(delta_f * np.sum(np.conj(s_obs)*s_obs/ sigma)))
    return np.exp(norm - 0.5 *4*np.real(delta_f * np.sum(np.conj(s_obs)*s_obs/ sigma)))

# Simulate multiple segments with changing s_obs
def simulate_segments(num_segments, freq_grid, sigma, bounds, xi):
    ZS_list = []
    ZN_list = []

    for i in tqdm(range(num_segments)):
        Amp_true = np.random.uniform(bounds[0][0], bounds[0][1])
        f0_true = np.random.uniform(bounds[1][0], bounds[1][1])
        t0_true = np.random.uniform(bounds[2][0], bounds[2][1])
        tau_true = np.random.uniform(bounds[3][0], bounds[3][1])
        print(Amp_true,f0_true,t0_true, tau_true  )
        s_obs = observed_strain(Amp_true, freq_grid, f0_true, t0_true, tau_true, sigma, xi)

        Z_S = compute_ZS(s_obs, freq_grid, sigma, bounds)
        Z_N = compute_ZN(s_obs, freq_grid, sigma)
        print(Z_N )
        ZS_list.append(Z_S)
        ZN_list.append(Z_N)

    return ZS_list, ZN_list

# Compute total log-likelihood
def total_log_likelihood(xi, ZS_list, ZN_list):
    log_likelihood = 0.0
    for Z_S, Z_N in zip(ZS_list, ZN_list):
        term = xi * Z_S + (1 - xi) * Z_N
        if term > 0:
            log_likelihood += np.log(term)
        else:
            log_likelihood += -np.inf
    return log_likelihood

# Parameters
sigma = 1.0

bounds = [(0.4, 0.9), (1, 40), (0.1,4), (0.05, 0.1)]
num_segments = 1
xi_values = np.linspace(0, 1, 101)
xi = 0.8
fs = 100  # Sampling frequency (Hz)
duration = 10# Signal duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time array
freq_grid = np.fft.rfftfreq(len(t), 1 / fs)  # Frequency array

#%%

# Simulate evidences
ZS_list,ZN_list = simulate_segments(num_segments, freq_grid, sigma, bounds, xi)

# Compute likelihood values
log_likelihood_values = [total_log_likelihood(x, ZS_list, ZN_list) for x in xi_values]
likelihood_values = np.exp(log_likelihood_values - np.max(log_likelihood_values))  # Normalize for stability

# Plot total likelihood
plt.plot(xi_values, likelihood_values, label="Total Likelihood (Normalized)")
plt.xlabel("ξ")
plt.ylabel("Likelihood")
plt.title("Total Likelihood vs ξ (Log-Safe)")
plt.grid(True)
plt.legend()
plt.show()

# Print evidences
print("Signal Evidences (Z_S):", ZS_list)
print("Noise Evidences (Z_N):", ZN_list)

