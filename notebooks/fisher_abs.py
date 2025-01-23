
#%%
import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
from tqdm import tqdm

# Sine-Gaussian in frequency domain
def sine_gaussian_freq(f, Amp, f0, t0, tau):
    gaussian_envelope = np.exp(-np.pi**2 * tau**2 * (f - f0)**2 / 4)
    time_shift = np.exp(-2j * np.pi * f * t0)
    return Amp * (tau / 2) * gaussian_envelope * time_shift

# Observed strain: signal + noise
def observed_strain(Amp_true, freq_grid, f0, t0, tau, sigma, xi):
    noise = np.random.normal(0, sigma, len(freq_grid))
    signal = np.abs(sine_gaussian_freq(freq_grid, Amp_true, f0, t0, tau))
    return signal + noise if np.random.uniform(0.0, 1.0) < xi else noise

# Log-Likelihood function
def log_likelihood(Amp, f0, t0, tau, s_obs, freq_grid, sigma):
    h_theta = np.abs(sine_gaussian_freq(freq_grid, Amp, f0, t0, tau))
    diff = s_obs - h_theta
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    return norm - 0.5* np.sum((diff / sigma)**2)

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
    return np.exp(norm - 0.5 *np.sum((s_obs / sigma)**2))

# Simulate multiple segments with changing s_obs
def simulate_segments(num_segments, freq_grid, sigma, bounds, xi):
    ZS_list = []
    ZN_list = []

    for i in tqdm(range(num_segments)):
        Amp_true = np.random.uniform(bounds[0][0], bounds[0][1])
        f0_true = np.random.uniform(bounds[1][0], bounds[1][1])
        t0_true = np.random.uniform(bounds[2][0], bounds[2][1])
        tau_true = np.random.uniform(bounds[3][0], bounds[3][1])
        s_obs = observed_strain(Amp_true, freq_grid, f0_true, t0_true, tau_true, sigma, xi)

        Z_S = compute_ZS(s_obs, freq_grid, sigma, bounds)
        Z_N = compute_ZN(s_obs, freq_grid, sigma)

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
freq_grid = np.linspace(0.5, 2.5, 100)
sigma = 1.0
bounds = [(0.1, 2), (0.5, 2.0), (0, 1), (0.1, 0.5)]
num_segments = 1000
xi_values = np.linspace(0, 1, 101)
xi = 0.8

#%%
# Simulate evidences
ZS_list, ZN_list = simulate_segments(num_segments, freq_grid, sigma, bounds, xi)

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

