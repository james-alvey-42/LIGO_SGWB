
#%%
import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define sine-Gaussian function in time domain
def sine_gaussian_time(t, Amp, f0, t0, tau):
    gaussian_envelope = np.exp(-((t - t0)**2) / tau**2)
    sine_wave = np.sin(2 * np.pi * f0 * (t - t0))
    return Amp * sine_wave * gaussian_envelope

# Define time parameters
fs = 100  # Sampling frequency (Hz)
duration = 10  # Signal duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time array
dt = t[1] - t[0]  # Time step

# Signal parameters
Amp = 1 # Initial amplitude
f0 = 10  # Central frequency of the sine-Gaussian
t0 = 0.3 # Center of the Gaussian envelope
tau = 0.1  # Width of the Gaussian envelope

# Noise parameters
sigma = 1  # Standard deviation of noise

# Generate sine-Gaussian signal in time domain
signal_time = sine_gaussian_time(t, Amp, f0, t0, tau)

# Compute FFT of the sine-Gaussian signal (using rFFT for real-valued signals)
signal_freq_fft = np.fft.rfft(signal_time) * dt

# Compute frequency grid (only positive frequencies)
freqs = np.fft.rfftfreq(len(t), dt)

# Compute power spectral density (PSD) of the signal
power_spectrum_signal = np.abs(signal_freq_fft) ** 2

# Compute frequency resolution Δf
delta_f = 1 / (t[-1] - t[0])  # Δf = 1 / T
#print(delta_f,freqs[1]-freqs[0])
# Compute noise power spectral density S_n(f) = 2σ²Δf
S_n_f = 2 * sigma**2 * delta_f

# Compute correct SNR using the updated definition
SNR_computed_corrected = np.sqrt(np.sum(power_spectrum_signal / S_n_f))

# Compute required amplitude for SNR = 0.5
desired_SNR = 0.5
A_required_corrected = (desired_SNR / SNR_computed_corrected) * Amp

# Display results
print(f"Computed SNR: {SNR_computed_corrected}")
print(f"Required amplitude for SNR=0.5: {A_required_corrected}")

# Plot frequency spectrum of the signal
plt.figure(figsize=(8, 5))
plt.plot(freqs, power_spectrum_signal, label="Power Spectrum |S(f)|²")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectrum")
plt.title("Fourier Transform of Sine-Gaussian Signal")
plt.grid()
plt.legend()
plt.show()

#%%
import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define sine-Gaussian function in time domain
def sine_gaussian_time(t, Amp, f0, t0, tau):
    gaussian_envelope = np.exp(-((t - t0)**2) / tau**2)
    sine_wave = np.sin(2 * np.pi * f0 * (t - t0))
    return Amp * sine_wave * gaussian_envelope

# Observed strain: signal + noise
def observed_strain(Amp,t,freq_grid,sigma, xi):
    noise_real = np.random.normal(0, sigma, len(freq_grid))
    noise_imag = np.random.normal(0, sigma, len(freq_grid))
    noise = noise_real + 1j * noise_imag
    f0 = 10  # Central frequency of the sine-Gaussian
    t0 = 0.3  # Center of the Gaussian envelope
    tau = 0.1  # Width of the Gaussian envelope
    signal = sine_gaussian_time(t, Amp, f0, t0, tau)
    h_f = np.fft.rfft(signal)*(t[1]-t[0])  # Use np.fft.rfft for consistency with rfftfreq
    return h_f + noise if np.random.uniform(0.0, 1.0) < xi else noise

def observed_strain_plot(Amp,t,freq_grid,sigma, xi):
    noise_real = np.random.normal(0, sigma, len(freq_grid))
    noise_imag = np.random.normal(0, sigma, len(freq_grid))
    noise = noise_real + 1j * noise_imag
    f0 = 10  # Central frequency of the sine-Gaussian
    t0 = 0.3  # Center of the Gaussian envelope
    tau = 0.1  # Width of the Gaussian envelope
    signal = sine_gaussian_time(t, Amp, f0, t0, tau)
    h_f = np.fft.rfft(signal)*(t[1]-t[0])  # Use np.fft.rfft for consistency with rfftfreq
    return h_f + noise if np.random.uniform(0.0, 1.0) < xi else noise,h_f,noise


"""
plt.plot(freqs,(signal).real)
plt.plot(freqs,(noise).real,alpha=0.3)
plt.plot(freqs,(obs).real,alpha=0.5)
"""

#%%

# Log-Likelihood function
def log_likelihood(Amp,t, s_obs, freq_grid, sigma):
    f0 = 10  # Central frequency of the sine-Gaussian
    t0 = 0.3  # Center of the Gaussian envelope
    tau = 0.1  # Width of the Gaussian envelope
    signal = sine_gaussian_time(t, Amp, f0, t0, tau)
    h_f = np.fft.rfft(signal)*(t[1]-t[0])
    diff = s_obs - h_f
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    delta_f = freq_grid[1] - freq_grid[0]  # Frequency bin width
    return - 0.5*4*np.real(delta_f * np.sum(np.conj(diff)*diff/ sigma))

# Prior function
def prior(Amp, bounds):
    A_min, A_max = bounds[0]
   
    if A_min <= Amp <= A_max:
        volume = np.prod([b[1] - b[0] for b in bounds])
        return 1.0 / volume
    return 0.0

# Combined integrand (likelihood * prior)
def integrand(Amp,t, s_obs, freq_grid, sigma, bounds):
    likelihood_value = np.exp(log_likelihood(Amp, t,s_obs, freq_grid, sigma))
    prior_value = prior(Amp, bounds)
    return likelihood_value * prior_value

# Compute signal evidence (Z_S)
def compute_ZS(t,s_obs, freq_grid, sigma, bounds):
    Z_S, _ = nquad(
        integrand,
        ranges=bounds,
        args=(t,s_obs, freq_grid, sigma, bounds)
    )
    return Z_S

# Compute noise evidence (Z_N)
def compute_ZN(s_obs, freq_grid, sigma):
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    delta_f = freq_grid[1] - freq_grid[0]  # Frequency bin width
    #print(norm - 0.5 *4*np.real(delta_f * np.sum(np.conj(s_obs)*s_obs/ sigma)))
    return np.exp(- 0.5 *4*np.real(delta_f * np.sum(np.conj(s_obs)*s_obs/ sigma)))

# Simulate multiple segments with changing s_obs
def simulate_segments(num_segments, freq_grid, t,sigma, bounds, xi):
    ZS_list = []
    ZN_list = []

    for i in tqdm(range(num_segments)):
        Amp_true = np.random.uniform(bounds[0][0], bounds[0][1])
        print(bounds[0][0], bounds[0][1])
        print(Amp_true)
        s_obs = observed_strain(Amp_true,t,freq_grid, sigma, xi)

        Z_S = compute_ZS(t,s_obs, freq_grid, sigma, bounds)
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

#%%
fs = 100  # Sampling frequency (Hz)
duration = 10  # Signal duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time array
dt = t[1] - t[0]  # Time step
freqs = np.fft.rfftfreq(len(t), dt)
sigma = 1  # Standard deviation of noise

bounds = [(10,20)]


# %%
xi=0.5
num_segments=1000
ZS_list_1, ZN_list_1=simulate_segments(num_segments, freqs, t,sigma, bounds, xi)


# %%
xi_values = np.linspace(0, 1, 101)

log_likelihood_values = [total_log_likelihood(x, ZS_list_1, ZN_list_1) for x in xi_values]
likelihood_values = np.exp(log_likelihood_values - np.max(log_likelihood_values))  # Normalize for stability

# %%
# Plot total likelihood
plt.plot(xi_values, likelihood_values, label="Total Likelihood (Normalized)")
plt.xlabel("ξ")
plt.ylabel("Likelihood")
plt.axvline(0.5)
plt.title("Total Likelihood vs ξ (Log-Safe)")
plt.grid(True)
plt.legend()
plt.show()
# %%

# %%

# Log-Likelihood function with normalization
def log_likelihood_withnorm(Amp, t, s_obs, freq_grid, sigma):
    f0 = 10  # Central frequency of the sine-Gaussian
    t0 = 0.3  # Center of the Gaussian envelope
    tau = 0.1  # Width of the Gaussian envelope
    signal = sine_gaussian_time(t, Amp, f0, t0, tau)
    h_f = np.fft.rfft(signal) * (t[1] - t[0])
    diff = s_obs - h_f
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    delta_f = freq_grid[1] - freq_grid[0]
    return norm - 0.5 * 4 * np.real(delta_f * np.sum(np.conj(diff) * diff / sigma))

# Find the maximum log-likelihood value for numerical stability
def find_logL_max(bounds, t, s_obs, freq_grid, sigma):
    A_vals = np.linspace(bounds[0][0], bounds[0][1], 100)
    logL_vals = [log_likelihood_withnorm(A, t, s_obs, freq_grid, sigma) for A in A_vals]
    return max(logL_vals)

# Compute signal evidence (Z_S)
def compute_ZS_withnorm(t, s_obs, freq_grid, sigma, bounds):
    logL_max = find_logL_max(bounds, t, s_obs, freq_grid, sigma)
    
    def integrand(Amp, t, s_obs, freq_grid, sigma, bounds, logL_max):
        logL = log_likelihood_withnorm(Amp, t, s_obs, freq_grid, sigma)
        likelihood_value = np.exp(logL - logL_max)
        prior_value = prior(Amp, bounds)
        return likelihood_value * prior_value
    
    Z_S_shifted, _ = nquad(
        integrand,
        ranges=bounds,
        args=(t, s_obs, freq_grid, sigma, bounds, logL_max)
    )
    return np.exp(logL_max) * Z_S_shifted

# Compute noise evidence (Z_N)
def compute_ZN_withnorm(s_obs, freq_grid, sigma):
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    delta_f = freq_grid[1] - freq_grid[0]
    logZ_N = norm - 0.5 * 4 * np.real(delta_f * np.sum(np.conj(s_obs) * s_obs / sigma))
    return np.exp(logZ_N)

# Simulate multiple segments
def simulate_segments(num_segments, freq_grid, t, sigma, bounds, xi):
    ZS_list_withnorm = []
    ZN_list_withnorm = []
    ZS_list = []
    ZN_list = []
    for _ in tqdm(range(num_segments)):
        Amp_true = np.random.uniform(bounds[0][0], bounds[0][1])
        s_obs = observed_strain(Amp_true, t, freq_grid, sigma, xi)
        Z_S_withnorm = compute_ZS_withnorm(t, s_obs, freq_grid, sigma, bounds)
        Z_N_withnorm = compute_ZN_withnorm(s_obs, freq_grid, sigma)
        Z_S = compute_ZS(t, s_obs, freq_grid, sigma, bounds)
        Z_N = compute_ZN(s_obs, freq_grid, sigma)
        ZS_list_withnorm.append(Z_S_withnorm)
        ZN_list_withnorm.append(Z_N_withnorm)
        ZS_list.append(Z_S)
        ZN_list.append(Z_N)
    return ZS_list_withnorm, ZN_list_withnorm,ZS_list,ZN_list

# Parameters
fs = 100  # Sampling frequency (Hz)
duration = 10  # Signal duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
dt = t[1] - t[0]
freqs = np.fft.rfftfreq(len(t), dt)
sigma = 1  # Standard deviation of noise
bounds = [(10, 20)]  # Amplitude bounds
xi = 0.5
num_segments = 1000

# Run simulation
ZS_list_withnorm, ZN_list_withnorm,ZS_list,ZN_list = simulate_segments(num_segments, freqs, t, sigma, bounds, xi)
# %%
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

# %%
xi_values = np.linspace(0, 1, 101)

log_likelihood_values_withnorm = [total_log_likelihood(x, ZS_list_withnorm, ZN_list_withnorm) for x in xi_values]
likelihood_values_withnorm = np.exp(log_likelihood_values_withnorm - np.max(log_likelihood_values_withnorm))  # Normalize for stability

log_likelihood_values= [total_log_likelihood(x, ZS_list, ZN_list) for x in xi_values]
likelihood_values = np.exp(log_likelihood_values - np.max(log_likelihood_values)) 
# %%

# Plot total likelihood
plt.plot(xi_values, likelihood_values_withnorm,linestyle=':', label="Total Likelihood (Normalized)")

plt.plot(xi_values, likelihood_values,alpha=0.4, label="Total Likelihood (unormalized)")
plt.xlabel("ξ")
plt.ylabel("Likelihood")
plt.axvline(0.5)
plt.title("Total Likelihood vs ξ (Log-Safe)")
plt.grid(True)
plt.legend()
plt.show()

# %%
