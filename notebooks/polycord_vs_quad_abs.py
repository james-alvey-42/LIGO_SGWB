#%%
import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import subprocess

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
    Z_S, errors= nquad(
        integrand,
        ranges=bounds,
        args=(s_obs, freq_grid, sigma, bounds)
    )
    return Z_S,errors

# Compute noise evidence (Z_N)
def compute_ZN(s_obs, freq_grid, sigma):
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    return np.exp(norm - 0.5 *np.sum((s_obs / sigma)**2))

# Simulate multiple segments with changing s_obs
bounds = [(0.1, 2), (0.5, 2.0), (0, 1), (0.1, 0.5)]  # Parameter bounds: [A, f0, t0, tau]
freq_grid = np.linspace(0.5, 2.5, 100)

sigma=1
xi=0.4
s_obs_list=[]
Z_S_list=[]
Z_N_list=[]

for i in tqdm(range(100)):
    Amp_true = np.random.uniform(bounds[0][0], bounds[0][1])
    f0_true = np.random.uniform(bounds[1][0], bounds[1][1])
    t0_true = np.random.uniform(bounds[2][0], bounds[2][1])
    tau_true = np.random.uniform(bounds[3][0], bounds[3][1])
    s_obs    = observed_strain(Amp_true, freq_grid, f0_true, t0_true, tau_true, sigma, xi)
    s_obs_list.append(s_obs)
    Z_S,errors = compute_ZS(s_obs, freq_grid, sigma, bounds)
    Z_N = compute_ZN(s_obs, freq_grid, sigma)
    Z_S_list.append(Z_S)
    Z_N_list.append(Z_N)
print(np.log(Z_S_list),np.log(Z_N_list))


def run_polychord_for_strain(index, s_obs):
    output_dir = os.path.join("outputs", f"sine_gaussian_{index + 1}")
    os.makedirs(output_dir, exist_ok=True)

    script_name = f"run_strain_{index + 1}.py"

    # Dynamically generate the script
    script = f"""
import os
from cobaya.run import run
import numpy as np

# Likelihood function for PolyChord

# Sine-Gaussian in frequency domain
def sine_gaussian_freq(f, Amp, f0, t0, tau):
    gaussian_envelope = np.exp(-np.pi**2 * tau**2 * (f - f0)**2 / 4)
    time_shift = np.exp(-2j * np.pi * f * t0)
    return Amp * (tau / 2) * gaussian_envelope * time_shift
    
    
def likelihood(s_obs, sigma, freq_grid, **kwargs):
    Amp = kwargs["Amp"]
    f0 = kwargs["f0"]
    t0 = kwargs["t0"]
    tau = kwargs["tau"]

    h_theta = np.abs(sine_gaussian_freq(freq_grid, Amp, f0, t0, tau))
    diff = s_obs - h_theta
    norm = -0.5 * len(freq_grid) * np.log(2 * np.pi * sigma**2)
    return norm - 0.5* np.sum((diff / sigma)**2)

    return logL

# Frequency grid
freq_grid = np.linspace(0.5, 2.5, 100)

# Observed strain
s_obs = np.array({list(s_obs)})
sigma = {1.0}

# Define parameter space
parameters_dictionary = {{
    "Amp": {{"prior": {{"min": 0.1, "max": 2.0}}}},  # Amplitude
    "f0": {{"prior": {{"min": 0.5, "max": 2.0}}}},  # Frequency
    "t0": {{"prior": {{"min": 0.0, "max": 1.0}}}},  # Time offset
    "tau": {{"prior": {{"min": 0.1, "max": 0.5}}}},  # Decay time
}}

# PolyChord settings
polychord_settings = {{
    "nlive": 300,
    "precision_criterion": 1e-5,
}}

def wrapped_likelihood(**kwargs):
    return likelihood(s_obs, sigma, freq_grid, **kwargs)

info = {{
    "likelihood": {{
        "sine_gaussian_likelihood": {{
            "external": wrapped_likelihood,
            "input_params": ["Amp", "f0", "t0", "tau"],
        }}
    }},
    "params": parameters_dictionary,
    "sampler": {{"polychord": polychord_settings}},
    "output": "{output_dir}",
}}

updated_info, sampler = run(info)
"""

    # Write the script to a file
    with open(script_name, "w") as f:
        f.write(script)

    try:
        # Run the script in a separate Python process
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running strain {index + 1}: {e}")
    finally:
        # Clean up the temporary script file
        if os.path.exists(script_name):
            os.remove(script_name)

# Loop through observed strains
for i, s_obs in enumerate(s_obs_list):
    run_polychord_for_strain(i, s_obs)
# %%

def total_log_likelihood(xi, ZS_list, ZN_list):
    log_likelihood = 0.0
    for Z_S, Z_N in zip(ZS_list, ZN_list):
        term = xi * Z_S + (1 - xi) * Z_N
        log_likelihood += np.log(term)

    return log_likelihood


xi_values = np.linspace(0, 1, 101)

log_likelihood_values = [total_log_likelihood(x, Z_S_list, Z_N_list) for x in xi_values]
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
print("Signal Evidences (Z_S):", Z_S_list)
print("Noise Evidences (Z_N):", Z_N_list)



# %%
