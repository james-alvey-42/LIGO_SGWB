#%%
import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import swyft
import sys
from swyft.networks import OnlineStandardizingLayer, ResidualNetWithChannel
sys.path.insert(0, "../submarine")
import simulator2 as sim
import data
# %%

#%%
# Define sine-Gaussian function in time domain
def sine_gaussian_time(t, Amp, f0, t0, tau):
    gaussian_envelope = np.exp(-((t - t0)**2) / tau**2)
    sine_wave = np.sin(2 * np.pi * f0 * (t - t0))
    return Amp * sine_wave * gaussian_envelope

def prior(Amp, bounds):
    A_min, A_max = bounds[0]
   
    if A_min <= Amp <= A_max:
        volume = np.prod([b[1] - b[0] for b in bounds])
        return 1.0 / volume
    return 0.0

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

# Parameters
fs = 100  # Sampling frequency (Hz)
duration = 10  # Signal duration (seconds)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
dt = t[1] - t[0]
freqs = np.fft.rfftfreq(len(t), dt)
sigma = 1  # Standard deviation of noise
bounds = [(10, 20)]  # Amplitude bounds
xi = 0.3
num_segments = 1
# %%
simulator = sim.SineGaussianSim()
sample = simulator.sample()
plt.plot(sample['noise'])
plt.plot(sample['signal'])
plt.plot(sample['data'])

# %%
real_data_list = []
xi_labels = []
for _ in range(100000):
    sample = simulator.sample()    
    real_data_list.append(sample['real_data'])
    xi_labels.append(sample['xi'])

# %%
import torch
from sbi import utils as utils

from sbi.inference import SNPE
prior = utils.BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

# Convert to tensors
real_data_tensor = torch.tensor(np.array(real_data_list), dtype=torch.float32)
xi_tensor = torch.tensor(np.array(xi_labels), dtype=torch.float32)
real_data_tensor.shape

inference = SNPE(prior)
density_estimator = inference.append_simulations(theta=xi_tensor, x=real_data_tensor).train()





#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import swyft
from sbi import utils as utils
from sbi.inference import SNPE

# === Define a Neural Network That Computes log p(xi | s) ===
class ZsZnNetwork(nn.Module):
    """ Neural network that models both Z_S and Z_N, then computes the posterior density. """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Outputs Zs and Zn
        )

    def forward(self, s, xi):
        """
        Forward pass:
        - Compute Z_S(s) and Z_N(s).
        - Compute posterior density log p(xi | s).
        """
        zs, zn = self.nn(s).chunk(2, dim=-1)  # Split output into Zs and Zn
        zs, zn = zs.exp(), zn.exp()  # Ensure positivity

        # Compute log-likelihood of xi given s
        likelihood = zs * xi + zn * (1 - xi)
        log_likelihood = torch.log(likelihood + 1e-9)  # Add epsilon for numerical stability
        
        return log_likelihood

# === Generate Training Data Using Swyft Simulator ===
simulator = sim.SineGaussianSim()
num_samples = 10000

real_data_list = []
xi_labels = []

for _ in range(num_samples):
    sample = simulator.sample()
    real_data_list.append(sample['real_data'])
    xi_labels.append(sample['xi'])

# Convert to tensors
real_data_tensor = torch.tensor(np.array(real_data_list), dtype=torch.float32)
xi_tensor = torch.tensor(np.array(xi_labels), dtype=torch.float32)
real_data_tensor.shape
xi_tensor 
#%%
# === Define the Prior for xi ===
prior = utils.BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

# === Train the Custom Posterior Model Using SNPE ===
inference = SNPE(prior)
density_estimator = inference.append_simulations(theta=xi_tensor, x=real_data_tensor).train()
#%%
posterior=inference.build_posterior()

#%%
nsamples=10000
samples=posterior.sample((nsamples,), x=real_data_tensor[10])
samples.shape
plt.hist(samples[:,0],bins=100)
plt.axvline(xi_labels[0])

samples
#%%
# === Compute Posterior for a New Observation ===
def compute_posterior(observation):
    """
    Compute p(xi | s) using the trained network.
    """
    posterior_samples = density_estimator.sample((1000,), x=torch.tensor(observation, dtype=torch.float32))
    return posterior_samples

# === Example: Compute Posterior for a New Observation ===
obs_noise = sim.noise()
obs_signal = sim.signal(15)  # Example with an arbitrary amplitude
obs_data = sim.get_data(obs_noise, obs_signal, 0.7)  # Example with xi = 0.7
obs_real_data = sim.get_real_data(obs_data)

post_vals = compute_posterior(obs_real_data)

# === Plot Posterior ===
plt.hist(post_vals.numpy(), bins=50, density=True, alpha=0.75, label="Estimated Posterior")
plt.xlabel(r'$\xi$')
plt.ylabel(r'Posterior $p(\xi|\bar{s})$')
plt.title('Posterior Estimated from Custom Density Estimator')
plt.legend()
plt.show()

# %%
# === Train the Custom Posterior Model Using SNPE ===
inference = SNPE(prior, density_estimator=density_estimator_fn(1002))
density_estimator = inference.append_simulations(theta=xi_tensor, x=real_data_tensor).train()

# %%
real_data_tensor.shape
# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import swyft
from sbi import utils as utils
from sbi.inference import SNPE
from sbi.utils.sbiutils import standardizing_net
#from sbi.neural_nets.base import DensityEstimator

# === Define a Neural Network for Z_S and Z_N ===
class ZsZnNetwork(nn.Module):
    """ Neural network that models both Z_S and Z_N. """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Outputs Zs and Zn
        )

    def forward(self, s):
        zs, zn = self.nn(s).chunk(2, dim=-1)  # Split output into Zs and Zn
        return zs, zn  # Ensure positivity

# === Define Custom Density Estimator ===
class CustomDensityEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = ZsZnNetwork(input_dim, hidden_dim)

    def log_prob(self, xi, s):
        """
        Compute log probability log(p(xi | s)) = log( Z_S(s) * xi + Z_N(s) * (1 - xi) )
        """
        Zs, Zn = self.network(s)  # Get Zs and Zn
        likelihood = xi * Zs + (1 - xi) * Zn  # Compute likelihood function
        return torch.log(likelihood)  # Add small value for numerical stability



# === Generate Training Data Using Swyft Simulator ===
simulator = sim.SineGaussianSim()
num_samples = 10000

real_data_list = []
xi_labels = []

for _ in range(num_samples):
    noise = simulator.noise()
    Amp = simulator.source_prior()
    signal = simulator.signal(Amp)
    xi = simulator.xi_prior()
    
    data = simulator.get_data(noise, signal, xi)
    real_data = simulator.get_real_data(data)
    
    real_data_list.append(real_data)
    xi_labels.append(xi)

# Convert to tensors
real_data_tensor = torch.tensor(np.array(real_data_list), dtype=torch.float32)
xi_tensor = torch.tensor(np.array(xi_labels), dtype=torch.float32).unsqueeze(-1)

# === Define the Prior for xi ===
prior = utils.BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

# Extract number of features dynamically
num_features = real_data_tensor.shape[1]

# Define function that returns the custom density estimator
def density_estimator_fn(theta, x):
    return CustomDensityEstimator(input_dim=num_features)

# === Train the Custom Posterior Model Using SNPE ===
inference = SNPE(prior, density_estimator=density_estimator_fn)
density_estimator = inference.append_simulations(theta=xi_tensor, x=real_data_tensor).train()

#%%
# === Compute Posterior for a New Observation ===
def compute_posterior(observation):
    """
    Compute p(xi | s) using the trained network.
    """
    posterior_samples = density_estimator.sample((1000,), x=torch.tensor(observation, dtype=torch.float32))
    return posterior_samples

# === Example: Compute Posterior for a New Observation ===
obs_noise = simulator.noise()
obs_signal = simulator.signal(15)  # Example with an arbitrary amplitude
obs_data = simulator.get_data(obs_noise, obs_signal, 0.7)  # Example with xi = 0.7
obs_real_data = simulator.get_real_data(obs_data)

post_vals = compute_posterior(obs_real_data)

# === Plot Posterior ===
plt.hist(post_vals.numpy(), bins=50, density=True, alpha=0.75, label="Estimated Posterior")
plt.xlabel(r'$\xi$')
plt.ylabel(r'Posterior $p(\xi|\bar{s})$')
plt.title('Posterior Estimated from Custom Density Estimator')
plt.legend()
plt.show()

# %%
posterior_samples = density_estimator.sample((10000,),torch.tensor(real_data_tensor[0], dtype=torch.float32))

# %%
plt.hist(posterior_samples)
# %%
plt.plot(real_data_tensor[3])


# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import swyft
from sbi import utils as utils
from sbi.inference import SNPE

class ZsZnNetwork(nn.Module):
    """ Neural network that models both Z_S and Z_N. """
    def __init__(self, input_dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.LeakyReLU(0.1),
            nn.Linear(500, 64),
            nn.Linear(64, 2)
        )
        #self.nn[-1].bias.data = torch.tensor([-672.0, -103.0])  # Start near expected logZ
    def forward(self, s):
        log_zs, log_zn = self.nn(s).chunk(2, dim=-1)
        #print(log_zs,log_zn)# Split into two outputs
        return (log_zs), (log_zn)  # Return actual values

# === Define Custom Density Estimator with Loss and Sampling ===
class CustomDensityEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = ZsZnNetwork(input_dim)


    def log_prob(self, xi, s):
        """ Compute log probability in log-space for numerical stability. """
        log_zs, log_zn = self.network(s)  # Get log values

        # 🛠 Compute log-likelihood safely using logsumexp
        log_likelihood = torch.logsumexp(
            torch.stack([
                log_zs + torch.log(xi ),  # Avoid log(0)
                log_zn + torch.log(1 - xi)
            ], dim=0),
            dim=0
        )

        return log_likelihood


    def loss(self, xi, s):
        """ Compute the loss as Negative Log Likelihood (NLL). """
        return -torch.mean(self.log_prob(xi, s))  # Maximize log likelihood

    def sample(self, sample_shape, s):
        """
        Generate posterior samples p(xi | s) using importance sampling.
        """
        with torch.no_grad():
            zs,zn = self.network(s)
            #print(zs,zn)
            xi_values = torch.linspace(0, 1, steps=sample_shape[0])  # Discretized xi

            # Compute posterior
            posterior_values = xi_values * zs + (1 - xi_values)*zn  

            # Normalize (avoid division by zero)
            posterior_values /= posterior_values.sum()
            print(posterior_values.sum())
            # Print for debugging
            print("Sum of posterior_values:", posterior_values.sum())

            # Convert to NumPy before using np.random.choice
            xi_values = xi_values.cpu().numpy()
            posterior_values = posterior_values.cpu().numpy()

            # Draw samples proportional to the posterior values
            return np.random.choice(xi_values, size=10000, p=posterior_values)
#%%
# === Generate Training Data Using Swyft Simulator ===
simulator = sim.SineGaussianSim()
num_samples = 100000

real_data_list = []
xi_labels = []

for _ in range(num_samples):
    noise = simulator.noise()
    Amp = simulator.source_prior()
    signal = simulator.signal(Amp)
    xi = simulator.xi_prior()
    
    data = simulator.get_data(noise, signal, xi)
    real_data = simulator.get_real_data(data)
    
    real_data_list.append(real_data)
    xi_labels.append(xi)

#%%
# Convert to tensors
real_data_tensor = torch.tensor(np.array(real_data_list), dtype=torch.float32)
xi_tensor = torch.tensor(np.array(xi_labels), dtype=torch.float32).unsqueeze(-1)


real_data_mean = real_data_tensor.mean(dim=0, keepdim=True)
real_data_std = real_data_tensor.std(dim=0, keepdim=True)  # Avoid division by zero
real_data_tensor_norm = (real_data_tensor - real_data_mean) / real_data_std

plt.plot(real_data_tensor[2])
plt.plot(real_data_tensor_norm[2])

#%%

#%%
# === Define the Prior for xi ===
prior = utils.BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

# Extract number of features dynamically
num_features = real_data_tensor_norm.shape[1]

# Define function that returns the custom density estimator
def density_estimator_fn(a,b):
    return CustomDensityEstimator(input_dim=num_features)

# === Train the Custom Posterior Model Using SNPE ===
inference = SNPE(prior, density_estimator=density_estimator_fn)
density_estimator = inference.append_simulations(theta=xi_tensor, x=real_data_tensor_norm).train(max_num_epochs= 10,learning_rate=0.0005, show_train_summary=True)

#%%

posterior_samples = density_estimator.sample((100000,),torch.tensor(real_data_tensor[0], dtype=torch.float32))
plt.hist(posterior_samples)
xi_tensor[0]
#%%

def train_model(density_estimator, xi_tensor, real_data_tensor, num_epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(density_estimator.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = density_estimator.loss(xi_tensor, real_data_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:  # Print every 100 epochs
            print(f"Epoch {epoch}: Loss = {loss.item()}")

#%%
train_model(density_estimator, xi_tensor, real_data_tensor, num_epochs=1000, lr=0.001)
#%%
s = torch.tensor(real_data_tensor[50], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
posterior_samples = density_estimator.sample((10000,), s)

#%%
# === Compute Posterior for a New Observation ===
def compute_posterior(observation):
    """
    Compute p(xi | s) using the trained network.
    """
    posterior_samples = density_estimator.sample((1000,), x=torch.tensor(observation, dtype=torch.float32))
    return posterior_samples

# === Example: Compute Posterior for a New Observation ===
obs_noise = sim.noise()
obs_signal = sim.signal(15)  # Example with an arbitrary amplitude
obs_data = sim.get_data(obs_noise, obs_signal,
#%%
simulator.source_prior()

# %%
xi_values = torch.linspace(0, 1, steps=1000)  # Discretized xi
posterior_values = xi_values * 253  + (1 - xi_values) * 100  # Compute posterior
posterior_values /= posterior_values.sum()  # Normalize
            
# Draw samples proportional to the posterior values
sss=xi_values[torch.multinomial(posterior_values, 1000)]


#%%
#
plt.plot(xi_values,posterior_values)  # Compute posterior

# %%
plt.hist(sss)
# %%
samples = np.random.choice(xi_values, size=1000, p=posterior_values /= posterior_values.sum())  # Normalize to make it a probability distribution

# Sample from the posterior distribution
# %%
import numpy as np
import matplotlib.pyplot as plt

# Define xi values
xi_values = np.linspace(0, 1, 10000)  # Discretized xi

# Compute posterior
posterior_values = xi_values * 253 + (1 - xi_values) * 100  
#plt.plot(xi_values, posterior_values , color='red', linewidth=2, label="True Distribution")


posterior_values /= posterior_values.sum()  # Normalize to make it a probability distribution

plt.plot(xi_values, posterior_values , color='red', linewidth=2, label="True Distribution")
#%%
# Sample from the posterior distribution
samples = np.random.choice(xi_values, size=1000000, p=posterior_values)

# Plot the histogram of sampled values
plt.hist(samples, bins=50, density=True, alpha=0.6)

plt.plot(xi_values, posterior_values*10000 , color='red', linewidth=2, label="True Distribution")
plt.xlabel("Sampled xi values")
plt.ylabel("Density")
plt.title("Histogram of Sampled Values")
plt.show()

# %%
posterior_values = xi_values * 253 + (1 - xi_values) * 100  

posterior_values.sum()
# %%
import numpy as np
import matplotlib.pyplot as plt

# Define xi values
N = 1000  # Number of steps
xi_values = np.linspace(0, 1, N)  # Discretized xi
dx = xi_values[1] - xi_values[0]  # Step size (should be 1 / (N - 1))

# Compute posterior
posterior_values = xi_values * 253 + (1 - xi_values) * 100  
posterior_values /= posterior_values.sum()  # Normalize

# Sample from the posterior distribution
samples = np.random.choice(xi_values, size=100000, p=posterior_values)

# Plot histogram of sampled values
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', edgecolor='black', label="Sampled Values")

# Overlay actual posterior distribution (scaled by step size)
plt.plot(xi_values, posterior_values / dx, color='red', linewidth=2, label="True Distribution / dx")
plt.xlabel("Sampled xi values")
plt.ylabel("Density")
plt.title("Histogram of Sampled Values vs True Distribution")
plt.legend()
plt.show()

# %%
xi_values = torch.linspace(0, 1, steps=100)  # Discretized xi

# Compute posterior
posterior_values = xi_values * np.exp(-672.798974294187) + (1 - xi_values) * np.exp(-100)

# Normalize (avoid division by zero)
posterior_values /= posterior_values.sum()
print(posterior_values.sum())
# Print for debugging
print("Sum of posterior_values:", posterior_values.sum())

# Convert to NumPy before using np.random.choice
xi_values = xi_values.cpu().numpy()
posterior_values = posterior_values.cpu().numpy()
plt.hist(np.random.choice(xi_values, size=10000, p=posterior_values))
# %%
 np.exp(-672.798974294187)
# %%
 np.exp(-100)
# %%
