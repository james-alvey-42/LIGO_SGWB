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
"""plt.plot(sample['noise'])
plt.plot(sample['signal'])
plt.plot(sample['data'])
"""
# %%
class Network(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        self.norm = swyft.networks.OnlineStandardizingLayer(shape=(1002,))
        self.xi_logratios = swyft.LogRatioEstimator_1dim(
            num_features=1002, num_params=1, varnames="xi")

    def forward(self, A, B):
        xi_lr = self.xi_logratios(self.norm(A["real_data"]), B["xi"])
        return xi_lr
    
import torch

class Network2(swyft.SwyftModule):
    def __init__(self, settings=None, sim=None,lr = 1e-5):
        super().__init__()

        self.num_params = 1  # Number of parameters to infer
        self.npts = 1002  # Number of frequency points
        self.channels = 1  # Number of input channels
        self.num_feat_param = self.channels

        # Standardization layers for non-log data
        self.nl_channel1_nolog = OnlineStandardizingLayer(shape=(self.npts,))

        # Residual networks for log and non-log data
        self.resnet_no_log = ResidualNetWithChannel(
            channels=self.channels,
            in_features=self.npts,
            out_features=self.num_params,
            hidden_features=64,
            num_blocks=2,
            dropout_probability=0.1,
            use_batch_norm=True,
        )
        self.resnet_log = ResidualNetWithChannel(
            channels=self.channels,
            in_features=self.npts,
            out_features=self.num_params,
            hidden_features=64,
            num_blocks=2,
            dropout_probability=0.1,
            use_batch_norm=True,
        )
        # Log-ratio estimator for multi-dimensional inference
        self.log_ratio_estimator = swyft.LogRatioEstimator_1dim(
            num_features=1 * self.num_params * self.num_feat_param,
            num_params=1,
            varnames="xi",
        )

    def forward(self, A, B):
        # Non-log-transformed data
        no_log_data = A["real_data"]
        log_data = torch.log10(A["real_data"])

        # Normalize data
        norm_channel1_nolog = self.nl_channel1_nolog(no_log_data)
        #norm_channel1_log = self.nl_channel1_nolog(log_data)

        # Ensure correct shape (Batch, Channels=1, Features)
        no_log_full_data = norm_channel1_nolog.unsqueeze(1)
        #log_full_data = norm_channel1_log.unsqueeze(1)

        # Process with residual networks
        no_log_compression = self.resnet_no_log(no_log_full_data)
        #log_compression = self.resnet_log(log_full_data)

        # Flatten output
        s_no_log = no_log_compression.reshape(-1, self.num_params * self.num_feat_param)
        #s_log = log_compression.reshape(-1, self.num_params * self.num_feat_param)
        s_total = s_no_log #torch.cat((s_no_log,s_log))
        # Compute log-ratio for Bayesian inference
        log_ratio = self.log_ratio_estimator(s_total, B["xi"])

        return log_ratio
# %%
dm = data.OnTheFlyDataModule(simulator, Nsims_per_epoch=500000, batch_size=64)
trainer = swyft.SwyftTrainer(accelerator='gpu', enable_progress_bar=True)
network = Network2()
#%%
trainer.fit(network, dm)

"""
# %%
#network1 = Network()

#network1 = Network.load_from_checkpoint('/home/zaldivar/Documents/Androniki/phd/LIGO/LIGO_SGWB/notebooks/lightning_logs/version_12/checkpoints/epoch=11-step=1320.ckpt')
network2 = Network2()
network2 = Network2.load_from_checkpoint('/home/zaldivar/Documents/Androniki/phd/LIGO/LIGO_SGWB/notebooks/lightning_logs/version_39/checkpoints/epoch=12-step=14222.ckpt')

# %%
##1 segment 
observation = simulator.sample(conditions={"xi": np.array([0.8])})
xi=simulator.transform_samples(np.reshape(np.linspace(0.0, 1.0, 100), (-1, 1)))
prior_samples = swyft.Samples(
        xi=simulator.transform_samples(np.reshape(np.linspace(0.0, 1.0, 100), (-1, 1)))
    )

predictions2 = trainer.infer(
network2, observation, prior_samples.get_dataloader(batch_size=2048)
)       
#swyft.plot_posterior(predictions, ["xi[0]"], smooth=2.0);
swyft.plot_posterior(predictions2, ["xi[0]"], smooth=2.0);

# %%
ZS_list_withnorm = []
ZN_list_withnorm = []

for i in range(1):
    Z_S_withnorm = compute_ZS_withnorm(t, observation ['data'], freqs, sigma, bounds)
    Z_N_withnorm = compute_ZN_withnorm(observation ['data'], freqs, sigma)
    ZS_list_withnorm.append(Z_S_withnorm)
    ZN_list_withnorm.append(Z_N_withnorm)

# %%

from cobaya.run import run
# Run PolyChord for multiple observations and combine likelihoods
ZS_lists = ZS_list_withnorm  # List of Z_S for each observation
ZN_lists = ZN_list_withnorm  # List of Z_N for each observation


def combined_likelihood(xi, ZS_lists, ZN_lists):
    log_likelihood = 0.0
    for Z_S, Z_N in zip(ZS_lists, ZN_lists):
        term = xi * Z_S + (1 - xi) * Z_N
        if term > 0:
            log_likelihood += np.log(term)
        else:
            log_likelihood += -np.inf
    return log_likelihood




def wrapped_likelihood(**kwargs):
    return combined_likelihood(kwargs["xi"], ZS_lists, ZN_lists)

info = {
    "likelihood": {
        "xi_likelihood": {
            "external": wrapped_likelihood,
            "input_params": ["xi"],
        }
    },
    "params": {"xi": {"prior": {"min": 0.0, "max": 1.0}}},
    "sampler": {"polychord": {"nlive": 300, "precision_criterion": 1e-5}},
    "output": "polychord_xi_combined_output",
}

updated_info, sampler = run(info)
# %%
posterior_samples = np.loadtxt("polychord_xi_combined_output_polychord_raw/polychord_xi_combined_output_equal_weights.txt")

swyft.plot_posterior(predictions, ["xi[0]"], smooth=2.0);
plt.hist(posterior_samples[:,2],bins='auto',density=True)

swyft.plot_posterior(predictions1, ["xi[0]"], smooth=2.0);
plt.hist(posterior_samples[:,2],bins='auto',density=True)


# %%
##1000 segment 

list_obs=[]

for idx in range(1000):
    observation = simulator.sample(conditions={"xi": np.array([0.1])})
    list_obs.append(observation['data'])
    prior_samples = swyft.Samples(
        xi=simulator.transform_samples(np.reshape(np.linspace(0.0, 1.0, 100), (-1, 1)))
    )
    predictions = trainer.infer(
        network2, observation, prior_samples.get_dataloader(batch_size=2048)
    )
    if idx == 0:
        lrs_total = predictions
        swyft.plot_posterior(predictions, ["xi[0]"], smooth=2.0);
    else:
        lrs_total.logratios += predictions.logratios
        swyft.plot_posterior(predictions, ["xi[0]"], smooth=2.0, fig=plt.gcf());
# %%
plt.axvline(observation["xi"][0], color="red", linestyle="--")
swyft.plot_posterior(lrs_total, ["xi[0]"], smooth=2.0, color='green', fig=plt.gcf());
#plt.xlim(0.0001,1)

#plt.xscale('log')
# %%
ZS_list_withnorm = []
ZN_list_withnorm = []

for i in range(1000):
    Z_S_withnorm = compute_ZS_withnorm(t, list_obs[i], freqs, sigma, bounds)
    Z_N_withnorm = compute_ZN_withnorm(list_obs[i], freqs, sigma)
    ZS_list_withnorm.append(Z_S_withnorm)
    ZN_list_withnorm.append(Z_N_withnorm)

# %%
from cobaya.run import run

# Load Z_S and Z_N lists (these should be precomputed from your simulations)
ZS_list = ZS_list_withnorm
ZN_list = ZN_list_withnorm

# Define likelihood function
def likelihood(xi, ZS_list, ZN_list):
    log_likelihood = 0.0
    for Z_S, Z_N in zip(ZS_list, ZN_list):
        term = xi * Z_S + (1 - xi) * Z_N
        if term > 0:
            log_likelihood += np.log(term)
        else:
            return -np.inf  # Avoid log of zero
    return log_likelihood

# Prior on xi (uniform between 0 and 1)
def prior_xi(xi):
    if 0.0 <= xi <= 1.0:
        return 1.0
    return 0.0

# Define parameter space
parameters_dictionary = {
    "xi": {"prior": {"min": 0.0, "max": 1.0}},
}

# PolyChord settings
polychord_settings = {
    "nlive": 300,
    "precision_criterion": 1e-5,
}

def wrapped_likelihood(**kwargs):
    return likelihood(kwargs["xi"], ZS_list, ZN_list)

# PolyChord configuration
info = {
    "likelihood": {
        "xi_likelihood": {
            "external": wrapped_likelihood,
            "input_params": ["xi"],
        }
    },
    "params": parameters_dictionary,
    "sampler": {"polychord": polychord_settings},
    "output": "polychord_xi_output",
}

# Run PolyChord
updated_info, sampler = run(info)

import numpy as np

# Load the posterior samples
posterior_samples = np.loadtxt("polychord_xi_output_polychord_raw/polychord_xi_output_equal_weights.txt")

# Check the shape of the array
print("Shape of posterior samples:", posterior_samples.shape)

# %%
swyft.plot_posterior(lrs_total, ["xi[0]"], smooth=2.0, color='green',labels=['ξ']);
plt.hist(posterior_samples[:,2],density=True,label='polychord')
plt.axvline(0.8,color='red',label='injected')
#plt.savefig('0.1.png')
plt.legend()
# %%
"""