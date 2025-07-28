import torch.nn as nn
import torch
import pandas as pd
from utils import xavier_init, he_init

# Define Poisson class with reparameterizable sampling
class Poisson:
    def __init__(self, rate, t=0.0):
        """
        Args:
            rate (Tensor): The Poisson rate parameter (lambda), shape [batch_size, feature_dim]
            t (float): Temperature for soft sampling (0 = Regular Poisson)
        """
        # Clamp rate to avoid instability or infinities
        self.rate = rate.clamp(min=1e-10, max=1e7)
        self.log_rate = torch.log(self.rate)
        self.t = t

    def rsample(self, hard: bool = False):
        """
        Sample from the Poisson distribution using exponential inter-arrival times.
        Args:
            hard (bool): If True, use hard thresholding (no temperature)
        Returns:
            Tensor: Sampled Poisson counts of shape [batch_size, feature_dim]
        """
        # Compute dynamic number of trials per sample
        n_trials_per_sample = (torch.ceil(self.rate * 5)).clamp(min=1)

        all_samples = []

        # Iterate over each sample in the batch instead of the entire batch to only generate the nessecary exp(1) samples
        for i in range(self.rate.size(0)):
            # Limit number of trials for computational efficiency
            n_trials = int(min(n_trials_per_sample[i].max().item(), 1e5))
            rate = self.rate[i]  # Rate vector for the i'th sample

            # Sample exponential inter-arrival times: [n_trials, feature_dim]
            x = torch.distributions.Exponential(rate).rsample((n_trials,))
            times = torch.cumsum(x, dim=0)  # Cumulative event times
            indicator = times < 1.0  # Events that occur before time=1

            # Soft Poisson with temperature
            if not (hard or self.t == 0):
                indicator = torch.sigmoid((1.0 - times) / self.t)

            # Sum up occurrences per feature
            all_samples.append(indicator.sum(0).float())

        # Return samples in tensor form: [batch_size, feature_dim]
        return torch.stack(all_samples)

    def kl(self, prior):
        """
        Compute KL divergence between the current Poisson distribution and a prior Poisson distribution.
        Args:
            prior (Tensor): Rate of the prior Poisson distribution
        Returns:
            Tensor: KL divergence per dimension
        """
        r = prior.clamp(1e-5, 1e5) # lower- and upper bound on the prior rate matrix
        KL = self.rate * torch.log(self.rate / r) - self.rate + r
        return KL


# Define the VAE-MS (Variational Autoencoder for Mutational Signatures) model
class VAEMS(nn.Module):
    def __init__(self, input_dim, l_dim, h_dim, activation, T=0.0, start_sigs=None):
        """
        Args:
            input_dim (int): Dimensionality of input data
            l_dim (list[int]): List with 2 integer elements for encoder hidden layers
            h_dim (int): Dimension of latent Poisson rate vector
            activation (str): Name of activation function (e.g., 'ReLU')
            T (float): Temperature for Poisson reparameterization
            start_sigs (pd.DataFrame or ndarray): Optional initial decoder weight matrix
        """
        super(VAEMS, self).__init__()
        self.t = T
        self.activation = getattr(nn, activation)()  

        # Encoder network
        self.enc1 = nn.Linear(input_dim, l_dim[0])
        self.BN1 = nn.BatchNorm1d(l_dim[0])
        self.enc2 = nn.Linear(l_dim[0], l_dim[1])
        self.BN2 = nn.BatchNorm1d(l_dim[1])
        self.enc3 = nn.Linear(l_dim[1], h_dim)
        self.BN3 = nn.BatchNorm1d(h_dim)
        self.inp_lamb = nn.Linear(h_dim, h_dim)  # Output lambda (Poisson rate)

        # Decoder weights: initialized from signature data or random
        if start_sigs is None:
            self.dec_weight = nn.Parameter(torch.rand(h_dim, input_dim))  # Random init
        else:
            self.dec_weight = nn.Parameter(
                torch.tensor(
                    start_sigs.values if isinstance(start_sigs, pd.DataFrame) else start_sigs,
                    dtype=torch.float32
                )
            )

        # Normalize decoder weights to sum to 1 per row
        diag = self.dec_weight.sum(dim=1, keepdim=True)
        self.dec_weight_scaled = nn.Parameter(self.dec_weight / diag)

        # weight initialization depending on activation function
        if activation in ("relu", "elu", "silu"):
            self.apply(he_init)
        if activation in ("gelu", "tanh", "softplus"):
            self.apply(xavier_init)

    def encode(self, x):
        """
        Encode input into Poisson rate parameters.
        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim]
        Returns:
            Tensor: Lambda (rate) vector of shape [batch_size, h_dim]
        """
        x = self.activation(self.BN1(self.enc1(x)))
        x = self.activation(self.BN2(self.enc2(x)))
        x = self.activation(self.BN3(self.enc3(x)))
        return self.inp_lamb(x)

    def forward(self, data):
        """
        Forward pass of the VAE.
        Args:
            data (tuple): Tuple with input data as the first element
        Returns:
            Tuple:
                - Reconstruction (Tensor): [batch_size, input_dim]
                - Latent sample (Tensor): [batch_size, h_dim]
                - Poisson distribution object
                - Decoder matrix (Tensor): [h_dim, input_dim]
        """
        validation = not torch.is_grad_enabled()  # True during validation/inference

        # Encode input and constrain lambda to be non-negative
        lamb = self.encode(data[0]).abs()

        # Normalize decoder weights
        diag = torch.abs(self.dec_weight_scaled.sum(dim=1)).detach()
        decode_matrix = torch.abs(self.dec_weight_scaled) / diag.reshape(-1, 1)

        # Rescale lambda by decoder row sums
        lamb = lamb * diag

        # Sample from reparameterized Poisson
        Poisson_dist = Poisson(lamb, self.t)
        h_reparametrized = Poisson_dist.rsample(hard=validation)

        # Decode from latent sample to input space
        x = torch.matmul(h_reparametrized.type(torch.float32), decode_matrix)

        return x, h_reparametrized, Poisson_dist, decode_matrix
