Works with $q$-state Potts spins ($s_i \in {0,1,\dots,q{-}1}$)
Uses a 2D convolutional encoder/decoder
Outputs logits or softmax over $q$ classes
Can reconstruct full lattice images









1. 2D VAE for Potts Model



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Load Potts data
X = np.load("potts_configs.npy")  # shape: (N, L, L) integer spins
T = np.load("temperatures.npy")   # shape: (N,)
q = int(X.max() + 1)              # number of Potts states

# One-hot encode (N, q, L, L)
X_onehot = np.eye(q)[X].transpose(0, 3, 1, 2).astype(np.float32)
dataset = TensorDataset(torch.tensor(X_onehot))
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 2D Convolutional VAE
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=2, q=3):
        super().__init__()
        self.q = q
        # Encoder
        self.enc_conv1 = nn.Conv2d(q, 16, 3, stride=2, padding=1)  # (B,16,L/2,L/2)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1) # (B,32,L/4,L/4)
        self.enc_fc1 = nn.Linear(32*7*7, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 32*7*7)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(16, q, 4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.enc_fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = h.view(-1, 32, 7, 7)
        h = F.relu(self.dec_conv1(h))
        h = self.dec_conv2(h)  # logits for q classes
        return h  # shape: (B, q, L, L)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar

def loss_function(logits, x_true, mu, logvar):
    # Reconstruction loss: cross-entropy
    recon = F.cross_entropy(logits, x_true.argmax(dim=1), reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + KLD

# Initialize model
vae = ConvVAE(latent_dim=2, q=q)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Train
vae.train()
for epoch in range(15):
    total_loss = 0
    for batch in loader:
        x = batch[0]  # shape (B, q, L, L)
        optimizer.zero_grad()
        logits, mu, logvar = vae(x)
        loss = loss_function(logits, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss = {total_loss:.1f}")







2. Visualizing Reconstructions



vae.eval()
with torch.no_grad():
    sample = torch.tensor(X_onehot[:8])  # batch of 8
    logits, mu, _ = vae(sample)
    preds = torch.argmax(logits, dim=1)  # shape (B, L, L)

fig, axes = plt.subplots(8, 2, figsize=(4, 16))
for i in range(8):
    axes[i,0].imshow(np.argmax(sample[i].numpy(), axis=0), cmap='tab20', vmin=0, vmax=q-1)
    axes[i,0].set_title("Original")
    axes[i,1].imshow(preds[i].numpy(), cmap='tab20', vmin=0, vmax=q-1)
    axes[i,1].set_title("Reconstruction")
    axes[i,0].axis('off')
    axes[i,1].axis('off')
plt.tight_layout()
plt.show()







Notes:





Assumes L=28 (if not, adjust decoder shapes).
You can use temperature-dependent coloring in the latent space like before.
The decoder produces logits, which allows clean multiclass cross-entropy.




Let me know if you’d like:



Conditional VAE (e.g. conditioned on temperature)
Animations of reconstructions over temperature
Export to interactive latent space plots
