1. PCA with Scikit-learn (for baseline comparison)



import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assume X is (N_samples, L*L) Ising configurations
# Load Ising data
X = np.load("ising_configurations.npy")  # shape: (N, L*L)
T = np.load("temperatures.npy")  # shape: (N,)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA projection colored by temperature
plt.figure(figsize=(8, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=T, cmap='coolwarm', s=5)
plt.colorbar(sc, label='Temperature')
plt.title("PCA of Ising Configurations")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.show()







2. VAE in PyTorch



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load and preprocess data
X = np.load("ising_configurations.npy").astype(np.float32)
T = np.load("temperatures.npy")  # for coloring

X = (X - X.mean()) / X.std()  # Normalize
dataset = TensorDataset(torch.tensor(X))
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define VAE
class VAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # mean
        self.fc22 = nn.Linear(128, latent_dim)  # logvar
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # use sigmoid for binary data

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training
vae = VAE(input_dim=X.shape[1])
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

vae.train()
for epoch in range(20):
    total_loss = 0
    for batch in loader:
        x = batch[0]
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss = {total_loss:.2f}")

# Inference for visualization
vae.eval()
with torch.no_grad():
    x_tensor = torch.tensor(X)
    _, mu, _ = vae(x_tensor)

mu_np = mu.numpy()

# Plot latent space
plt.figure(figsize=(8, 6))
plt.scatter(mu_np[:, 0], mu_np[:, 1], c=T, cmap='coolwarm', s=5)
plt.colorbar(label="Temperature")
plt.xlabel("Latent dimension 1")
plt.ylabel("Latent dimension 2")
plt.title("VAE latent space")
plt.tight_layout()
plt.show()







Optional: Latent Variance Near $T_c$





To track criticality:

import pandas as pd

df = pd.DataFrame({'T': T, 'z1': mu_np[:,0], 'z2': mu_np[:,1]})
df['T_bin'] = np.round(df['T'], 2)
variance = df.groupby('T_bin')[['z1', 'z2']].var()
variance.plot()
plt.title("Latent variance vs temperature")
plt.xlabel("Temperature")
plt.ylabel("Latent variance")
plt.show()





Let me know if you’d like:



Potts model version
2D VAE decoder to reconstruct images
Diffusion model baseline (as an extension)
