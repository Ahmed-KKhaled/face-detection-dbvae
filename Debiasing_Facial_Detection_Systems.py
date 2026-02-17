import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Basic Conv Block
# --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


# --------------------
# Encoder (same as training)
# --------------------
def make_standard_classifier(in_channels=3, n_filters=12, latent_dim=100):

    H, W = 64, 64  # ⚠️ حط نفس حجم الصورة اللي اتدربت عليه

    model = nn.Sequential(
        ConvBlock(in_channels, n_filters, 5, 2, 2),
        ConvBlock(n_filters, 2*n_filters, 5, 2, 2),
        ConvBlock(2*n_filters, 4*n_filters, 3, 2, 1),
        ConvBlock(4*n_filters, 6*n_filters, 3, 2, 1),
        nn.Flatten(),
        nn.Linear(H // 16 * W // 16 * 6*n_filters, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 2 * latent_dim + 1)
    )

    return model


# --------------------
# Sampling
# --------------------
def sampling(z_mean, z_logsigma):
    eps = torch.randn_like(z_mean)
    sigma = torch.exp(0.5 * z_logsigma)
    return z_mean + sigma * eps


# --------------------
# Decoder
# --------------------
class FaceDecoder(nn.Module):
    def __init__(self, latent_dim=100, n_filters=12):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 6 * n_filters),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(6 * n_filters, 4 * n_filters, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * n_filters, n_filters, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters, 3, 5, 2, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 72, 4, 4)
        return self.deconv(x)


# --------------------
# DB-VAE Model
# --------------------
class DB_VAE(nn.Module):

    def __init__(self, latent_dim=100):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = make_standard_classifier(latent_dim=latent_dim)
        self.decoder = FaceDecoder(latent_dim=latent_dim)

    def encode(self, x):
        encoder_output = self.encoder(x)
        y_logits = encoder_output[:, 0].unsqueeze(-1)
        z_mean = encoder_output[:, 1:self.latent_dim+1]
        z_logsigma = encoder_output[:, self.latent_dim+1:]
        return y_logits, z_mean, z_logsigma

    def forward(self, x):
        y_logits, z_mean, z_logsigma = self.encode(x)
        z = sampling(z_mean, z_logsigma)
        recon = self.decoder(z)
        return y_logits, z_mean, z_logsigma, recon

    def predict(self, x):
        y_logits, _, _ = self.encode(x)
        return y_logits
