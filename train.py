"""
train.py — AE con overlap consistency loss + spectral loss
Pipeline: Audio → Mel → AE (overlap pairs) → PCA → GMM → save

Usage:
    python train.py --data_dir /path/to/audio --output_dir ./model
    python train.py --data_dir ./audio --output_dir ./model --epochs 150
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SR           = 22050
N_FFT        = 2048
HOP          = 512
N_MELS       = 128
SEQ_FRAMES   = 64
LATENT_DIM   = 64
PCA_DIM      = 32
GMM_COMP     = 12
BATCH_SIZE   = 16
EPOCHS       = 120
LR           = 3e-4
OVERLAP      = SEQ_FRAMES // 2   # 50% overlap between consecutive chunks
CONSISTENCY_W = 2.0              # weight for overlap consistency loss
SPECTRAL_W    = 0.5              # weight for spectral loss

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ─── DATASET ──────────────────────────────────────────────────────────────────
class OverlapPairDataset(Dataset):
    """
    Each item: (chunk_t, chunk_t1) where chunk_t1 starts OVERLAP frames after chunk_t.
    Shared frames: chunk_t[OVERLAP:] == chunk_t1[:OVERLAP]
    """
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_audio(data_dir):
    exts = (".mp3", ".wav", ".flac", ".ogg")
    files = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.lower().endswith(exts)
    ]
    if not files:
        raise ValueError(f"No audio files in {data_dir}")
    print(f"Found {len(files)} files")

    pairs = []
    singles = []

    for path in files:
        print(f"  {os.path.basename(path)}")
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print(f"    skip: {e}")
            continue

        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel + 1e-8, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        T = mel_db.shape[1]
        # extract overlapping pairs: step = OVERLAP (50% overlap)
        for start in range(0, T - SEQ_FRAMES - OVERLAP, OVERLAP):
            c0 = mel_db[:, start:start + SEQ_FRAMES]
            c1 = mel_db[:, start + OVERLAP:start + OVERLAP + SEQ_FRAMES]
            t0 = torch.tensor(c0, dtype=torch.float32)
            t1 = torch.tensor(c1, dtype=torch.float32)
            pairs.append((t0, t1))
            singles.append(t0)

    print(f"Total pairs: {len(pairs)}  shape: {pairs[0][0].shape}")
    return pairs, singles


# ─── ENCODER ──────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, n_mels, seq_frames, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_mels, seq_frames)
            flat = self.conv(dummy).view(1, -1).shape[1]
        self.flat = flat
        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


# ─── DECODER ──────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, flat, latent_dim, n_mels, seq_frames):
        super().__init__()
        self.seq_frames = seq_frames
        self._flat_shape = (512, flat // 512)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, flat),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, n_mels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), *self._flat_shape)
        return self.deconv(h)[:, :, :self.seq_frames]


class AE(nn.Module):
    def __init__(self, n_mels=N_MELS, seq_frames=SEQ_FRAMES, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(n_mels, seq_frames, latent_dim)
        self.decoder = Decoder(
            self.encoder.flat, latent_dim, n_mels, seq_frames
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ─── LOSSES ───────────────────────────────────────────────────────────────────
def spectral_loss(recon, target, n_bands=8):
    band = recon.shape[1] // n_bands
    loss = 0.0
    for i in range(n_bands):
        r = recon[:,  i*band:(i+1)*band, :].mean(dim=(1,2))
        t = target[:, i*band:(i+1)*band, :].mean(dim=(1,2))
        loss = loss + nn.functional.mse_loss(r, t)
    return loss / n_bands


def consistency_loss(z0, z1):
    """
    z0 and z1 encode overlapping chunks — their latents should be similar.
    """
    return nn.functional.mse_loss(z0, z1)


def total_loss(recon0, x0, recon1, x1, z0, z1):
    recon  = nn.functional.mse_loss(recon0, x0) + nn.functional.mse_loss(recon1, x1)
    spec   = spectral_loss(recon0, x0) + spectral_loss(recon1, x1)
    cons   = consistency_loss(z0, z1)
    return recon + SPECTRAL_W * spec + CONSISTENCY_W * cons, recon, spec, cons


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def train(data_dir, output_dir, epochs, latent_dim):
    os.makedirs(output_dir, exist_ok=True)

    pairs, singles = load_audio(data_dir)
    loader = DataLoader(
        OverlapPairDataset(pairs), batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True
    )

    model = AE(N_MELS, SEQ_FRAMES, latent_dim).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    params = sum(p.numel() for p in model.parameters())
    print(f"\nDevice: {DEVICE}  |  Params: {params:,}")
    print(f"Latent: {latent_dim}  PCA→{PCA_DIM}  GMM: {GMM_COMP} components")
    print(f"Consistency weight: {CONSISTENCY_W}  Spectral weight: {SPECTRAL_W}")
    print(f"Training {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        tot = tot_r = tot_s = tot_c = 0
        for x0, x1 in loader:
            x0, x1 = x0.to(DEVICE), x1.to(DEVICE)
            r0, z0 = model(x0)
            r1, z1 = model(x1)
            loss, rl, sl, cl = total_loss(r0, x0, r1, x1, z0, z1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot   += loss.item()
            tot_r += rl.item()
            tot_s += sl.item()
            tot_c += cl.item()
        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            n = len(loader)
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"loss={tot/n:.4f}  "
                  f"recon={tot_r/n:.4f}  "
                  f"spec={tot_s/n:.4f}  "
                  f"cons={tot_c/n:.4f}")

    # ─── LATENT EXTRACTION ────────────────────────────────────────────────────
    print("\nExtracting latent vectors...")
    model.eval()

    class SingleDataset(Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]

    all_z = []
    with torch.no_grad():
        for batch in DataLoader(SingleDataset(singles), batch_size=128):
            z = model.encoder(batch.to(DEVICE))
            all_z.append(z.cpu().numpy())
    Z = np.concatenate(all_z, axis=0)
    print(f"  Latent matrix: {Z.shape}")

    # ─── PCA ──────────────────────────────────────────────────────────────────
    print(f"Fitting PCA {latent_dim} → {PCA_DIM}...")
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)
    pca = PCA(n_components=PCA_DIM, random_state=42)
    Z_pca = pca.fit_transform(Z_scaled)
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # ─── GMM ──────────────────────────────────────────────────────────────────
    print(f"Fitting GMM ({GMM_COMP} components)...")
    gmm = GaussianMixture(
        n_components=GMM_COMP, covariance_type="full",
        max_iter=300, random_state=42,
    )
    gmm.fit(Z_pca)
    print(f"  BIC: {gmm.bic(Z_pca):.2f}")

    # ─── SAVE ─────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(output_dir, "vae.pt"))
    for name, obj in [("gmm.pkl", gmm), ("pca.pkl", pca), ("scaler.pkl", scaler)]:
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(obj, f)
    meta = dict(
        n_mels=N_MELS, seq_frames=SEQ_FRAMES, latent_dim=latent_dim,
        pca_dim=PCA_DIM, gmm_comp=GMM_COMP, sr=SR, n_fft=N_FFT, hop=HOP,
        model_type="ae_overlap",
    )
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved → {output_dir}/")
    print("  vae.pt | gmm.pkl | pca.pkl | scaler.pkl | meta.pkl")


# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", default="./model")
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    args = p.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.latent_dim)