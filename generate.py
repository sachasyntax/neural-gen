"""
generate.py — AE overlap + GMM walk + Griffin-Lim → WAV
Usage:
    python generate.py --duration 60 --output out.wav
    python generate.py --duration 120 --step_size 0.3 --smoothing_window 5 --output drone.wav
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import soundfile as sf
import librosa
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ─── AE (mirrors train.py exactly) ───────────────────────────────────────────
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
    def __init__(self, n_mels, seq_frames, latent_dim):
        super().__init__()
        self.encoder = Encoder(n_mels, seq_frames, latent_dim)
        self.decoder = Decoder(self.encoder.flat, latent_dim, n_mels, seq_frames)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ─── GRIFFIN-LIM VOCODER (ottimizzato per noise/texture) ─────────────────────
def mel_to_wav(mel_norm, sr, n_fft, hop, n_mels):
    """
    mel_norm: normalized mel [N_MELS, T] in roughly [-3, 3] range
    Denormalize → db → power → Griffin-Lim con parametri noise-ottimizzati
    """
    # denormalize: scale back to db range [-80, 0]
    mel_db = mel_norm * 20.0 - 40.0
    mel_db = np.clip(mel_db, -80.0, 0.0)
    mel_power = librosa.db_to_power(mel_db)
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_iter=128,
    )
    return wav


# ─── TEMPORAL SMOOTHING ───────────────────────────────────────────────────────
def smooth_trajectory(traj, window=5):
    return uniform_filter1d(traj, size=window, axis=0)


# ─── LATENT WALK ─────────────────────────────────────────────────────────────
def latent_walk_pca(gmm, n_steps, step_size=0.3, momentum=0.6, jump_prob=0.03):
    dim  = gmm.means_.shape[1]
    traj = []
    z, _ = gmm.sample(1)
    z    = z[0]
    vel  = np.zeros(dim)

    for _ in range(n_steps):
        # cluster jump — forza esplorazione di regioni diverse
        if np.random.rand() < jump_prob:
            comp   = np.random.choice(gmm.n_components, p=gmm.weights_)
            target = gmm.means_[comp] + np.random.multivariate_normal(
                np.zeros(dim), 0.3 * np.diag(np.diag(gmm.covariances_[comp]))
            )
            vel = 0.3 * (target - z)

        probs    = gmm.predict_proba(z.reshape(1, -1))[0]
        dominant = np.argmax(probs)
        # usa solo diagonale della covarianza per stabilità numerica
        cov_diag = np.diag(np.diag(gmm.covariances_[dominant]))
        noise    = np.random.multivariate_normal(np.zeros(dim), step_size * cov_diag)
        vel      = momentum * vel + (1 - momentum) * noise
        z        = z + vel
        traj.append(z.copy())

    return np.array(traj)


# ─── CROSSFADE ────────────────────────────────────────────────────────────────
def crossfade(a, b, fade=4096):
    fade = min(fade, len(a) // 2, len(b) // 2)
    a[-fade:] *= np.linspace(1, 0, fade)
    b[:fade]  *= np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:] + b[:fade], b[fade:]])


# ─── GENERATE ─────────────────────────────────────────────────────────────────
def generate(model_dir, duration_sec, output_path,
             step_size, smoothing_window, momentum, jump_prob):
    with open(os.path.join(model_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    n_mels     = meta["n_mels"]
    seq_frames = meta["seq_frames"]
    latent_dim = meta["latent_dim"]
    sr         = meta["sr"]
    n_fft      = meta["n_fft"]
    hop        = meta["hop"]

    chunk_sec = seq_frames * hop / sr
    n_chunks  = max(1, int(np.ceil(duration_sec / chunk_sec)))
    print(f"Duration: {duration_sec}s  →  {n_chunks} chunks × {chunk_sec:.3f}s")

    model = AE(n_mels, seq_frames, latent_dim).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "vae.pt"), map_location=DEVICE)
    )
    model.eval()

    with open(os.path.join(model_dir, "gmm.pkl"), "rb") as f:
        gmm = pickle.load(f)
    with open(os.path.join(model_dir, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print("Walking latent space...")
    traj_pca = latent_walk_pca(gmm, n_chunks, step_size=step_size,
                               momentum=momentum, jump_prob=jump_prob)

    print(f"Smoothing (window={smoothing_window})...")
    traj_pca = smooth_trajectory(traj_pca, window=smoothing_window)

    traj_z = scaler.inverse_transform(pca.inverse_transform(traj_pca))

    # log latent variance for diagnostics
    print(f"Latent walk std: {traj_z.std(axis=0).mean():.4f}  "
          f"range: {traj_z.max()-traj_z.min():.4f}")

    print("Decoding chunks...")
    audio_chunks = []
    for i, z in enumerate(traj_z):
        z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mel = model.decoder(z_t).squeeze(0).cpu().numpy()
        wav = mel_to_wav(mel, sr, n_fft, hop, n_mels)
        audio_chunks.append(wav)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1}/{n_chunks}")

    print("Stitching...")
    result = audio_chunks[0]
    for chunk in audio_chunks[1:]:
        result = crossfade(result, chunk)

    target = int(duration_sec * sr)
    result = result[:target]

    peak = np.abs(result).max()
    if peak > 1e-8:
        result = result / peak * 0.95

    sf.write(output_path, result, sr)
    print(f"\n→ {output_path}  ({duration_sec}s @ {sr}Hz)")


# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",        default="./model")
    p.add_argument("--duration",         type=float, required=True)
    p.add_argument("--output",           default="output.wav")
    p.add_argument("--step_size",        type=float, default=0.3,
                   help="Walk step size (default 0.3)")
    p.add_argument("--smoothing_window", type=int,   default=5,
                   help="Smoothing window (default 5)")
    p.add_argument("--momentum",         type=float, default=0.6,
                   help="Walk momentum (default 0.6)")
    p.add_argument("--jump_prob",        type=float, default=0.03,
                   help="Cluster jump probability (default 0.03)")
    args = p.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Model dir not found: {args.model_dir}")
        sys.exit(1)

    generate(args.model_dir, args.duration, args.output,
             args.step_size, args.smoothing_window,
             args.momentum, args.jump_prob)
