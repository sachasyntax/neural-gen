```
═══════════════════════════════════════════════════════════
NEURAL LATENT AUDIO GENERATOR
Overlap AE + PCA + GMM + Griffin-Lim
═══════════════════════════════════════════════════════════

DEPENDENCIES
────────────
pip install torch torchaudio librosa soundfile scikit-learn scipy numpy

FILES
─────
train.py      — learns the timbral structure of the dataset and saves the model
generate.py   — walks the latent space and exports a WAV file
./model/      — created by train.py

TRAINING
────────
python train.py --data_dir /path/to/audio --output_dir ./model

options:
  --epochs      120   default
  --latent_dim  64    default

input: .wav .mp3 .flac .ogg
all files auto-converted to mono 22050Hz

estimated time on Apple M2 with ~1h of audio:
  120 epochs  →  40–60 min

GENERATION
──────────
python generate.py --duration 60 --output out.wav

parameters:
  --duration          output length in seconds (required)
  --output            WAV path (default output.wav)
  --model_dir         model directory (default ./model)
  --step_size         step size in latent space (default 0.3)
  --smoothing_window  trajectory smoothness (default 5)
  --momentum          movement inertia (default 0.6)
  --jump_prob         probability of jumping to a different region (default 0.03)

PARAMETER LOGIC
───────────────
step_size         small = slow, subtle evolution
                  large = wide, abrupt variation

smoothing_window  high = very gradual transitions
                  low  = sharper changes

momentum          high = movement holds direction for longer
                  low  = direction changes frequently

jump_prob         low  = stays within the same timbral region
                  high = jumps often across different dataset regions

EXAMPLES BY TYPE
────────────────
VARIED NOISE
  python generate.py --duration 60 --output noise.wav \
    --step_size 0.8 --momentum 0.3 --smoothing_window 2 --jump_prob 0.08

STATIC DRONE
  python generate.py --duration 300 --output drone.wav \
    --step_size 0.1 --momentum 0.8 --smoothing_window 10 --jump_prob 0.01

SLOW EVOLUTION
  python generate.py --duration 600 --output slow.wav \
    --step_size 0.2 --momentum 0.7 --smoothing_window 8 --jump_prob 0.02

MAXIMUM EXPLORATION
  python generate.py --duration 120 --output explore.wav \
    --step_size 1.5 --momentum 0.2 --smoothing_window 2 --jump_prob 0.15

ARCHITECTURE
────────────
Audio → Mel Spectrogram (128 mel, 22050Hz)
      → Conv1D Autoencoder with overlap consistency loss
      → PCA (64 → 32 dim)
      → GMM (12 components)
      → Random walk in latent space
      → Decoder → Mel
      → Griffin-Lim → WAV

═══════════════════════════════════════════════════════════
```
