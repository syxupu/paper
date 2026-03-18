"""Synthetic Emotional Speech Dataset
Mimics EmoV-DB statistical properties for prosody control experiments.
No real audio data required — features generated from emotion-specific
statistical models of F0, energy, and spectral shape.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ── Emotion-specific prosodic parameters ────────────────────────────────────
EMOTION_PARAMS = [
    {'name': 'neutral',   'f0_mean': 120, 'f0_std': 15, 'e_mean': 0.50, 'e_std': 0.10},
    {'name': 'happy',     'f0_mean': 180, 'f0_std': 30, 'e_mean': 0.70, 'e_std': 0.15},
    {'name': 'angry',     'f0_mean': 200, 'f0_std': 50, 'e_mean': 0.90, 'e_std': 0.20},
    {'name': 'sad',       'f0_mean':  90, 'f0_std': 10, 'e_mean': 0.30, 'e_std': 0.08},
    {'name': 'fearful',   'f0_mean': 160, 'f0_std': 40, 'e_mean': 0.60, 'e_std': 0.18},
    {'name': 'surprised', 'f0_mean': 220, 'f0_std': 60, 'e_mean': 0.75, 'e_std': 0.20},
]

# Sub-emotion deltas: (f0_delta, energy_delta) for 2 sub-emotions per emotion.
# These encode fine-grained prosodic variation within each coarse category.
# BERT-conditioned models can learn this from text tokens; one-hot cannot.
SUB_DELTAS = [
    [(0, 0.00), (+25, +0.12)],     # neutral: formal vs slightly engaged
    [(0, 0.00), (+35, +0.15)],     # happy: content vs excited
    [(-40, -0.05), (+45, +0.05)],  # angry: cold anger vs hot anger
    [(0, 0.00), (-25, -0.10)],     # sad: melancholic vs depressed
    [(0, 0.00), (+55, +0.15)],     # fearful: anxiety vs panic
    [(-35, -0.05), (+60, +0.18)],  # surprised: mild vs shock
]


class SyntheticEmoSpeechDataset(Dataset):
    """Generates synthetic emotional speech features.

    Each sample contains:
      mel:               [T_mel, n_mels]  — mel-spectrogram proxy
      f0:                [T_mel]          — fundamental frequency (Hz)
      energy:            [T_mel]          — RMS energy
      duration:          [T_text]         — frame counts per text token
      tts_tokens:        [T_text]         — token IDs for TTS text encoder
      bert_input_ids:    [T_text]         — same tokens for BERT surrogate
      bert_attention_mask: [T_text]       — all ones (no padding)
      text_mask:         [T_text]         — boolean mask (all True)
      emotion_idx:       scalar           — coarse emotion label (0-5)
      sub_emotion_idx:   scalar           — fine-grained sub-emotion (0-1)
      mos_score:         scalar           — naturalness proxy (1-5)
    """

    def __init__(self, n_samples=4000, t_text=15, t_mel=30, n_mels=80,
                 vocab_size=64, n_emotions=6, seed=0):
        rng = np.random.default_rng(seed)
        N = n_samples
        T_text, T_mel, M = t_text, t_mel, n_mels

        self.t_text = t_text
        self.t_mel  = t_mel

        # ── Sample emotion / sub-emotion labels ─────────────────────────────
        emotions     = rng.integers(0, n_emotions, size=N)
        sub_emotions = rng.integers(0, 2,          size=N)

        # ── Text tokens — sub-emotion-specific distributions ─────────────────
        # Sub-emotion identity is encoded in the token distribution so that a
        # transformer encoder can learn to separate sub-emotions.
        # Token layout: [emotion_header(1), sub_header(1), content_tokens(13)]
        # Content tokens drawn from sub-emotion-specific range within vocab.
        tts_tokens    = np.zeros((N, T_text), dtype=np.int64)
        bert_input_ids = np.zeros((N, T_text), dtype=np.int64)
        for i in range(N):
            e, s = int(emotions[i]), int(sub_emotions[i])
            # Header tokens: unique per (emotion, sub-emotion) pair
            header = [e, 6 + e * 2 + s]                        # tokens 0-17
            # Content: sub-emotion-specific token range (width 8, offset by 4)
            base = 18 + e * 8 + s * 4                          # range starts
            base = base % (vocab_size - 8)
            content = rng.integers(base, base + 8, size=T_text - 2)
            tokens = np.array(header + list(content), dtype=np.int64)
            tts_tokens[i]     = tokens
            bert_input_ids[i] = tokens

        # ── Prosodic features ─────────────────────────────────────────────────
        f0_vals    = np.zeros((N, T_mel),   dtype=np.float32)
        energy_vals = np.zeros((N, T_mel),  dtype=np.float32)
        mel_specs  = np.zeros((N, T_mel, M), dtype=np.float32)
        mos_scores = np.zeros(N,             dtype=np.float32)

        t_axis = np.linspace(0, 1, T_mel)

        for i in range(N):
            e, s = int(emotions[i]), int(sub_emotions[i])
            p  = EMOTION_PARAMS[e]
            df, de = SUB_DELTAS[e][s]

            f0_mean   = p['f0_mean'] + df
            f0_std    = p['f0_std']
            e_mean    = p['e_mean'] + de
            e_std     = p['e_std']

            # ── F0 contour with emotion-specific shape ────────────────────
            if e in (1, 2, 5):          # rising-falling (excited/angry/surprised)
                shape = np.sin(np.pi * t_axis)
            elif e == 3:                # falling (sad)
                shape = 1.0 - t_axis
            elif e == 4:                # rising (fearful)
                shape = t_axis
            else:                       # neutral: slight arch
                shape = 1.0 - 2 * (t_axis - 0.5) ** 2

            f0 = (f0_mean
                  + f0_std * shape * rng.normal(0, 0.5, T_mel)
                  + rng.normal(0, 3, T_mel))
            f0 = np.clip(f0, 60, 400).astype(np.float32)
            f0_vals[i] = f0

            # ── Energy contour ─────────────────────────────────────────────
            energy = (e_mean + e_std * rng.normal(0, 1, T_mel))
            energy = np.clip(energy, 0.0, 1.0).astype(np.float32)
            energy_vals[i] = energy

            # ── Mel spectrogram: harmonic + formant synthesis ─────────────
            mel = np.zeros((T_mel, M), dtype=np.float32)
            for t_idx in range(T_mel):
                f  = f0[t_idx]
                en = energy[t_idx]
                # Harmonic series
                for h in range(1, 7):
                    bin_c = int(np.clip(h * f / 400.0 * M, 0, M - 1))
                    lo, hi = max(0, bin_c - 2), min(M, bin_c + 3)
                    mel[t_idx, lo:hi] += en / (h ** 1.2)
                # Emotion-specific formant emphasis
                f1_bin = int(np.clip(M * (0.20 + e * 0.06), 0, M - 1))
                f2_bin = int(np.clip(M * (0.45 + e * 0.04), 0, M - 1))
                for fb in [f1_bin, f2_bin]:
                    lo, hi = max(0, fb - 4), min(M, fb + 5)
                    mel[t_idx, lo:hi] += en * 0.4
            # Normalize per frame then add observation noise
            mx = mel.max(axis=1, keepdims=True) + 1e-8
            mel = mel / mx
            mel += rng.normal(0, 0.015, (T_mel, M)).astype(np.float32)
            mel_specs[i] = mel

            # ── MOS naturalness proxy ─────────────────────────────────────
            # Higher MOS for: smooth F0, high emotional expressivity
            f0_smoothness   = 1.0 / (1.0 + np.mean(np.abs(np.diff(f0))) / (f0_mean + 1))
            expressivity    = np.abs(f0_mean - 120) / 200.0   # deviation from neutral
            energy_strength = float(np.mean(energy))
            mos = 1.5 + 1.8 * f0_smoothness + 0.9 * expressivity + 0.8 * energy_strength
            mos_scores[i] = np.clip(mos, 1.0, 5.0).astype(np.float32)

        # ── Duration: 2 frames per token (exactly sums to T_mel) ────────────
        assert T_mel % T_text == 0, "T_mel must be divisible by T_text"
        frames_per_token = T_mel // T_text
        durations = np.full((N, T_text), frames_per_token, dtype=np.int64)

        # ── Store as tensors ─────────────────────────────────────────────────
        self.tts_tokens          = torch.from_numpy(tts_tokens)
        self.bert_input_ids      = torch.from_numpy(bert_input_ids)
        self.bert_attention_mask = torch.ones((N, T_text), dtype=torch.long)
        self.text_mask           = torch.ones((N, T_text), dtype=torch.bool)
        self.mel                 = torch.from_numpy(mel_specs)
        self.f0                  = torch.from_numpy(f0_vals)
        self.energy              = torch.from_numpy(energy_vals)
        self.duration            = torch.from_numpy(durations)
        self.emotion_idx         = torch.from_numpy(emotions.astype(np.int64))
        self.sub_emotion_idx     = torch.from_numpy(sub_emotions.astype(np.int64))
        self.mos_score           = torch.from_numpy(mos_scores)

    def __len__(self):
        return len(self.tts_tokens)

    def __getitem__(self, idx):
        return {
            'tts_tokens':           self.tts_tokens[idx],
            'bert_input_ids':       self.bert_input_ids[idx],
            'bert_attention_mask':  self.bert_attention_mask[idx],
            'text_mask':            self.text_mask[idx],
            'mel':                  self.mel[idx],
            'f0':                   self.f0[idx],
            'energy':               self.energy[idx],
            'duration':             self.duration[idx],
            'emotion_idx':          self.emotion_idx[idx],
            'sub_emotion_idx':      self.sub_emotion_idx[idx],
            'mos_score':            self.mos_score[idx],
        }


def get_dataloaders(hp, data_seed=0):
    """Create train/val/test DataLoaders from synthetic dataset."""
    dataset = SyntheticEmoSpeechDataset(
        n_samples  = hp['n_samples'],
        t_text     = hp['t_text'],
        t_mel      = hp['t_mel'],
        n_mels     = hp['n_mels'],
        vocab_size = hp['vocab_size'],
        n_emotions = hp['n_emotions'],
        seed       = data_seed,
    )
    N       = len(dataset)
    n_train = int(N * hp['train_frac'])
    n_val   = int(N * hp['val_frac'])
    n_test  = N - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(data_seed),
    )
    bs = hp['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)
    return train_loader, val_loader, test_loader
