"""Real ESD (Emotional Speech Dataset) DataLoader
Loads actual audio from ESD directory structure:
  ESD/Emotion Speech Dataset/<speaker_id>/<Emotion>/<speaker>_<utt>.wav

Features extracted per sample (matching SyntheticEmoSpeechDataset format):
  mel:               [T_mel, n_mels]  — mel-spectrogram
  f0:                [T_mel]          — fundamental frequency (Hz)
  energy:            [T_mel]          — RMS energy
  duration:          [T_text]         — uniform (T_mel//T_text per token)
  tts_tokens:        [T_text]         — uniform random placeholder
  bert_input_ids:    [T_text]         — same
  bert_attention_mask: [T_text]       — all ones
  text_mask:         [T_text]         — all True
  emotion_idx:       scalar           — 0=Angry,1=Happy,2=Neutral,3=Sad,4=Surprise
  sub_emotion_idx:   scalar           — 0/1 based on speaker ID parity
  mos_score:         scalar           — proxy from F0 smoothness + expressivity
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Emotion directory name → index
EMOTION_MAP = {
    'Angry':    0,
    'Happy':    1,
    'Neutral':  2,
    'Sad':      3,
    'Surprise': 4,
}

# F0 reference means per emotion (for MOS proxy, matching synthetic conventions)
EMOTION_F0_REF = {0: 200, 1: 180, 2: 120, 3: 90, 4: 220}


def _load_audio(path, sr=16000):
    """Load wav to mono float32 at target sample rate."""
    audio, orig_sr = sf.read(path, dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio


def _extract_features(audio, sr=16000, t_mel=30, n_mels=80, hop_length=256,
                       n_fft=1024, fmin=80, fmax=400):
    """Extract mel, f0, energy → each trimmed/padded to t_mel frames."""
    # ── Mel spectrogram ──────────────────────────────────────────────────
    mel_db = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=50, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel_db, ref=np.max)
    # mel_db shape: [n_mels, T] → transpose to [T, n_mels]
    mel = mel_db.T  # [T, n_mels]
    # Normalize to [0, 1]
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # ── F0 via pyin ───────────────────────────────────────────────────────
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length,
        fill_na=None,
    )
    if f0 is None:
        f0 = np.full(mel.shape[0], 120.0, dtype=np.float32)
    else:
        f0 = np.where(np.isnan(f0), 0.0, f0).astype(np.float32)
        # Interpolate unvoiced regions with mean voiced F0
        voiced = f0[f0 > 0]
        mean_f0 = voiced.mean() if len(voiced) > 0 else 120.0
        f0 = np.where(f0 == 0, mean_f0, f0)

    # ── RMS energy ────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]  # [T]
    rms = rms.astype(np.float32)
    rms_max = rms.max() + 1e-8
    rms = rms / rms_max

    # ── Trim / pad to t_mel frames ────────────────────────────────────────
    def fit_frames(arr, target):
        if len(arr) >= target:
            return arr[:target]
        pad = target - len(arr)
        return np.pad(arr, (0, pad), mode='edge')

    def fit_2d(arr2d, target):
        T = arr2d.shape[0]
        if T >= target:
            return arr2d[:target]
        pad = target - T
        return np.pad(arr2d, ((0, pad), (0, 0)), mode='edge')

    mel = fit_2d(mel, t_mel).astype(np.float32)
    f0  = fit_frames(f0, t_mel).astype(np.float32)
    rms = fit_frames(rms, t_mel).astype(np.float32)

    return mel, f0, rms


def _mos_proxy(f0, emotion_idx):
    """Compute MOS proxy matching the synthetic dataset formula."""
    f0_mean  = float(np.mean(f0))
    f0_smoothness = 1.0 / (1.0 + float(np.mean(np.abs(np.diff(f0)))) / (f0_mean + 1))
    ref_f0   = EMOTION_F0_REF.get(emotion_idx, 120)
    expressivity = abs(f0_mean - 120) / 200.0
    energy_strength = 0.5  # placeholder; real energy normalized to [0,1]
    mos = 1.5 + 1.8 * f0_smoothness + 0.9 * expressivity + 0.8 * energy_strength
    return float(np.clip(mos, 1.0, 5.0))


class ESDDataset(Dataset):
    """Real ESD audio loader with identical output format to SyntheticEmoSpeechDataset.

    Args:
        esd_root:     path to ESD/ extraction root (contains "Emotion Speech Dataset/")
        n_speakers:   number of speakers to include (sorted, e.g. 5 → 0001..0005)
        t_text:       text sequence length (placeholder tokens)
        t_mel:        mel frame length
        n_mels:       mel bins
        vocab_size:   vocabulary for placeholder tokens
        sr:           sample rate
        max_per_emotion: cap per (speaker, emotion) to limit dataset size
    """

    def __init__(self, esd_root, n_speakers=5, t_text=15, t_mel=30, n_mels=80,
                 vocab_size=64, sr=16000, max_per_emotion=70, seed=0):
        assert HAS_LIBROSA, "librosa not installed; run: pip install librosa soundfile"

        self.t_text = t_text
        self.t_mel  = t_mel
        self.n_mels = n_mels

        esd_dir = os.path.join(esd_root, 'Emotion Speech Dataset')
        if not os.path.isdir(esd_dir):
            # Try direct root
            esd_dir = esd_root

        speakers = sorted([
            d for d in os.listdir(esd_dir)
            if os.path.isdir(os.path.join(esd_dir, d)) and d.isdigit()
        ])[:n_speakers]

        rng = np.random.default_rng(seed)

        # ── Collect file paths ────────────────────────────────────────────
        records = []  # list of (wav_path, emotion_idx, speaker_id_int)
        for spk in speakers:
            spk_int = int(spk)
            spk_dir = os.path.join(esd_dir, spk)
            for emo_name, emo_idx in EMOTION_MAP.items():
                emo_dir = os.path.join(spk_dir, emo_name)
                if not os.path.isdir(emo_dir):
                    continue
                wavs = sorted([
                    f for f in os.listdir(emo_dir) if f.endswith('.wav')
                ])
                if max_per_emotion and len(wavs) > max_per_emotion:
                    chosen = rng.choice(len(wavs), max_per_emotion, replace=False)
                    wavs = [wavs[i] for i in sorted(chosen)]
                for w in wavs:
                    records.append((
                        os.path.join(emo_dir, w),
                        emo_idx,
                        spk_int,
                    ))

        print(f"  ESDDataset: {len(records)} files from {len(speakers)} speakers")

        # ── Pre-load and extract features ─────────────────────────────────
        N = len(records)
        mel_arr    = np.zeros((N, t_mel, n_mels), dtype=np.float32)
        f0_arr     = np.zeros((N, t_mel),          dtype=np.float32)
        energy_arr = np.zeros((N, t_mel),          dtype=np.float32)
        emotion_arr    = np.zeros(N, dtype=np.int64)
        sub_emo_arr    = np.zeros(N, dtype=np.int64)
        mos_arr        = np.zeros(N, dtype=np.float32)

        # Text placeholders: uniform random tokens per sample
        frames_per_tok = t_mel // t_text
        duration_arr   = np.full((N, t_text), frames_per_tok, dtype=np.int64)
        tts_tokens_arr = rng.integers(0, vocab_size, size=(N, t_text)).astype(np.int64)

        for i, (wav_path, emo_idx, spk_int) in enumerate(records):
            if i % 200 == 0:
                print(f"    [{i}/{N}] loading audio...")
            try:
                audio = _load_audio(wav_path, sr=sr)
                mel, f0, rms = _extract_features(
                    audio, sr=sr, t_mel=t_mel, n_mels=n_mels,
                )
            except Exception as e:
                print(f"    WARNING: failed to load {wav_path}: {e}")
                mel = np.zeros((t_mel, n_mels), dtype=np.float32)
                f0  = np.full(t_mel, 120.0, dtype=np.float32)
                rms = np.full(t_mel, 0.5, dtype=np.float32)

            mel_arr[i]    = mel
            f0_arr[i]     = f0
            energy_arr[i] = rms
            emotion_arr[i]  = emo_idx
            sub_emo_arr[i]  = int(spk_int % 2)  # parity of speaker ID
            mos_arr[i]      = _mos_proxy(f0, emo_idx)

        # ── Store as tensors ──────────────────────────────────────────────
        self.tts_tokens          = torch.from_numpy(tts_tokens_arr)
        self.bert_input_ids      = torch.from_numpy(tts_tokens_arr.copy())
        self.bert_attention_mask = torch.ones((N, t_text), dtype=torch.long)
        self.text_mask           = torch.ones((N, t_text), dtype=torch.bool)
        self.mel                 = torch.from_numpy(mel_arr)
        self.f0                  = torch.from_numpy(f0_arr)
        self.energy              = torch.from_numpy(energy_arr)
        self.duration            = torch.from_numpy(duration_arr)
        self.emotion_idx         = torch.from_numpy(emotion_arr)
        self.sub_emotion_idx     = torch.from_numpy(sub_emo_arr)
        self.mos_score           = torch.from_numpy(mos_arr)

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


def get_real_dataloaders(hp, esd_root, n_speakers=5, max_per_emotion=70, data_seed=0):
    """Create train/val/test DataLoaders from real ESD audio.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = ESDDataset(
        esd_root        = esd_root,
        n_speakers      = n_speakers,
        t_text          = hp['t_text'],
        t_mel           = hp['t_mel'],
        n_mels          = hp['n_mels'],
        vocab_size       = hp['vocab_size'],
        max_per_emotion = max_per_emotion,
        seed            = data_seed,
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
