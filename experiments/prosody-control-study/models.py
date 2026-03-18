"""Model classes for prosody control comparative study.

Conditions implemented:
  H1 sweep  — BetaVAECondition (β=1,2,4,8), BetaVAENoKLCondition (β=0)
  H2        — BERTConditionedProsody (frozen BERT + 2-layer MLP)
  H2 ablat  — BERTFrozenDirectCondition (frozen BERT + single linear)
  H2 base   — FastSpeech2OneHotCondition (6-class emotion embedding)
  H3        — AcousticOracle (conv extractor + linear MOS probe)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def length_regulate(h, durations):
    """Expand text-frame features to mel-frame resolution.

    Args:
        h:         [B, T_text, d_model]
        durations: [B, T_text]  integer frame counts per token
    Returns:       [B, T_mel,  d_model]
    """
    B, T_text, d_model = h.shape
    outputs = []
    for b in range(B):
        dur_b    = durations[b].long().clamp(min=0)             # [T_text]
        expanded = torch.repeat_interleave(h[b], dur_b, dim=0)  # [T_mel_b, d_model]
        outputs.append(expanded)
    max_T  = max(o.size(0) for o in outputs)
    padded = torch.zeros(B, max_T, d_model, device=h.device)
    for b, o in enumerate(outputs):
        padded[b, :o.size(0)] = o
    return padded  # [B, T_mel, d_model]


def compute_mig_score(model, loader, device):
    """Mutual Information Gap — measures β-VAE disentanglement quality.

    MIG = (MI_top1 - MI_top2) / H(emotion)
    Higher MIG → latent z has one dedicated dimension per factor.
    """
    from sklearn.feature_selection import mutual_info_classif

    all_z, all_emotion = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            out = model(batch)
            if 'z' not in out:
                return 0.0
            all_z.append(out['z'].cpu())
            all_emotion.append(batch['emotion_idx'].cpu())

    if not all_z:
        return 0.0

    Z = torch.cat(all_z).numpy()                  # [N, z_dim]
    E = torch.cat(all_emotion).numpy().astype(int) # [N]

    try:
        mi_vec    = mutual_info_classif(Z, E, discrete_features=False, random_state=42)
        sorted_mi = np.sort(mi_vec)[::-1]
        counts    = np.bincount(E, minlength=6)
        probs     = counts / len(E)
        h_e       = -np.sum(probs * np.log(probs + 1e-10))
        if len(sorted_mi) >= 2:
            mig = float((sorted_mi[0] - sorted_mi[1]) / max(h_e, 1e-6))
        else:
            mig = 0.0
        return max(mig, 0.0)
    except Exception:
        return 0.0


# ════════════════════════════════════════════════════════════════════════════
# SHARED BACKBONE COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """2-layer Transformer encoder over token embeddings."""

    def __init__(self, vocab_size, d_model, n_layers, nhead):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.0, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, tokens, mask=None):
        """tokens: [B, T_text] → [B, T_text, d_model]"""
        x = self.embedding(tokens)   # [B, T, d_model]
        x = self.encoder(x)          # [B, T, d_model]
        return x


class VarianceAdaptor(nn.Module):
    """Predicts F0, energy, duration given text encoding + conditioning vector."""

    def __init__(self, d_model, cond_dim):
        super().__init__()
        in_dim = d_model + cond_dim
        self.proj        = nn.Linear(in_dim, d_model)
        self.f0_head     = nn.Linear(d_model, 1)
        self.energy_head = nn.Linear(d_model, 1)
        self.dur_head    = nn.Linear(d_model, 1)

    def forward(self, h_text, cond):
        """
        h_text: [B, T_text, d_model]
        cond:   [B, cond_dim]
        Returns: f0_pred, energy_pred, dur_pred  each [B, T_text]
        """
        B, T, _ = h_text.shape
        cond_exp = cond.unsqueeze(1).expand(-1, T, -1)          # [B, T, cond_dim]
        h = F.relu(self.proj(torch.cat([h_text, cond_exp], -1))) # [B, T, d_model]
        f0_pred     = self.f0_head(h).squeeze(-1)               # [B, T_text]
        energy_pred = self.energy_head(h).squeeze(-1)           # [B, T_text]
        dur_pred    = F.softplus(self.dur_head(h)).squeeze(-1)  # [B, T_text] positive
        return f0_pred, energy_pred, dur_pred


class MelDecoder(nn.Module):
    """2-layer Transformer decoder: [B, T_mel, d_model] → [B, T_mel, n_mels]."""

    def __init__(self, d_model, n_mels, n_layers, nhead):
        super().__init__()
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.0, norm_first=True,
        )
        self.decoder  = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_model, n_mels)

    def forward(self, h):
        """h: [B, T_mel, d_model] → [B, T_mel, n_mels]"""
        return self.proj_out(self.decoder(h))


class ProsodyEncoder(nn.Module):
    """Convolutional encoder: mel → (mu, logvar) in prosody latent space."""

    def __init__(self, n_mels, z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128,    64,  kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)

    def forward(self, mel):
        """mel: [B, T_mel, n_mels] → mu, logvar: [B, z_dim]"""
        x = mel.transpose(1, 2)   # [B, n_mels, T_mel]
        x = self.conv(x)          # [B, 64, T_mel]
        x = x.mean(dim=2)         # [B, 64] — global average pooling
        return self.fc_mu(x), self.fc_logvar(x)


# ════════════════════════════════════════════════════════════════════════════
# CONDITION 1-4: β-VAE PROSODY ENCODER
# ════════════════════════════════════════════════════════════════════════════

class BetaVAECondition(nn.Module):
    """β-VAE prosody conditioning.

    Loss = mel_mse + 0.1*(f0_loss + energy_loss) + β * KL(q||N(0,I))

    Higher β → more disentanglement pressure → worse reconstruction (H1).
    """

    def __init__(self, hp, beta=1.0):
        super().__init__()
        self.hp   = hp
        self.beta = beta

        self.prosody_encoder = ProsodyEncoder(hp['n_mels'], hp['z_dim'])
        self.text_encoder    = TextEncoder(
            vocab_size=hp['vocab_size'], d_model=hp['d_model'],
            n_layers=hp['n_text_encoder_layers'], nhead=hp['nhead'],
        )
        # Project z to conditioning dimension
        self.z_proj = nn.Linear(hp['z_dim'], hp['cond_dim'])
        self.variance_adaptor = VarianceAdaptor(hp['d_model'], hp['cond_dim'])
        self.mel_decoder = MelDecoder(
            hp['d_model'], hp['n_mels'],
            hp['n_mel_decoder_layers'], hp['nhead'],
        )
        # Broadcast z into mel-decoder input space
        self.z_broadcast = nn.Linear(hp['z_dim'], hp['d_model'])

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, batch):
        tokens   = batch['tts_tokens']
        mel_tgt  = batch['mel']
        durations = batch['duration']

        # Encode prosody from teacher-forced target mel
        mu, logvar = self.prosody_encoder(mel_tgt)
        z          = self.reparameterize(mu, logvar)         # [B, z_dim]

        # Text encoding
        h_text = self.text_encoder(tokens)                   # [B, T_text, d_model]

        # Conditioning from z
        cond = self.z_proj(z)                                # [B, cond_dim]
        f0_pred, energy_pred, dur_pred = self.variance_adaptor(h_text, cond)

        # Mel decoding with z broadcast
        h_exp = length_regulate(h_text, durations)           # [B, T_mel, d_model]
        z_emb = self.z_broadcast(z).unsqueeze(1).expand(-1, h_exp.size(1), -1)
        mel_pred = self.mel_decoder(h_exp + z_emb)           # [B, T_mel, n_mels]

        return {
            'mel_pred':    mel_pred,
            'f0_pred':     f0_pred,
            'energy_pred': energy_pred,
            'dur_pred':    dur_pred,
            'mu':          mu,
            'logvar':      logvar,
            'z':           z.detach(),          # for MIG; no gradient
        }

    def compute_loss(self, outputs, batch):
        mel_tgt = batch['mel']
        mel_loss = F.mse_loss(outputs['mel_pred'], mel_tgt)

        # Down-sample target prosody to T_text for supervision
        T_text = outputs['f0_pred'].size(1)
        f0_ds  = F.adaptive_avg_pool1d(
            batch['f0'].unsqueeze(1), T_text).squeeze(1)
        en_ds  = F.adaptive_avg_pool1d(
            batch['energy'].unsqueeze(1), T_text).squeeze(1)
        f0_loss     = F.mse_loss(outputs['f0_pred'],     f0_ds  / 400.0)  # normalize Hz
        energy_loss = F.mse_loss(outputs['energy_pred'], en_ds)

        # β-weighted KL  (uses mu and logvar, both graph-connected)
        kl_loss = -0.5 * torch.mean(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        )
        total = mel_loss + 0.1 * (f0_loss + energy_loss) + self.beta * kl_loss

        return total, {
            'total':    total.item(),
            'mel_loss': mel_loss.item(),
            'kl_loss':  kl_loss.item(),
            'f0_loss':  f0_loss.item(),
        }

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ════════════════════════════════════════════════════════════════════════════
# CONDITION 5: β-VAE NO KL — deterministic AE ceiling
# ════════════════════════════════════════════════════════════════════════════

class BetaVAENoKLCondition(BetaVAECondition):
    """Deterministic autoencoder: reparameterize → always returns mu; KL=0.

    STRATEGY OVERRIDE vs BetaVAECondition:
      - reparameterize() bypasses sampling
      - compute_loss() omits KL term
    Expected: best reconstruction (ceiling), zero MIG score.
    """

    def __init__(self, hp, beta=0.0):
        super().__init__(hp, beta=0.0)

    def reparameterize(self, mu, logvar):
        # STRATEGY OVERRIDE: always deterministic — no stochasticity
        return mu

    def compute_loss(self, outputs, batch):
        mel_tgt  = batch['mel']
        mel_loss = F.mse_loss(outputs['mel_pred'], mel_tgt)

        T_text = outputs['f0_pred'].size(1)
        f0_ds  = F.adaptive_avg_pool1d(
            batch['f0'].unsqueeze(1), T_text).squeeze(1)
        en_ds  = F.adaptive_avg_pool1d(
            batch['energy'].unsqueeze(1), T_text).squeeze(1)
        f0_loss     = F.mse_loss(outputs['f0_pred'],     f0_ds / 400.0)
        energy_loss = F.mse_loss(outputs['energy_pred'], en_ds)

        # STRATEGY OVERRIDE: NO KL term; pure reconstruction
        total = mel_loss + 0.1 * (f0_loss + energy_loss)

        return total, {
            'total':    total.item(),
            'mel_loss': mel_loss.item(),
            'kl_loss':  0.0,
            'f0_loss':  f0_loss.item(),
        }


# ════════════════════════════════════════════════════════════════════════════
# BERT SURROGATE  (simulates frozen pre-trained BERT)
# ════════════════════════════════════════════════════════════════════════════

class BERTSurrogate(nn.Module):
    """Lightweight transformer that simulates bert-base-uncased.

    Pre-trained on 12-class sub-emotion classification (6 emotions × 2 sub-types)
    to learn sub-emotion-discriminative [CLS] embeddings.
    After pre-training, weights are frozen for BERT-conditioned conditions.
    """

    def __init__(self, vocab_size, hidden_dim=128, n_layers=2, nhead=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True, dropout=0.0, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Pre-training head: 12-class sub-emotion classifier
        self.sub_emotion_head = nn.Linear(hidden_dim, 12)

    def get_cls_embedding(self, input_ids):
        """Extract [CLS] embedding: [B, hidden_dim]"""
        B       = input_ids.size(0)
        tok_emb = self.embedding(input_ids)                          # [B, T, hidden]
        cls     = self.cls_token.expand(B, -1, -1)                   # [B, 1, hidden]
        x       = torch.cat([cls, tok_emb], dim=1)                   # [B, T+1, hidden]
        x       = self.encoder(x)                                    # [B, T+1, hidden]
        return x[:, 0, :]                                            # [B, hidden] CLS

    def forward(self, input_ids):
        cls    = self.get_cls_embedding(input_ids)                   # [B, hidden]
        logits = self.sub_emotion_head(cls)                          # [B, 12]
        return logits, cls


# ════════════════════════════════════════════════════════════════════════════
# CONDITION 6: BERT-CONDITIONED PROSODY (proposed, H2)
# ════════════════════════════════════════════════════════════════════════════

class BERTConditionedProsody(nn.Module):
    """H2 proposed: frozen BERT [CLS] → 2-layer MLP projection → conditioning.

    STRATEGY: 2-layer MLP (bert_dim→64→cond_dim) allows non-linear remapping
    of BERT's semantic space to prosody-relevant dimensions.
    """

    def __init__(self, hp):
        super().__init__()
        self.hp       = hp
        bert_dim      = hp['bert_dim']

        self.bert = BERTSurrogate(
            vocab_size=hp['vocab_size'], hidden_dim=bert_dim,
            n_layers=2, nhead=4,
        )
        # 2-layer MLP: bert_dim → 64 → cond_dim
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, 64), nn.ReLU(),
            nn.Linear(64, hp['cond_dim']),
        )
        self.text_encoder = TextEncoder(
            vocab_size=hp['vocab_size'], d_model=hp['d_model'],
            n_layers=hp['n_text_encoder_layers'], nhead=hp['nhead'],
        )
        self.variance_adaptor = VarianceAdaptor(hp['d_model'], hp['cond_dim'])
        self.mel_decoder = MelDecoder(
            hp['d_model'], hp['n_mels'],
            hp['n_mel_decoder_layers'], hp['nhead'],
        )
        self.mel_proj = nn.Linear(hp['cond_dim'], hp['d_model'])

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, batch):
        tokens    = batch['tts_tokens']
        bert_ids  = batch['bert_input_ids']
        durations = batch['duration']

        # BERT forward (frozen after pre-training)
        with torch.no_grad():
            _, cls_embed = self.bert(bert_ids)          # [B, bert_dim]

        cond   = self.projection(cls_embed)              # [B, cond_dim]  (2-layer MLP)
        h_text = self.text_encoder(tokens)               # [B, T_text, d_model]
        f0_pred, energy_pred, dur_pred = self.variance_adaptor(h_text, cond)

        h_exp  = length_regulate(h_text, durations)      # [B, T_mel, d_model]
        c_emb  = self.mel_proj(cond).unsqueeze(1).expand(-1, h_exp.size(1), -1)
        mel_pred = self.mel_decoder(h_exp + c_emb)       # [B, T_mel, n_mels]

        return {
            'mel_pred':      mel_pred,
            'f0_pred':       f0_pred,
            'energy_pred':   energy_pred,
            'dur_pred':      dur_pred,
            'semantic_cond': cond.detach(),
        }

    def compute_loss(self, outputs, batch):
        mel_tgt  = batch['mel']
        mel_loss = F.mse_loss(outputs['mel_pred'], mel_tgt)
        T_text   = outputs['f0_pred'].size(1)
        f0_ds    = F.adaptive_avg_pool1d(
            batch['f0'].unsqueeze(1), T_text).squeeze(1)
        en_ds    = F.adaptive_avg_pool1d(
            batch['energy'].unsqueeze(1), T_text).squeeze(1)
        f0_loss     = F.mse_loss(outputs['f0_pred'],     f0_ds / 400.0)
        energy_loss = F.mse_loss(outputs['energy_pred'], en_ds)
        total = mel_loss + 0.1 * (f0_loss + energy_loss)
        return total, {'total': total.item(), 'mel_loss': mel_loss.item()}

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ════════════════════════════════════════════════════════════════════════════
# CONDITION 7: BERT FROZEN DIRECT (ablation, H2)
# ════════════════════════════════════════════════════════════════════════════

class BERTFrozenDirectCondition(nn.Module):
    """H2 ablation: frozen BERT [CLS] → single linear (no MLP depth).

    STRATEGY OVERRIDE vs BERTConditionedProsody:
      - self.projection is absent
      - self.direct_linear is a single nn.Linear (no hidden layer, no activation)
    Tests whether the 2-layer MLP depth is necessary.
    """

    def __init__(self, hp):
        # Explicitly call nn.Module.__init__ — intentionally NOT calling
        # BERTConditionedProsody.__init__ to avoid instantiating self.projection
        nn.Module.__init__(self)
        self.hp      = hp
        bert_dim     = hp['bert_dim']

        self.bert = BERTSurrogate(
            vocab_size=hp['vocab_size'], hidden_dim=bert_dim,
            n_layers=2, nhead=4,
        )
        # STRATEGY OVERRIDE: single affine layer only — no MLP depth
        self.direct_linear = nn.Linear(bert_dim, hp['cond_dim'])

        self.text_encoder = TextEncoder(
            vocab_size=hp['vocab_size'], d_model=hp['d_model'],
            n_layers=hp['n_text_encoder_layers'], nhead=hp['nhead'],
        )
        self.variance_adaptor = VarianceAdaptor(hp['d_model'], hp['cond_dim'])
        self.mel_decoder = MelDecoder(
            hp['d_model'], hp['n_mels'],
            hp['n_mel_decoder_layers'], hp['nhead'],
        )
        self.mel_proj = nn.Linear(hp['cond_dim'], hp['d_model'])

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, batch):
        tokens    = batch['tts_tokens']
        bert_ids  = batch['bert_input_ids']
        durations = batch['duration']

        with torch.no_grad():
            _, cls_embed = self.bert(bert_ids)          # [B, bert_dim]

        # STRATEGY OVERRIDE: single linear, no activation, no intermediate layer
        cond   = self.direct_linear(cls_embed)          # [B, cond_dim]
        h_text = self.text_encoder(tokens)
        f0_pred, energy_pred, dur_pred = self.variance_adaptor(h_text, cond)
        h_exp  = length_regulate(h_text, durations)
        c_emb  = self.mel_proj(cond).unsqueeze(1).expand(-1, h_exp.size(1), -1)
        mel_pred = self.mel_decoder(h_exp + c_emb)

        return {
            'mel_pred':      mel_pred,
            'f0_pred':       f0_pred,
            'energy_pred':   energy_pred,
            'dur_pred':      dur_pred,
            'semantic_cond': cond.detach(),
        }

    def compute_loss(self, outputs, batch):
        mel_tgt  = batch['mel']
        mel_loss = F.mse_loss(outputs['mel_pred'], mel_tgt)
        T_text   = outputs['f0_pred'].size(1)
        f0_ds    = F.adaptive_avg_pool1d(
            batch['f0'].unsqueeze(1), T_text).squeeze(1)
        en_ds    = F.adaptive_avg_pool1d(
            batch['energy'].unsqueeze(1), T_text).squeeze(1)
        f0_loss     = F.mse_loss(outputs['f0_pred'],     f0_ds / 400.0)
        energy_loss = F.mse_loss(outputs['energy_pred'], en_ds)
        total = mel_loss + 0.1 * (f0_loss + energy_loss)
        return total, {'total': total.item(), 'mel_loss': mel_loss.item()}

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ════════════════════════════════════════════════════════════════════════════
# CONDITION 8: FASTSPEECH2 ONE-HOT BASELINE (H2 reference)
# ════════════════════════════════════════════════════════════════════════════

class FastSpeech2OneHotCondition(nn.Module):
    """H2 baseline: 6-class emotion embedding → conditioning vector.

    Only sees coarse emotion label; cannot distinguish sub-emotion variation.
    """

    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.emotion_embedding = nn.Embedding(hp['n_emotions'], hp['cond_dim'])
        self.text_encoder      = TextEncoder(
            vocab_size=hp['vocab_size'], d_model=hp['d_model'],
            n_layers=hp['n_text_encoder_layers'], nhead=hp['nhead'],
        )
        self.variance_adaptor  = VarianceAdaptor(hp['d_model'], hp['cond_dim'])
        self.mel_decoder       = MelDecoder(
            hp['d_model'], hp['n_mels'],
            hp['n_mel_decoder_layers'], hp['nhead'],
        )
        self.mel_proj = nn.Linear(hp['cond_dim'], hp['d_model'])

    def forward(self, batch):
        tokens      = batch['tts_tokens']
        emotion_idx = batch['emotion_idx']
        durations   = batch['duration']

        cond   = self.emotion_embedding(emotion_idx)   # [B, cond_dim]
        h_text = self.text_encoder(tokens)
        f0_pred, energy_pred, dur_pred = self.variance_adaptor(h_text, cond)
        h_exp  = length_regulate(h_text, durations)
        c_emb  = self.mel_proj(cond).unsqueeze(1).expand(-1, h_exp.size(1), -1)
        mel_pred = self.mel_decoder(h_exp + c_emb)

        return {
            'mel_pred':    mel_pred,
            'f0_pred':     f0_pred,
            'energy_pred': energy_pred,
            'dur_pred':    dur_pred,
        }

    def compute_loss(self, outputs, batch):
        mel_tgt  = batch['mel']
        mel_loss = F.mse_loss(outputs['mel_pred'], mel_tgt)
        T_text   = outputs['f0_pred'].size(1)
        f0_ds    = F.adaptive_avg_pool1d(
            batch['f0'].unsqueeze(1), T_text).squeeze(1)
        en_ds    = F.adaptive_avg_pool1d(
            batch['energy'].unsqueeze(1), T_text).squeeze(1)
        f0_loss     = F.mse_loss(outputs['f0_pred'],     f0_ds / 400.0)
        energy_loss = F.mse_loss(outputs['energy_pred'], en_ds)
        total = mel_loss + 0.1 * (f0_loss + energy_loss)
        return total, {'total': total.item(), 'mel_loss': mel_loss.item()}

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ════════════════════════════════════════════════════════════════════════════
# CONDITION 9: ACOUSTIC ORACLE (H3)
# ════════════════════════════════════════════════════════════════════════════

class AcousticOracle(nn.Module):
    """H3: Convolutional acoustic extractor + linear probe for MOS regression.

    Simulates the ASVspoof ECAPA-TDNN oracle:
      - Conv feature extractor (frozen after joint training)
      - Statistics-pooling (mean + std concatenation)
      - 2-layer linear probe predicting MOS score
    Evaluated via Spearman ρ between predicted and reference MOS.
    """

    def __init__(self, hp):
        super().__init__()
        self.hp          = hp
        n_mels           = hp['n_mels']
        emb_dim          = hp['oracle_embedding_dim']   # 64

        # Convolutional extractor (simulates ECAPA-TDNN)
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128,    128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(128,    emb_dim, kernel_size=3, padding=1), nn.ReLU(),
        )
        # Statistics pooling: mean + std → 2*emb_dim
        self.stats_fc = nn.Linear(emb_dim * 2, emb_dim)

        # 2-layer MOS regression probe
        self.probe = nn.Sequential(
            nn.Linear(emb_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def extract_embedding(self, mel):
        """mel: [B, T_mel, n_mels] → embedding: [B, emb_dim]"""
        x    = mel.transpose(1, 2)              # [B, n_mels, T_mel]
        x    = self.conv_extractor(x)           # [B, emb_dim, T_mel]
        mean = x.mean(dim=2)                    # [B, emb_dim]
        std  = x.std(dim=2).clamp(min=1e-6)     # [B, emb_dim]
        return F.relu(self.stats_fc(torch.cat([mean, std], dim=1)))  # [B, emb_dim]

    def forward(self, batch):
        mel       = batch['mel']
        embedding = self.extract_embedding(mel)         # [B, emb_dim]
        mos_pred  = self.probe(embedding)               # [B, 1]
        return {'mos_pred': mos_pred, 'embedding': embedding.detach()}

    def compute_loss(self, outputs, batch):
        mos_tgt  = batch['mos_score'].float().unsqueeze(-1)  # [B, 1]
        loss     = F.mse_loss(outputs['mos_pred'], mos_tgt)
        return loss, {'mse_loss': loss.item()}

    def freeze_extractor(self):
        for p in self.conv_extractor.parameters():
            p.requires_grad = False
        for p in self.stats_fc.parameters():
            p.requires_grad = False

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
