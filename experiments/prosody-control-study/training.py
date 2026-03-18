"""Training loops and evaluation functions for prosody control study."""
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from scipy import stats as scipy_stats

from models import (
    BetaVAECondition, BetaVAENoKLCondition,
    BERTConditionedProsody, BERTFrozenDirectCondition,
    FastSpeech2OneHotCondition, AcousticOracle,
    compute_mig_score,
)


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed):
    """Deterministic reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_utmos_proxy(mel_mse, f0_rmse):
    """UTMOS-proxy naturalness score in [1, 5].

    Formula:
        utmos = 1.5 + 2.5 * exp(-10 * mel_mse) + 1.0 * exp(-0.005 * f0_rmse)

    Properties:
        mel_mse=0, f0_rmse=0  → utmos ≈ 5.0   (perfect reconstruction)
        mel_mse=0.1           → mel contribution ≈ 0.92
        mel_mse=0.3           → mel contribution ≈ 0.25
    Direction: higher is better (mimics human MOS scale 1–5).
    """
    mel_part = 2.5 * np.exp(-10.0 * float(mel_mse))
    f0_part  = 1.0 * np.exp(-0.005 * float(f0_rmse))
    utmos    = 1.5 + mel_part + f0_part
    return float(np.clip(utmos, 1.0, 5.0))


# ════════════════════════════════════════════════════════════════════════════
# BERT SURROGATE PRE-TRAINING
# ════════════════════════════════════════════════════════════════════════════

def pretrain_bert(bert_surrogate, train_loader, hp, device):
    """Pre-train BERTSurrogate on 12-class sub-emotion classification.

    This simulates having a language model pre-trained on emotion-rich text.
    After training, call bert_surrogate.freeze_bert() (or equivalent) to lock
    the weights before using them in the main experiment.
    """
    bert_surrogate = bert_surrogate.to(device)
    bert_surrogate.train()
    optimizer = torch.optim.Adam(
        bert_surrogate.parameters(), lr=hp['bert_pretrain_lr']
    )
    ce = nn.CrossEntropyLoss()

    for epoch in range(hp['bert_pretrain_epochs']):
        total_loss = 0.0
        correct    = 0
        n          = 0
        for batch in train_loader:
            bert_ids   = batch['bert_input_ids'].to(device)
            emotion    = batch['emotion_idx'].to(device)
            sub_emo    = batch['sub_emotion_idx'].to(device)
            sub_label  = emotion * 2 + sub_emo          # 12-class label

            optimizer.zero_grad()
            logits, _ = bert_surrogate(bert_ids)        # [B, 12]
            loss      = ce(logits, sub_label)
            loss.backward()
            nn.utils.clip_grad_norm_(bert_surrogate.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == sub_label).sum().item()
            n          += bert_ids.size(0)

        acc = correct / max(n, 1)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  BERT pre-train epoch {epoch+1}/{hp['bert_pretrain_epochs']} "
                  f"loss={total_loss/max(len(train_loader),1):.4f} acc={acc:.3f}")

    return bert_surrogate.state_dict()


# ════════════════════════════════════════════════════════════════════════════
# ONE-EPOCH TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device):
    """Standard training step; returns per-batch-averaged loss dict or None on NaN."""
    model.train()
    losses = defaultdict(float)
    n      = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        outputs = model(batch)
        loss, loss_dict = model.compute_loss(outputs, batch)

        # ── NaN guard ──────────────────────────────────────────────────────
        if torch.isnan(loss) or loss.item() > 200:
            print(f"FAIL: NaN/divergence detected loss={loss.item():.4f}")
            return None

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            losses[k] += v
        n += 1

    return {k: v / max(n, 1) for k, v in losses.items()}


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION — GENERATIVE MODELS
# ════════════════════════════════════════════════════════════════════════════

def evaluate_generative(model, loader, device):
    """Compute mel_mse, f0_rmse, utmos_proxy on a data split."""
    model.eval()
    mel_mse_sum  = 0.0
    f0_rmse_sum  = 0.0
    n            = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            outputs = model(batch)

            # Mel reconstruction MSE
            mel_mse = F.mse_loss(outputs['mel_pred'], batch['mel']).item()

            # F0: upsample text-rate predictions to mel-rate for comparison
            f0_pred   = outputs['f0_pred']               # [B, T_text]
            f0_target = batch['f0']                      # [B, T_mel]
            f0_up     = F.interpolate(
                f0_pred.unsqueeze(1),
                size=f0_target.size(1), mode='linear', align_corners=False,
            ).squeeze(1)
            # f0_pred was trained on normalized Hz (/400); undo for RMSE in Hz
            f0_rmse = torch.sqrt(
                F.mse_loss(f0_up * 400.0, f0_target)
            ).item()

            mel_mse_sum  += mel_mse
            f0_rmse_sum  += f0_rmse
            n            += 1

    mel_mse_avg = mel_mse_sum / max(n, 1)
    f0_rmse_avg = f0_rmse_sum / max(n, 1)
    return {
        'mel_mse':    mel_mse_avg,
        'f0_rmse':    f0_rmse_avg,
        'utmos':      compute_utmos_proxy(mel_mse_avg, f0_rmse_avg),
    }


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION — ACOUSTIC ORACLE
# ════════════════════════════════════════════════════════════════════════════

def evaluate_oracle(oracle, loader, device):
    """Compute MSE and Spearman ρ between predicted and reference MOS."""
    oracle.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            outputs   = oracle(batch)
            all_pred.append(outputs['mos_pred'].cpu().squeeze(-1))
            all_true.append(batch['mos_score'].cpu().float())

    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()

    mse       = float(np.mean((pred - true) ** 2))
    rho, pval = scipy_stats.spearmanr(pred, true)
    return {
        'mse':           mse,
        'spearman_rho':  float(rho),
        'spearman_pval': float(pval),
    }


def train_oracle(oracle, train_loader, val_loader, hp, device):
    """Train oracle extractor + probe jointly; find best checkpoint by val Spearman ρ."""
    oracle = oracle.to(device)
    optimizer  = torch.optim.Adam(oracle.get_trainable_parameters(), lr=hp['probe_lr'])
    best_rho   = -2.0
    best_state = None

    for epoch in range(hp['probe_epochs']):
        oracle.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            outputs    = oracle(batch)
            loss, _    = oracle.compute_loss(outputs, batch)
            if torch.isnan(loss):
                break
            loss.backward()
            nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            optimizer.step()

        val_m = evaluate_oracle(oracle, val_loader, device)
        if val_m['spearman_rho'] > best_rho:
            best_rho   = val_m['spearman_rho']
            best_state = {k: v.clone() for k, v in oracle.state_dict().items()}
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Oracle epoch {epoch+1}/{hp['probe_epochs']} "
                  f"rho={val_m['spearman_rho']:.4f}")

    if best_state is not None:
        oracle.load_state_dict(best_state)
    return oracle, {'best_val_spearman_rho': best_rho}


# ════════════════════════════════════════════════════════════════════════════
# MULTI-SEED CONDITION RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_condition(
    condition_name, model_class, model_kwargs,
    hp, train_loader, val_loader, test_loader,
    device, start_time, bert_pretrained_state=None,
):
    """Run one condition across all seeds; return per-seed and aggregated metrics."""
    seeds        = hp['seeds']
    seed_results = []

    for seed in seeds:
        # ── Time budget guard ─────────────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed > hp['time_budget_sec'] * 0.93:
            print(f"  TIME_BUDGET_EXCEEDED: halting at seed={seed} "
                  f"(elapsed={elapsed:.0f}s)")
            break

        # ── Per-seed isolation ────────────────────────────────────────────
        set_all_seeds(seed)

        # ── Instantiate fresh model ───────────────────────────────────────
        model = model_class(**model_kwargs).to(device)

        # Load pre-trained BERT weights if applicable
        if bert_pretrained_state is not None and hasattr(model, 'bert'):
            model.bert.load_state_dict(bert_pretrained_state)
            model.freeze_bert()

        params    = model.get_trainable_parameters()
        optimizer = torch.optim.Adam(params, lr=hp['lr_generative'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp['epochs_generative']
        )

        best_val_loss = float('inf')
        best_state    = None

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(hp['epochs_generative']):
            if time.time() - start_time > hp['time_budget_sec'] * 0.93:
                print(f"    budget exceeded mid-epoch at epoch={epoch}")
                break

            train_m = train_one_epoch(model, train_loader, optimizer, device)
            if train_m is None:
                print(f"CONDITION_FAILED: {condition_name} NaN at epoch={epoch}")
                break

            if (epoch + 1) % hp['val_interval'] == 0:
                val_m = evaluate_generative(model, val_loader, device)
                if val_m['mel_mse'] < best_val_loss:
                    best_val_loss = val_m['mel_mse']
                    best_state    = {k: v.clone()
                                     for k, v in model.state_dict().items()}

            scheduler.step()

        # ── Restore best checkpoint ───────────────────────────────────────
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Test evaluation ───────────────────────────────────────────────
        test_m = evaluate_generative(model, test_loader, device)

        # MIG score for VAE conditions only
        if isinstance(model, BetaVAECondition) and \
                not isinstance(model, BetaVAENoKLCondition):
            test_m['mig'] = compute_mig_score(model, test_loader, device)
        else:
            test_m['mig'] = None

        test_m['seed'] = seed
        seed_results.append(test_m)

        print(f"    seed={seed} mel_mse={test_m['mel_mse']:.4f} "
              f"f0_rmse={test_m['f0_rmse']:.1f}Hz "
              f"utmos={test_m['utmos']:.3f}"
              + (f" mig={test_m['mig']:.3f}" if test_m['mig'] is not None else ""))

    if not seed_results:
        return {'per_seed': [], 'aggregate': {}}

    # ── Aggregate across seeds ────────────────────────────────────────────
    numeric_keys = [k for k in seed_results[0]
                    if k != 'seed' and seed_results[0][k] is not None
                    and isinstance(seed_results[0][k], (int, float))]
    aggregate = {}
    for key in numeric_keys:
        vals = [r[key] for r in seed_results if r.get(key) is not None]
        if vals:
            aggregate[key] = {
                'mean': float(np.mean(vals)),
                'std':  float(np.std(vals)),
                'n':    len(vals),
                'vals': vals,
            }

    return {'per_seed': seed_results, 'aggregate': aggregate}


# ════════════════════════════════════════════════════════════════════════════
# PAIRED STATISTICAL TEST
# ════════════════════════════════════════════════════════════════════════════

def paired_ttest(a_vals, b_vals, method_a, method_b, metric):
    """Paired t-test between two conditions over aligned seeds."""
    a = np.array(a_vals)
    b = np.array(b_vals)
    n = min(len(a), len(b))
    if n < 2:
        return
    diff   = a[:n] - b[:n]
    t_stat, p_val = scipy_stats.ttest_1samp(diff, 0)
    mean_d = float(np.mean(diff))
    std_d  = float(np.std(diff))
    # Cohen's d
    d = mean_d / (std_d + 1e-8)
    print(f"PAIRED: {method_a} vs {method_b} metric={metric} "
          f"mean_diff={mean_d:+.4f} std_diff={std_d:.4f} "
          f"t={t_stat:.3f} p={p_val:.4f} cohen_d={d:.3f}")


def bootstrap_ci(vals, n_boot=1000, ci=0.95, seed=0):
    """Bootstrap 95% confidence interval for the mean."""
    rng   = np.random.default_rng(seed)
    arr   = np.array(vals)
    boots = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo    = np.percentile(boots, (1 - ci) / 2 * 100)
    hi    = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)
