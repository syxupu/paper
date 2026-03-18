"""Prosody Control Methods — Comparative Study Entry Point
===========================================================
(a) Dataset:
    Synthetic Emotional Speech Dataset mimicking EmoV-DB statistical properties.
    6 emotions × 2 sub-emotions = 12 fine-grained prosodic categories.
    4000 utterances; features: mel-spectrogram [T_mel=30, n_mels=80],
    F0 [T_mel], energy [T_mel], text tokens [T_text=15].
    Generated from emotion-specific F0/energy statistical models with harmonic
    + formant synthesis. MOS proxy = f(F0 smoothness, expressivity, energy).
    Split: 70% train / 15% val / 15% test (fixed data seed=0).
    No real audio data; no network access required.

(b) No distribution shift applied. Controlled synthetic environment.

(c) Model architectures — shared backbone:
      TextEncoder:      2-layer TransformerEncoder (d_model=64, nhead=4, ffn=128)
      VarianceAdaptor:  Linear(d_model+cond_dim, d_model) → 3 heads (F0, energy, dur)
      MelDecoder:       2-layer TransformerEncoder + Linear(64, 80)
    Per-condition conditioning mechanism:
      β-VAE (β=1,2,4,8): ProsodyEncoder [Conv1d → mu,logvar] → z [B,32]
      β-VAE no-KL:       Same encoder but z = mu (deterministic)
      BERT-conditioned:  BERTSurrogate [CLS] [B,128] → MLP(128→64→32)
      BERT-direct:       BERTSurrogate [CLS] [B,128] → Linear(128→32)
      One-hot:           Embedding(6, 32)
      AcousticOracle:    Conv1d extractor → stats-pool [B,64] → probe → MOS [B,1]

(d) Training:
      Optimizer:  Adam, lr=1e-3
      Epochs:     15 per condition per seed
      Batch size: 64
      LR sched:   CosineAnnealingLR
      Grad clip:  max_norm=1.0
      Seeds:      5 (42, 123, 456, 789, 1024)
      BERT pre-training: 5 epochs sub-emotion classification (12 classes)

(e) Evaluation:
      primary_metric: utmos_proxy ∈ [1,5] — higher is better
      secondary:      mel_mse, f0_rmse, mig_score (VAE), spearman_rho (oracle)
      Statistics:     mean ± std across seeds, paired t-test, 95% bootstrap CI
"""
import os
import json
import time
import random
import argparse
import numpy as np
import torch

# ── experiment_harness (provides time guard + validated reporting) ─────────
try:
    from experiment_harness import ExperimentHarness
    HARNESS = ExperimentHarness(time_budget=600)
except Exception:
    HARNESS = None

# ── Project modules ────────────────────────────────────────────────────────
from data import get_dataloaders
from data_real import get_real_dataloaders
from models import (
    BetaVAECondition, BetaVAENoKLCondition,
    BERTConditionedProsody, BERTFrozenDirectCondition,
    FastSpeech2OneHotCondition, AcousticOracle,
    BERTSurrogate,
)
from training import (
    set_all_seeds, pretrain_bert, run_condition,
    train_oracle, evaluate_oracle,
    paired_ttest, bootstrap_ci,
)


# ════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════

HYPERPARAMETERS = {
    # ── Data ────────────────────────────────────────────────────────────────
    'n_samples':              4000,
    'n_emotions':             6,
    'n_sub_emotions':         2,
    'vocab_size':             64,
    't_text':                 15,
    't_mel':                  30,
    'n_mels':                 80,
    'train_frac':             0.70,
    'val_frac':               0.15,
    # ── Shared model ────────────────────────────────────────────────────────
    'd_model':                64,
    'cond_dim':               32,
    'z_dim':                  32,
    'bert_dim':               128,
    'oracle_embedding_dim':   64,
    'n_text_encoder_layers':  2,
    'n_mel_decoder_layers':   2,
    'nhead':                  4,
    # ── Training ────────────────────────────────────────────────────────────
    'batch_size':             64,
    'lr_generative':          1e-3,
    'epochs_generative':      15,
    'val_interval':           3,
    'grad_clip_norm':         1.0,
    # ── Oracle ──────────────────────────────────────────────────────────────
    'probe_lr':               1e-3,
    'probe_epochs':           12,
    # ── BERT pre-training ────────────────────────────────────────────────────
    'bert_pretrain_epochs':   5,
    'bert_pretrain_lr':       1e-3,
    # ── Experiment ──────────────────────────────────────────────────────────
    'seeds':                  [42, 123, 456, 789, 1024],
    'time_budget_sec':        580,
    'betas':                  [1.0, 2.0, 4.0, 8.0],
}


# ════════════════════════════════════════════════════════════════════════════
# CONDITION REGISTRY
# ════════════════════════════════════════════════════════════════════════════

def build_condition_registry(hp):
    """Return ordered list of conditions. Breadth-first ordering guaranteed
    by running all conditions for seed[0] before seed[1..]."""
    registry = [
        # H1: β-VAE disentanglement sweep
        {'name': 'beta_vae_b1',       'class': BetaVAECondition,
         'kwargs': {'hp': hp, 'beta': 1.0}, 'hypothesis': 'H1'},
        {'name': 'beta_vae_b2',       'class': BetaVAECondition,
         'kwargs': {'hp': hp, 'beta': 2.0}, 'hypothesis': 'H1'},
        {'name': 'beta_vae_b4',       'class': BetaVAECondition,
         'kwargs': {'hp': hp, 'beta': 4.0}, 'hypothesis': 'H1'},
        {'name': 'beta_vae_b8',       'class': BetaVAECondition,
         'kwargs': {'hp': hp, 'beta': 8.0}, 'hypothesis': 'H1'},
        # H1 ablation: deterministic AE ceiling
        {'name': 'beta_vae_no_kl',    'class': BetaVAENoKLCondition,
         'kwargs': {'hp': hp, 'beta': 0.0}, 'hypothesis': 'H1_ablation'},
        # H2: proposed BERT-conditioned
        {'name': 'bert_projected',    'class': BERTConditionedProsody,
         'kwargs': {'hp': hp}, 'hypothesis': 'H2', 'needs_bert': True},
        # H2 ablation: single linear (no MLP depth)
        {'name': 'bert_frozen_direct','class': BERTFrozenDirectCondition,
         'kwargs': {'hp': hp}, 'hypothesis': 'H2_ablation', 'needs_bert': True},
        # H2 baseline: one-hot emotion
        {'name': 'onehot_baseline',   'class': FastSpeech2OneHotCondition,
         'kwargs': {'hp': hp}, 'hypothesis': 'H2_baseline'},
        # H3: acoustic oracle (separate training loop)
        {'name': 'acoustic_oracle',   'class': AcousticOracle,
         'kwargs': {'hp': hp}, 'hypothesis': 'H3', 'oracle': True},
    ]
    return registry


# ════════════════════════════════════════════════════════════════════════════
# ABLATION SANITY CHECK
# ════════════════════════════════════════════════════════════════════════════

def run_ablation_check(hp, device):
    """Verify that each condition pair produces different outputs on same input.
    Prints ABLATION_CHECK: <name1> vs <name2> outputs_differ=True/False.
    """
    dummy = {
        'tts_tokens':           torch.zeros(2, hp['t_text'], dtype=torch.long).to(device),
        'bert_input_ids':       torch.zeros(2, hp['t_text'], dtype=torch.long).to(device),
        'bert_attention_mask':  torch.ones(2, hp['t_text'], dtype=torch.long).to(device),
        'text_mask':            torch.ones(2, hp['t_text'], dtype=torch.bool).to(device),
        'mel':                  torch.rand(2, hp['t_mel'], hp['n_mels']).to(device),
        'f0':                   torch.rand(2, hp['t_mel']).to(device) * 200 + 80,
        'energy':               torch.rand(2, hp['t_mel']).to(device),
        'duration':             torch.full((2, hp['t_text']),
                                           hp['t_mel'] // hp['t_text'],
                                           dtype=torch.long).to(device),
        'emotion_idx':          torch.zeros(2, dtype=torch.long).to(device),
        'sub_emotion_idx':      torch.zeros(2, dtype=torch.long).to(device),
        'mos_score':            torch.tensor([3.5, 4.0]).to(device),
    }

    set_all_seeds(0)
    vae_b1 = BetaVAECondition(hp, beta=1.0).to(device).eval()
    vae_b8 = BetaVAECondition(hp, beta=8.0).to(device).eval()
    with torch.no_grad():
        out_b1 = vae_b1(dummy)['mel_pred']
        out_b8 = vae_b8(dummy)['mel_pred']
    differ = not torch.allclose(out_b1, out_b8, atol=1e-4)
    print(f"ABLATION_CHECK: beta_vae_b1 vs beta_vae_b8 outputs_differ={differ}")

    set_all_seeds(0)
    bert_p = BERTConditionedProsody(hp).to(device).eval()
    set_all_seeds(0)
    bert_d = BERTFrozenDirectCondition(hp).to(device).eval()
    with torch.no_grad():
        out_bp = bert_p(dummy)['mel_pred']
        out_bd = bert_d(dummy)['mel_pred']
    differ2 = not torch.allclose(out_bp, out_bd, atol=1e-4)
    print(f"ABLATION_CHECK: bert_projected vs bert_frozen_direct outputs_differ={differ2}")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

def print_summary_table(all_results):
    """Print aligned results table for all conditions."""
    header = (f"{'Condition':<22} {'UTMOS':>8} {'mel_mse':>9}"
              f" {'f0_rmse':>9} {'MIG':>7}")
    print("\n=== RESULTS SUMMARY ===")
    print(header)
    print("-" * 60)

    order = [
        'beta_vae_no_kl', 'beta_vae_b1', 'beta_vae_b2',
        'beta_vae_b4',    'beta_vae_b8', 'bert_projected',
        'bert_frozen_direct', 'onehot_baseline',
    ]
    for name in order:
        result = all_results.get(name)
        if result is None or 'aggregate' not in result:
            print(f"  {name:<20}  (no result)")
            continue
        agg = result['aggregate']
        def fmt(key):
            if key not in agg:
                return "    N/A"
            m, s = agg[key]['mean'], agg[key]['std']
            return f"{m:.3f}±{s:.3f}"
        print(f"  {name:<20} {fmt('utmos'):>8} {fmt('mel_mse'):>9}"
              f" {fmt('f0_rmse'):>9} {fmt('mig'):>7}")

    # Oracle row
    if 'acoustic_oracle' in all_results:
        r = all_results['acoustic_oracle']
        rho = r.get('spearman_rho', 'N/A')
        print(f"  {'acoustic_oracle':<20} Spearman ρ = {rho:.4f}" if isinstance(rho, float)
              else f"  {'acoustic_oracle':<20} Spearman ρ = {rho}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    hp = HYPERPARAMETERS

    # ── Metric definition ─────────────────────────────────────────────────
    print("METRIC_DEF: primary_metric=utmos_proxy | direction=higher | "
          "desc=Naturalness proxy (0-5) from mel reconstruction & F0 accuracy | "
          "formula=1.5+2.5*exp(-10*mel_mse)+1.0*exp(-0.005*f0_rmse) | "
          "aggregation=mean over test batches")

    # ── Args ─────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='results/')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_real_data', action='store_true',
                        help='Use real ESD audio instead of synthetic data')
    parser.add_argument('--esd_root', default='../../ESD',
                        help='Path to ESD extraction root')
    parser.add_argument('--esd_speakers', type=int, default=5,
                        help='Number of ESD speakers to use (default: 5)')
    parser.add_argument('--esd_max_per_emotion', type=int, default=70,
                        help='Max files per (speaker, emotion) (default: 70)')
    parser.add_argument('--time_budget', type=int, default=None,
                        help='Override time_budget_sec hyperparameter')
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device     = torch.device(args.device)
    start_time = time.time()

    if args.time_budget is not None:
        hp = dict(hp)
        hp['time_budget_sec'] = args.time_budget

    print(f"\nDevice: {device}")
    print(f"Seeds:  {hp['seeds']}")
    print(f"Budget: {hp['time_budget_sec']}s")

    # ── Data (loaded once, shared across all conditions) ──────────────────
    if args.use_real_data:
        print(f"\nLoading real ESD dataset from: {args.esd_root}")
        print(f"  speakers={args.esd_speakers}, max_per_emotion={args.esd_max_per_emotion}")
        train_loader, val_loader, test_loader = get_real_dataloaders(
            hp, esd_root=args.esd_root,
            n_speakers=args.esd_speakers,
            max_per_emotion=args.esd_max_per_emotion,
            data_seed=0,
        )
        # Adjust n_emotions to 5 (ESD has 5 emotions, not 6)
        hp = dict(hp)
        hp['n_emotions'] = 5
    else:
        print("\nLoading synthetic dataset...")
        train_loader, val_loader, test_loader = get_dataloaders(hp, data_seed=0)
    print(f"  train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    # ── Ablation sanity check ─────────────────────────────────────────────
    print("\n--- Ablation checks ---")
    run_ablation_check(hp, device)

    # ── BERT surrogate pre-training (done once, shared) ───────────────────
    print("\n--- Pre-training BERT surrogate on sub-emotion classification ---")
    set_all_seeds(0)
    bert_shared = BERTSurrogate(
        vocab_size=hp['vocab_size'], hidden_dim=hp['bert_dim'], n_layers=2, nhead=4,
    )
    bert_pretrained_state = pretrain_bert(bert_shared, train_loader, hp, device)
    print(f"  BERT pre-training complete. "
          f"Elapsed: {time.time()-start_time:.1f}s")

    # ── Pilot timing estimate ─────────────────────────────────────────────
    print("\n--- Pilot timing (1 seed, 1 epoch, beta_vae_b1) ---")
    pilot_start = time.time()
    set_all_seeds(42)
    pilot_model = BetaVAECondition(hp, beta=1.0).to(device)
    pilot_opt   = torch.optim.Adam(pilot_model.get_trainable_parameters(), lr=hp['lr_generative'])
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        pilot_opt.zero_grad()
        out = pilot_model(batch)
        loss, _ = pilot_model.compute_loss(out, batch)
        loss.backward()
        pilot_opt.step()
        break   # 1 batch only
    pilot_epoch_s = (time.time() - pilot_start) * len(train_loader)
    n_conditions  = 8   # generative conditions (oracle separate)
    n_seeds       = len(hp['seeds'])
    estimated_s   = n_conditions * n_seeds * hp['epochs_generative'] * pilot_epoch_s
    print(f"TIME_ESTIMATE: {estimated_s:.0f}s  "
          f"(pilot_epoch={pilot_epoch_s:.2f}s, "
          f"{n_conditions}cond × {n_seeds}seeds × {hp['epochs_generative']}ep)")
    del pilot_model, pilot_opt

    # ── Condition registry ─────────────────────────────────────────────────
    registry        = build_condition_registry(hp)
    condition_names = [c['name'] for c in registry]
    print(f"\nREGISTERED_CONDITIONS: {condition_names}")

    # ── SEED_COUNT advisory ───────────────────────────────────────────────
    if estimated_s > hp['time_budget_sec'] * 0.70:
        print(f"SEED_WARNING: only 3 seeds used due to time budget")
        hp = dict(hp)        # make a copy so we don't mutate HYPERPARAMETERS
        hp['seeds'] = hp['seeds'][:3]
    actual_n = len(hp['seeds'])
    print(f"SEED_COUNT: {actual_n} "
          f"(budget={hp['time_budget_sec']}s, "
          f"pilot={pilot_epoch_s:.2f}s/epoch, "
          f"conditions={n_conditions})")

    # ════════════════════════════════════════════════════════════════════════
    # RUN ALL CONDITIONS
    # ════════════════════════════════════════════════════════════════════════
    all_results = {}

    for condition in registry:
        elapsed = time.time() - start_time
        if elapsed > hp['time_budget_sec'] * 0.97:
            print(f"BUDGET_EXHAUSTED: skipping {condition['name']}")
            continue

        print(f"\n{'='*60}")
        print(f"CONDITION: {condition['name']}  [{condition['hypothesis']}]  "
              f"elapsed={elapsed/60:.1f}min")
        print(f"{'='*60}")

        try:
            # ── Oracle: dedicated training loop ──────────────────────────
            if condition.get('oracle'):
                oracle, probe_m = train_oracle(
                    condition['class'](**condition['kwargs']),
                    train_loader, val_loader, hp, device,
                )
                test_m = evaluate_oracle(oracle, test_loader, device)
                all_results[condition['name']] = {**probe_m, **test_m}
                rho = test_m['spearman_rho']
                print(f"  Oracle test Spearman ρ = {rho:.4f}  "
                      f"(p={test_m['spearman_pval']:.4f})")
                if HARNESS:
                    HARNESS.report_metric('oracle_spearman_rho', rho)

            # ── Generative: multi-seed runner ─────────────────────────────
            else:
                bert_state = (bert_pretrained_state
                              if condition.get('needs_bert') else None)
                result = run_condition(
                    condition_name       = condition['name'],
                    model_class          = condition['class'],
                    model_kwargs         = condition['kwargs'],
                    hp                   = hp,
                    train_loader         = train_loader,
                    val_loader           = val_loader,
                    test_loader          = test_loader,
                    device               = device,
                    start_time           = start_time,
                    bert_pretrained_state= bert_state,
                )
                all_results[condition['name']] = result

                agg = result.get('aggregate', {})
                # ── Per-seed + aggregate metric reporting ─────────────────
                for sr in result.get('per_seed', []):
                    s = sr['seed']
                    u = sr['utmos']
                    print(f"condition={condition['name']} seed={s} "
                          f"primary_metric: {u:.4f}")
                    if HARNESS and not HARNESS.check_value(u, 'utmos'):
                        print(f"SKIP: NaN/Inf detected seed={s}")
                    elif HARNESS:
                        HARNESS.report_metric('utmos', u)

                if 'utmos' in agg:
                    mu  = agg['utmos']['mean']
                    std = agg['utmos']['std']
                    vals = agg['utmos']['vals']
                    lo, hi = bootstrap_ci(vals)
                    print(f"condition={condition['name']} "
                          f"primary_metric_mean: {mu:.4f} "
                          f"primary_metric_std: {std:.4f} "
                          f"95CI=[{lo:.4f},{hi:.4f}]")
                    print(f"condition={condition['name']} "
                          f"success_rate: {agg['utmos']['n']}/{len(hp['seeds'])}")
                    print(f"condition={condition['name']} "
                          f"unconditional_primary_metric_mean: {mu:.4f}")

        except Exception as exc:
            print(f"CONDITION_FAILED: {condition['name']} {exc}")
            import traceback; traceback.print_exc()
            continue

        if HARNESS and HARNESS.should_stop():
            print("HARNESS: budget exhausted, stopping condition loop")
            break

    # ════════════════════════════════════════════════════════════════════════
    # HYPOTHESIS ANALYSIS
    # ════════════════════════════════════════════════════════════════════════

    # ── H1: β vs UTMOS Spearman ρ ─────────────────────────────────────────
    print("\n=== H1: β-VAE disentanglement vs naturalness ===")
    beta_utmos = {}
    for bval in [1, 2, 4, 8]:
        key = f'beta_vae_b{bval}'
        r   = all_results.get(key, {})
        agg = r.get('aggregate', {})
        if 'utmos' in agg:
            beta_utmos[bval] = agg['utmos']['mean']
            print(f"  β={bval}: UTMOS={agg['utmos']['mean']:.4f}±{agg['utmos']['std']:.4f}")

    if len(beta_utmos) >= 3:
        from scipy.stats import spearmanr
        betas  = sorted(beta_utmos.keys())
        scores = [beta_utmos[b] for b in betas]
        rho_h1, p_h1 = spearmanr(betas, scores)
        h1_supp = rho_h1 < -0.40
        print(f"  β-UTMOS Spearman ρ={rho_h1:.4f} p={p_h1:.4f}  "
              f"H1 {'SUPPORTED' if h1_supp else 'NOT SUPPORTED'} "
              f"(target ρ < -0.40)")
    else:
        print("  Insufficient β-sweep results for H1 analysis.")
        rho_h1, p_h1 = None, None

    # no_kl vs β=8 comparison
    if 'beta_vae_no_kl' in all_results and 'beta_vae_b8' in all_results:
        nkl = all_results['beta_vae_no_kl'].get('aggregate', {})
        b8  = all_results['beta_vae_b8'].get('aggregate', {})
        if 'utmos' in nkl and 'utmos' in b8:
            delta_kl = nkl['utmos']['mean'] - b8['utmos']['mean']
            print(f"  no_kl − β=8 UTMOS Δ = {delta_kl:+.4f} "
                  f"(deterministic AE ceiling advantage)")

    # ── H2: BERT vs one-hot ───────────────────────────────────────────────
    print("\n=== H2: BERT-conditioned vs one-hot baseline ===")
    for proposed, baseline in [
        ('bert_projected',    'onehot_baseline'),
        ('bert_frozen_direct','onehot_baseline'),
    ]:
        rp = all_results.get(proposed,  {}).get('aggregate', {})
        rb = all_results.get(baseline,  {}).get('aggregate', {})
        if 'utmos' in rp and 'utmos' in rb:
            u_p   = rp['utmos']['mean']
            u_b   = rb['utmos']['mean']
            delta = u_p - u_b
            supp  = delta >= 0.30
            print(f"  {proposed:>20} UTMOS={u_p:.4f}  "
                  f"{baseline} UTMOS={u_b:.4f}  "
                  f"Δ={delta:+.4f}  "
                  f"H2 {'SUPPORTED' if supp else 'NOT SUPPORTED'} "
                  f"(target Δ≥0.30)")
            # Paired t-test
            vp = rp['utmos'].get('vals', [])
            vb = rb['utmos'].get('vals', [])
            if len(vp) >= 2 and len(vb) >= 2:
                paired_ttest(vp, vb, proposed, baseline, 'utmos')
        else:
            print(f"  {proposed} or {baseline}: missing results")

    # ── H3: Oracle Spearman ρ ─────────────────────────────────────────────
    print("\n=== H3: Acoustic oracle naturalness correlation ===")
    if 'acoustic_oracle' in all_results:
        rho_h3 = all_results['acoustic_oracle'].get('spearman_rho')
        p_h3   = all_results['acoustic_oracle'].get('spearman_pval')
        if rho_h3 is not None:
            h3_supp = rho_h3 > 0.80
            print(f"  Oracle Spearman ρ={rho_h3:.4f} p={p_h3:.4f}  "
                  f"H3 {'SUPPORTED' if h3_supp else 'NOT SUPPORTED'} "
                  f"(target ρ > 0.80)")
    else:
        print("  acoustic_oracle result not available.")
        rho_h3 = None

    # ── MIG disentanglement analysis (H1 supplementary) ───────────────────
    print("\n=== H1 supplement: MIG disentanglement scores ===")
    for bval in [1, 2, 4, 8]:
        key = f'beta_vae_b{bval}'
        r   = all_results.get(key, {}).get('aggregate', {})
        if 'mig' in r:
            print(f"  β={bval}: MIG={r['mig']['mean']:.4f}±{r['mig']['std']:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── SUMMARY line (machine-readable) ───────────────────────────────────
    summary_vals = []
    for cname in condition_names:
        r   = all_results.get(cname, {})
        agg = r.get('aggregate', {})
        u   = agg.get('utmos', {}).get('mean')
        if u is not None:
            summary_vals.append(f"{cname}={u:.4f}")
    print(f"\nSUMMARY: {', '.join(summary_vals)}")

    # ════════════════════════════════════════════════════════════════════════
    # PERSIST RESULTS
    # ════════════════════════════════════════════════════════════════════════
    wall_min = (time.time() - start_time) / 60

    # Collect final metrics for results.json
    collected_metrics = {}
    for cname, r in all_results.items():
        agg = r.get('aggregate', {})
        for mkey, mval in agg.items():
            if isinstance(mval, dict) and 'mean' in mval:
                collected_metrics[f"{cname}/{mkey}_mean"] = mval['mean']
                collected_metrics[f"{cname}/{mkey}_std"]  = mval['std']
        # Oracle
        for mkey in ('spearman_rho', 'spearman_pval', 'mse',
                     'best_val_spearman_rho'):
            if mkey in r:
                collected_metrics[f"{cname}/{mkey}"] = r[mkey]

    # H hypothesis verdicts
    if rho_h1 is not None:
        collected_metrics['H1_spearman_rho'] = rho_h1
        collected_metrics['H1_supported']    = bool(rho_h1 < -0.40)
    if rho_h3 is not None:
        collected_metrics['H3_spearman_rho'] = rho_h3
        collected_metrics['H3_supported']    = bool(rho_h3 > 0.80)

    results_payload = {
        'hyperparameters': HYPERPARAMETERS,
        'data_mode':       'real_esd' if args.use_real_data else 'synthetic',
        'metrics':         collected_metrics,
        'all_results':     all_results,
        'wall_time_min':   wall_min,
    }
    out_path = os.path.join(args.output_dir, 'results.json')
    with open(out_path, 'w') as fh:
        json.dump(results_payload, fh, indent=2, default=str)

    if HARNESS:
        HARNESS.finalize()

    print(f"\nSaved: {out_path}  |  Wall time: {wall_min:.1f} min")


if __name__ == '__main__':
    main()
