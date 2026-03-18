"""Compare synthetic vs real ESD experiment results and validate against paper claims.

Usage:
    python compare_results.py \
        --synthetic results_synthetic/results.json \
        --real      results_real/results.json \
        --output    comparison_report.md
"""
import json
import os
import argparse


# ── Paper-claimed numbers from ProsAudit (paper_final.md) ───────────────────
# Extracted from the paper narrative and tables.
PAPER_CLAIMS = {
    # UTMOS means (Table 2 in paper)
    'beta_vae_b1':        {'utmos': 3.42, 'mel_mse': 0.0312, 'f0_rmse': 18.7},
    'beta_vae_b2':        {'utmos': 3.31, 'mel_mse': 0.0341, 'f0_rmse': 19.4},
    'beta_vae_b4':        {'utmos': 3.18, 'mel_mse': 0.0378, 'f0_rmse': 21.2},
    'beta_vae_b8':        {'utmos': 2.97, 'mel_mse': 0.0431, 'f0_rmse': 24.8},
    'beta_vae_no_kl':     {'utmos': 3.71, 'mel_mse': 0.0278, 'f0_rmse': 16.3},
    'bert_projected':     {'utmos': 3.89, 'mel_mse': 0.0241, 'f0_rmse': 14.9},
    'bert_frozen_direct': {'utmos': 3.64, 'mel_mse': 0.0289, 'f0_rmse': 17.1},
    'onehot_baseline':    {'utmos': 3.47, 'mel_mse': 0.0318, 'f0_rmse': 18.2},
    # H1: β-UTMOS Spearman ρ = -0.94 (p<0.001)
    # H2: bert_projected Δ UTMOS = +0.42 vs onehot
    # H3: oracle Spearman ρ = 0.87
    'H1_spearman_rho':    -0.94,
    'H2_bert_delta':      +0.42,
    'H3_spearman_rho':    0.87,
}

CONDITIONS_ORDER = [
    'beta_vae_no_kl', 'beta_vae_b1', 'beta_vae_b2',
    'beta_vae_b4', 'beta_vae_b8',
    'bert_projected', 'bert_frozen_direct', 'onehot_baseline',
    'acoustic_oracle',
]

METRICS = ['utmos', 'mel_mse', 'f0_rmse', 'mig']


def load_results(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_agg(results, condition, metric):
    """Return (mean, std) or (None, None) for a condition/metric."""
    if results is None:
        return None, None
    cond_r = results.get('all_results', {}).get(condition, {})
    agg    = cond_r.get('aggregate', {})
    if metric in agg and 'mean' in agg[metric]:
        return agg[metric]['mean'], agg[metric]['std']
    return None, None


def get_oracle(results):
    if results is None:
        return None, None
    r = results.get('all_results', {}).get('acoustic_oracle', {})
    return r.get('spearman_rho'), r.get('spearman_pval')


def fmt(val, std=None, decimals=4):
    if val is None:
        return 'N/A'
    s = f'{val:.{decimals}f}'
    if std is not None:
        s += f' ±{std:.{decimals}f}'
    return s


def delta_str(a, b):
    """Return signed delta string, or N/A."""
    if a is None or b is None:
        return 'N/A'
    d = a - b
    return f'{d:+.4f}'


def generate_report(syn, real, out_path):
    lines = []
    A = lines.append

    A('# ProsAudit: Synthetic vs Real ESD Comparison Report\n')

    # ── Data mode summary ────────────────────────────────────────────────
    syn_mode  = syn.get('data_mode', 'synthetic') if syn else 'N/A'
    real_mode = real.get('data_mode', 'real_esd') if real else 'N/A'
    syn_t  = syn.get('wall_time_min',  'N/A') if syn  else 'N/A'
    real_t = real.get('wall_time_min', 'N/A') if real else 'N/A'

    A('## Run Summary\n')
    A(f'| Run       | Data mode   | Wall time (min) |')
    A(f'|-----------|-------------|-----------------|')
    A(f'| Synthetic | {syn_mode:<11} | {syn_t if isinstance(syn_t, str) else f"{syn_t:.1f}":<15} |')
    A(f'| Real ESD  | {real_mode:<11} | {real_t if isinstance(real_t, str) else f"{real_t:.1f}":<15} |')
    A('')

    # ── Per-condition metric table ─────────────────────────────────────
    A('## Per-Condition Metrics\n')
    A('### UTMOS (higher = better)\n')
    A(f'| Condition            | Synthetic        | Real ESD         | Paper Claim | Syn-Paper Δ | Real-Paper Δ |')
    A(f'|----------------------|------------------|------------------|-------------|-------------|--------------|')

    for cname in CONDITIONS_ORDER:
        if cname == 'acoustic_oracle':
            continue
        sm, ss = get_agg(syn,  cname, 'utmos')
        rm, rs = get_agg(real, cname, 'utmos')
        paper  = PAPER_CLAIMS.get(cname, {}).get('utmos')

        sp_delta = delta_str(sm, paper)
        rp_delta = delta_str(rm, paper)

        A(f'| {cname:<20} | {fmt(sm, ss):<16} | {fmt(rm, rs):<16} | '
          f'{fmt(paper):<11} | {sp_delta:<11} | {rp_delta:<12} |')

    A('')
    A('### mel_mse (lower = better)\n')
    A(f'| Condition            | Synthetic        | Real ESD         | Paper Claim |')
    A(f'|----------------------|------------------|------------------|-------------|')
    for cname in CONDITIONS_ORDER:
        if cname == 'acoustic_oracle':
            continue
        sm, ss = get_agg(syn,  cname, 'mel_mse')
        rm, rs = get_agg(real, cname, 'mel_mse')
        paper  = PAPER_CLAIMS.get(cname, {}).get('mel_mse')
        A(f'| {cname:<20} | {fmt(sm, ss):<16} | {fmt(rm, rs):<16} | {fmt(paper):<11} |')

    A('')
    A('### f0_rmse (lower = better)\n')
    A(f'| Condition            | Synthetic        | Real ESD         | Paper Claim |')
    A(f'|----------------------|------------------|------------------|-------------|')
    for cname in CONDITIONS_ORDER:
        if cname == 'acoustic_oracle':
            continue
        sm, ss = get_agg(syn,  cname, 'f0_rmse')
        rm, rs = get_agg(real, cname, 'f0_rmse')
        paper  = PAPER_CLAIMS.get(cname, {}).get('f0_rmse')
        A(f'| {cname:<20} | {fmt(sm, ss):<16} | {fmt(rm, rs):<16} | {fmt(paper):<11} |')

    A('')
    A('### MIG Disentanglement (β-VAE only)\n')
    A(f'| Condition            | Synthetic        | Real ESD         |')
    A(f'|----------------------|------------------|------------------|')
    for cname in ['beta_vae_b1', 'beta_vae_b2', 'beta_vae_b4', 'beta_vae_b8']:
        sm, ss = get_agg(syn,  cname, 'mig')
        rm, rs = get_agg(real, cname, 'mig')
        A(f'| {cname:<20} | {fmt(sm, ss):<16} | {fmt(rm, rs):<16} |')

    # ── Oracle ────────────────────────────────────────────────────────
    A('')
    A('## Acoustic Oracle (H3)\n')
    s_rho, s_pval = get_oracle(syn)
    r_rho, r_pval = get_oracle(real)
    paper_rho = PAPER_CLAIMS['H3_spearman_rho']
    A(f'| Run       | Spearman ρ  | p-value    | Paper claim | Δ vs paper  |')
    A(f'|-----------|-------------|------------|-------------|-------------|')
    A(f'| Synthetic | {fmt(s_rho):<11} | {fmt(s_pval):<10} | {paper_rho:<11} | {delta_str(s_rho, paper_rho):<11} |')
    A(f'| Real ESD  | {fmt(r_rho):<11} | {fmt(r_pval):<10} | {paper_rho:<11} | {delta_str(r_rho, paper_rho):<11} |')

    # ── Hypothesis verdicts ───────────────────────────────────────────
    A('')
    A('## Hypothesis Verdicts\n')

    def h_verdict(results, key, direction='supported'):
        if results is None:
            return 'N/A'
        v = results.get('metrics', {}).get(key)
        if v is None:
            return 'N/A'
        return 'SUPPORTED' if v else 'NOT SUPPORTED'

    # H1
    A('### H1: β increases → UTMOS decreases (Spearman ρ < -0.40)\n')
    A(f'| Run       | H1 Spearman ρ | Verdict        | Paper claim ρ |')
    A(f'|-----------|---------------|----------------|---------------|')
    s_h1 = syn.get('metrics', {}).get('H1_spearman_rho') if syn else None
    r_h1 = real.get('metrics', {}).get('H1_spearman_rho') if real else None
    A(f'| Synthetic | {fmt(s_h1):<13} | {h_verdict(syn,  "H1_supported"):<14} | {PAPER_CLAIMS["H1_spearman_rho"]:<13} |')
    A(f'| Real ESD  | {fmt(r_h1):<13} | {h_verdict(real, "H1_supported"):<14} | {PAPER_CLAIMS["H1_spearman_rho"]:<13} |')

    A('')
    A('### H2: BERT-projected > one-hot baseline (Δ UTMOS ≥ 0.30)\n')
    A(f'| Run       | bert_projected | onehot | Δ UTMOS | Paper Δ | Verdict      |')
    A(f'|-----------|----------------|--------|---------|---------|--------------|')
    for run_label, res in [('Synthetic', syn), ('Real ESD', real)]:
        bm, _ = get_agg(res, 'bert_projected',  'utmos')
        om, _ = get_agg(res, 'onehot_baseline', 'utmos')
        delta  = (bm - om) if (bm is not None and om is not None) else None
        verdict = ('SUPPORTED' if delta is not None and delta >= 0.30
                   else ('NOT SUPPORTED' if delta is not None else 'N/A'))
        A(f'| {run_label:<9} | {fmt(bm):<14} | {fmt(om):<6} | {fmt(delta):<7} | '
          f'{PAPER_CLAIMS["H2_bert_delta"]:<7} | {verdict:<12} |')

    A('')
    A('### H3: Oracle Spearman ρ > 0.80\n')
    A(f'| Run       | Spearman ρ | Verdict        | Paper claim |')
    A(f'|-----------|------------|----------------|-------------|')
    for run_label, rho in [('Synthetic', s_rho), ('Real ESD', r_rho)]:
        verdict = ('SUPPORTED' if rho is not None and rho > 0.80
                   else ('NOT SUPPORTED' if rho is not None else 'N/A'))
        A(f'| {run_label:<9} | {fmt(rho):<10} | {verdict:<14} | {paper_rho:<11} |')

    # ── Validation summary ────────────────────────────────────────────
    A('')
    A('## Validation Summary\n')
    A('> Δ = |experiment - paper|, threshold < 0.10 considered "within tolerance"\n')

    validated = []
    discrepant = []
    for cname in ['beta_vae_b1', 'beta_vae_b4', 'bert_projected', 'onehot_baseline']:
        paper_u = PAPER_CLAIMS.get(cname, {}).get('utmos')
        for run_label, res in [('synthetic', syn), ('real', real)]:
            m, _ = get_agg(res, cname, 'utmos')
            if m is not None and paper_u is not None:
                diff = abs(m - paper_u)
                entry = f'{cname} ({run_label}): exp={m:.4f} paper={paper_u} Δ={diff:.4f}'
                if diff < 0.10:
                    validated.append(entry)
                else:
                    discrepant.append(entry)

    A('### Within tolerance (Δ < 0.10):')
    if validated:
        for e in validated:
            A(f'- {e}')
    else:
        A('- (none)')

    A('')
    A('### Discrepant (Δ ≥ 0.10):')
    if discrepant:
        for e in discrepant:
            A(f'- {e}')
    else:
        A('- (none)')

    A('')
    A('---')
    A('*Generated by compare_results.py*')

    report = '\n'.join(lines)
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"Report saved: {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', default='results_synthetic/results.json')
    parser.add_argument('--real',      default='results_real/results.json')
    parser.add_argument('--output',    default='comparison_report.md')
    args = parser.parse_args()

    syn  = load_results(args.synthetic)
    real = load_results(args.real)

    if syn is None and real is None:
        print("ERROR: neither results file found. Run synthetic and/or real experiments first.")
        return

    if syn is None:
        print(f"WARNING: synthetic results not found at {args.synthetic}")
    else:
        print(f"Loaded synthetic results: {len(syn.get('all_results', {}))} conditions")

    if real is None:
        print(f"WARNING: real results not found at {args.real}")
    else:
        print(f"Loaded real results: {len(real.get('all_results', {}))} conditions")

    report = generate_report(syn, real, args.output)
    print("\n--- REPORT PREVIEW (first 60 lines) ---")
    for line in report.split('\n')[:60]:
        print(line)


if __name__ == '__main__':
    main()
