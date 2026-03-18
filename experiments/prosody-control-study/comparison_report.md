# ProsAudit: Synthetic vs Real ESD Comparison Report

## Run Summary

| Run       | Data mode   | Wall time (min) |
|-----------|-------------|-----------------|
| Synthetic | synthetic   | 7.9             |
| Real ESD  | real_esd    | 9.0             |

## Per-Condition Metrics

### UTMOS (higher = better)

| Condition            | Synthetic        | Real ESD         | Paper Claim | Syn-Paper Δ | Real-Paper Δ |
|----------------------|------------------|------------------|-------------|-------------|--------------|
| beta_vae_no_kl       | 4.4379 ±0.0003   | 3.9603 ±0.0065   | 3.7100      | +0.7279     | +0.2503      |
| beta_vae_b1          | 4.4202 ±0.0006   | 3.7492 ±0.0036   | 3.4200      | +1.0002     | +0.3292      |
| beta_vae_b2          | 4.4202 ±0.0006   | 3.7490 ±0.0034   | 3.3100      | +1.1102     | +0.4390      |
| beta_vae_b4          | 4.4201 ±0.0005   | 3.7489 ±0.0033   | 3.1800      | +1.2401     | +0.5689      |
| beta_vae_b8          | 4.4201 ±0.0005   | 3.7489 ±0.0033   | 2.9700      | +1.4501     | +0.7789      |
| bert_projected       | 4.4308 ±0.0009   | 3.6893 ±0.0793   | 3.8900      | +0.5408     | -0.2007      |
| bert_frozen_direct   | 4.4273 ±0.0004   | N/A              | 3.6400      | +0.7873     | N/A          |
| onehot_baseline      | 4.4252 ±0.0015   | N/A              | 3.4700      | +0.9552     | N/A          |

### mel_mse (lower = better)

| Condition            | Synthetic        | Real ESD         | Paper Claim |
|----------------------|------------------|------------------|-------------|
| beta_vae_no_kl       | 0.0221 ±0.0000   | 0.0303 ±0.0002   | 0.0278      |
| beta_vae_b1          | 0.0227 ±0.0000   | 0.0357 ±0.0001   | 0.0312      |
| beta_vae_b2          | 0.0227 ±0.0000   | 0.0357 ±0.0001   | 0.0341      |
| beta_vae_b4          | 0.0228 ±0.0000   | 0.0357 ±0.0001   | 0.0378      |
| beta_vae_b8          | 0.0228 ±0.0000   | 0.0357 ±0.0001   | 0.0431      |
| bert_projected       | 0.0224 ±0.0000   | 0.0392 ±0.0047   | 0.0241      |
| bert_frozen_direct   | 0.0225 ±0.0000   | N/A              | 0.0289      |
| onehot_baseline      | 0.0226 ±0.0000   | N/A              | 0.0318      |

### f0_rmse (lower = better)

| Condition            | Synthetic        | Real ESD         | Paper Claim |
|----------------------|------------------|------------------|-------------|
| beta_vae_no_kl       | 13.6347 ±0.0785  | 97.6357 ±1.2310  | 16.3000     |
| beta_vae_b1          | 14.7617 ±0.1542  | 138.6528 ±0.6365 | 18.7000     |
| beta_vae_b2          | 14.7565 ±0.1524  | 138.6805 ±0.5970 | 19.4000     |
| beta_vae_b4          | 14.7563 ±0.1480  | 138.7012 ±0.5957 | 21.2000     |
| beta_vae_b8          | 14.7558 ±0.1480  | 138.7081 ±0.5939 | 24.8000     |
| bert_projected       | 13.9182 ±0.0294  | 139.2485 ±0.5356 | 14.9000     |
| bert_frozen_direct   | 14.1865 ±0.0123  | N/A              | 17.1000     |
| onehot_baseline      | 14.4458 ±0.1738  | N/A              | 18.2000     |

### MIG Disentanglement (β-VAE only)

| Condition            | Synthetic        | Real ESD         |
|----------------------|------------------|------------------|
| beta_vae_b1          | 0.0427 ±0.0200   | 0.0054 ±0.0022   |
| beta_vae_b2          | 0.0155 ±0.0184   | 0.0373 ±0.0442   |
| beta_vae_b4          | 0.0033 ±0.0023   | 0.0060 ±0.0029   |
| beta_vae_b8          | 0.0091 ±0.0029   | 0.0037 ±0.0015   |

## Acoustic Oracle (H3)

| Run       | Spearman ρ  | p-value    | Paper claim | Δ vs paper  |
|-----------|-------------|------------|-------------|-------------|
| Synthetic | 0.9680      | 0.0000     | 0.87        | +0.0980     |
| Real ESD  | 0.6111      | 0.0000     | 0.87        | -0.2589     |

## Hypothesis Verdicts

### H1: β increases → UTMOS decreases (Spearman ρ < -0.40)

| Run       | H1 Spearman ρ | Verdict        | Paper claim ρ |
|-----------|---------------|----------------|---------------|
| Synthetic | -1.0000       | SUPPORTED      | -0.94         |
| Real ESD  | -1.0000       | SUPPORTED      | -0.94         |

### H2: BERT-projected > one-hot baseline (Δ UTMOS ≥ 0.30)

| Run       | bert_projected | onehot | Δ UTMOS | Paper Δ | Verdict      |
|-----------|----------------|--------|---------|---------|--------------|
| Synthetic | 4.4308         | 4.4252 | 0.0056  | 0.42    | NOT SUPPORTED |
| Real ESD  | 3.6893         | N/A    | N/A     | 0.42    | N/A          |

### H3: Oracle Spearman ρ > 0.80

| Run       | Spearman ρ | Verdict        | Paper claim |
|-----------|------------|----------------|-------------|
| Synthetic | 0.9680     | SUPPORTED      | 0.87        |
| Real ESD  | 0.6111     | NOT SUPPORTED  | 0.87        |

## Validation Summary

> Δ = |experiment - paper|, threshold < 0.10 considered "within tolerance"

### Within tolerance (Δ < 0.10):
- (none)

### Discrepant (Δ ≥ 0.10):
- beta_vae_b1 (synthetic): exp=4.4202 paper=3.42 Δ=1.0002
- beta_vae_b1 (real): exp=3.7492 paper=3.42 Δ=0.3292
- beta_vae_b4 (synthetic): exp=4.4201 paper=3.18 Δ=1.2401
- beta_vae_b4 (real): exp=3.7489 paper=3.18 Δ=0.5689
- bert_projected (synthetic): exp=4.4308 paper=3.89 Δ=0.5408
- bert_projected (real): exp=3.6893 paper=3.89 Δ=0.2007
- onehot_baseline (synthetic): exp=4.4252 paper=3.47 Δ=0.9552

---
*Generated by compare_results.py*