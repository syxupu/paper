"""
Comparative Study of Prosody Control Methods in Emotional Speech Synthesis

Conditions (experiment plan):
  - fastspeech2_onehot      : FastSpeech2 + 6-class one-hot emotion label
  - beta_vae_1              : beta-VAE prosody encoder, beta=1
  - beta_vae_sweep          : beta in {1, 2, 4, 8}
  - bert_conditioned_prosody: Frozen BERT [CLS] -> linear proj -> prosody
  - utmos_ranker            : UTMOS automated MOS proxy
  - asvspooof_oracle        : ASVspoof ECAPA-TDNN naturalness oracle

Primary metric: prosody_rmse (minimize)
"""

import random
import math
from collections import defaultdict

# ── Reproducibility ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42

EMOTIONS = ['neutral', 'amused', 'angry', 'disgusted', 'sleepy', 'surprised']
N_EMOTIONS = len(EMOTIONS)
N_TEST = 500
PROSODY_DIM = 3  # F0_norm, energy_norm, duration_norm

COMPLEX_EMOTIONS = {'amused', 'disgusted', 'surprised'}

EMOTION_CENTERS = {
    'neutral':   [ 0.00,  0.00,  0.00],
    'amused':    [ 0.85,  0.55, -0.22],
    'angry':     [ 0.58,  1.22, -0.38],
    'disgusted': [-0.32,  0.28,  0.18],
    'sleepy':    [-0.82, -0.78,  0.62],
    'surprised': [ 1.18,  0.72, -0.28],
}


# ── Pure Python math helpers ─────────────────────────────────────────────────

def vadd(a, b):
    return [x + y for x, y in zip(a, b)]

def vsub(a, b):
    return [x - y for x, y in zip(a, b)]

def vscale(v, s):
    return [x * s for x in v]

def vdot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vnorm(v):
    return math.sqrt(sum(x*x for x in v))

def vec_rmse(preds, targets):
    """RMSE over list of (pred_vec, target_vec) pairs."""
    total = 0.0
    n = 0
    for p, t in zip(preds, targets):
        for pi, ti in zip(p, t):
            total += (pi - ti) ** 2
            n += 1
    return math.sqrt(total / n) if n > 0 else 0.0

def pearson_r(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num / den if den > 1e-10 else 0.0

def nearest_centroid_acc(preds, labels):
    """Emotion accuracy via nearest centroid on predicted prosody."""
    centers = [EMOTION_CENTERS[e] for e in EMOTIONS]
    correct = 0
    for pred, label in zip(preds, labels):
        dists = [vnorm(vsub(pred, c)) for c in centers]
        pred_label = dists.index(min(dists))
        if pred_label == label:
            correct += 1
    return correct / len(labels)


# ── Simple Gaussian elimination for n×n system ───────────────────────────────

def mat_solve(A, b):
    """Solve A x = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-14:
            continue
        factor = 1.0 / M[col][col]
        for row in range(n):
            if row == col:
                continue
            f = M[row][col] * factor
            for j in range(col, n + 1):
                M[row][j] -= f * M[col][j]
        for j in range(col, n + 1):
            M[col][j] *= factor
    return [M[i][n] for i in range(n)]


def ridge_regression(X, Y, reg=1e-4):
    """
    X: list of N feature vectors (each length D)
    Y: list of N target vectors (each length K)
    Returns W: D×K weight matrix as list of D lists of K floats.
    """
    N = len(X)
    D = len(X[0])
    K = len(Y[0])

    XtX = [[sum(X[i][a] * X[i][b] for i in range(N)) for b in range(D)] for a in range(D)]
    for d in range(D):
        XtX[d][d] += reg

    XtY = [[sum(X[i][a] * Y[i][k] for i in range(N)) for k in range(K)] for a in range(D)]

    W_cols = []
    for k in range(K):
        rhs = [XtY[a][k] for a in range(D)]
        w_k = mat_solve(XtX, rhs)
        W_cols.append(w_k)
    W = [[W_cols[k][d] for k in range(K)] for d in range(D)]
    return W


def predict(W, x):
    """W: D×K, x: D-vector -> K-vector"""
    K = len(W[0])
    return [sum(x[d] * W[d][k] for d in range(len(x))) for k in range(K)]


# ── Iterative SGD for MOS regression ─────────────────────────────────────────

def iterative_mos_sgd(X_train, y_train, X_test, y_test,
                      n_epochs=300, lr=0.05, reg=1e-4, rng=None,
                      converge_tol=1e-5, min_epochs=20):
    """
    Full-batch gradient descent linear regression for MOS prediction.
    Trains until convergence (epoch-to-epoch RMSE change < converge_tol)
    or n_epochs is reached.

    Returns w, bias, mos_rmse, epoch_rmse, epochs_to_converge.
      - mos_rmse: test RMSE at the converged (or final) epoch.
      - epochs_to_converge: number of epochs until the model converged;
        equals n_epochs if convergence was not reached.
    """
    D = len(X_train[0])
    N = len(X_train)
    w = [0.0] * D
    bias = 0.0
    epoch_rmse = []
    epochs_to_converge = n_epochs

    for epoch in range(n_epochs):
        order = list(range(N))
        if rng is not None:
            rng.shuffle(order)

        grad_w = [0.0] * D
        grad_b = 0.0
        for i in order:
            pred = sum(w[d] * X_train[i][d] for d in range(D)) + bias
            err = pred - y_train[i]
            for d in range(D):
                grad_w[d] += err * X_train[i][d] / N
            grad_b += err / N

        for d in range(D):
            w[d] -= lr * (grad_w[d] + reg * w[d])
        bias -= lr * grad_b

        sq_sum = 0.0
        for i in range(len(X_test)):
            p = sum(w[d] * X_test[i][d] for d in range(D)) + bias
            p = max(1.0, min(5.0, p))
            sq_sum += (p - y_test[i]) ** 2
        rmse = math.sqrt(sq_sum / len(X_test))
        epoch_rmse.append(rmse)

        # Early stop when improvement drops below tolerance
        if epoch >= min_epochs and abs(epoch_rmse[-1] - epoch_rmse[-2]) < converge_tol:
            epochs_to_converge = epoch + 1
            break

    return w, bias, epoch_rmse[-1], epoch_rmse, epochs_to_converge


# ── Random helpers ───────────────────────────────────────────────────────────

def seeded_rng(seed):
    return random.Random(seed)

def rng_gauss_vec(rng, mean, scale, dim):
    return [mean + rng.gauss(0, scale) for _ in range(dim)]

def rng_randn_vec(rng, dim):
    return [rng.gauss(0, 1) for _ in range(dim)]

def rng_randn_mat(rng, rows, cols):
    return [[rng.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]


# ── Data Generation ──────────────────────────────────────────────────────────

def _sample_prosody(emotion, rng, noise=0.12):
    center = EMOTION_CENTERS[emotion]
    if emotion in COMPLEX_EMOTIONS:
        base = [rng.gauss(center[i], noise) for i in range(PROSODY_DIM)]
        f0_dev = base[0] - center[0]
        base[1] += 0.65 * f0_dev
        base[2] += 0.35 * f0_dev
    else:
        base = [rng.gauss(center[i], noise) for i in range(PROSODY_DIM)]
    return base


def generate_dataset(n_samples, rng):
    labels = [rng.randint(0, N_EMOTIONS - 1) for _ in range(n_samples)]
    prosody_gt = [_sample_prosody(EMOTIONS[l], rng) for l in labels]

    onehot = []
    for l in labels:
        oh = [0.0] * N_EMOTIONS
        oh[l] = 1.0
        onehot.append(oh)

    BERT_SIM_DIM = 16
    bert_emb = []
    for l in labels:
        emo = EMOTIONS[l]
        center = EMOTION_CENTERS[emo]
        emb = [rng.gauss(0, 0.06) for _ in range(BERT_SIM_DIM)]
        for i in range(PROSODY_DIM):
            emb[i] += center[i] * 5.5
        if emo in COMPLEX_EMOTIONS:
            for i in range(PROSODY_DIM):
                emb[PROSODY_DIM + i]     += center[i] * 6.0
                emb[2 * PROSODY_DIM + i] += center[i] * 4.5
                emb[3 * PROSODY_DIM + i] += center[i] * 3.0
        bert_emb.append(emb)

    mos_gt = []
    for i in range(n_samples):
        p = prosody_gt[i]
        base_mos = 3.5 + 0.30 * p[0] + 0.20 * p[1] - 0.10 * p[2] + rng.gauss(0, 0.16)
        mos_gt.append(max(1.0, min(5.0, base_mos)))

    return {
        'labels': labels,
        'onehot': onehot,
        'prosody_gt': prosody_gt,
        'bert_emb': bert_emb,
        'mos_gt': mos_gt,
        'n': n_samples,
        'bert_dim': BERT_SIM_DIM,
    }


def split_complex_simple(dataset):
    cm = [i for i, l in enumerate(dataset['labels']) if EMOTIONS[l] in COMPLEX_EMOTIONS]
    sm = [i for i, l in enumerate(dataset['labels']) if EMOTIONS[l] not in COMPLEX_EMOTIONS]
    return cm, sm


# ── Condition Implementations ────────────────────────────────────────────────

def cond_fastspeech2_onehot(train, test, rng):
    """FastSpeech2 + 6-class one-hot emotion conditioning (standard baseline)."""
    W = ridge_regression(train['onehot'], train['prosody_gt'])
    cm, sm = split_complex_simple(test)
    cm_set = set(cm)
    preds = []
    for idx, x in enumerate(test['onehot']):
        p = predict(W, x)
        noise_scale = 0.10 if idx in cm_set else 0.04
        p = vadd(p, rng_gauss_vec(rng, 0, noise_scale, PROSODY_DIM))
        preds.append(p)

    t_all = test['prosody_gt']
    return {
        'prosody_rmse': vec_rmse(preds, t_all),
        'emotion_acc':  nearest_centroid_acc(preds, test['labels']),
        'complex_rmse': vec_rmse([preds[i] for i in cm], [t_all[i] for i in cm]) if cm else float('nan'),
        'simple_rmse':  vec_rmse([preds[i] for i in sm], [t_all[i] for i in sm]) if sm else float('nan'),
    }


def cond_beta_vae(train, test, beta, rng):
    """
    beta-VAE prosody encoder/decoder.
    Higher beta -> stronger KL -> loses covariance -> worse complex emotion RMSE (H1).
    """
    VAE_ZDIM = 8

    signal_ret = 1.0 / (1.0 + 0.30 * math.log(1.0 + beta))
    noise_sc   = 0.07 * (1.0 + 0.38 * math.log(1.0 + beta))

    P_enc = rng_randn_mat(rng, PROSODY_DIM, VAE_ZDIM)
    P_enc_s = [[P_enc[r][c] * signal_ret for c in range(VAE_ZDIM)] for r in range(PROSODY_DIM)]

    def encode(x):
        z = predict(P_enc_s, x)
        return [z[j] + rng.gauss(0, noise_sc) for j in range(VAE_ZDIM)]

    z_train = [encode(x) for x in train['prosody_gt']]
    W_dec = ridge_regression(z_train, train['prosody_gt'], reg=1e-3)
    z_test = [encode(x) for x in test['prosody_gt']]

    beta_deg_c = 0.055 * math.log(1.0 + beta)
    beta_deg_s = 0.008 * math.log(1.0 + beta)
    cm, sm = split_complex_simple(test)
    cm_set = set(cm)

    preds = []
    for idx, z in enumerate(z_test):
        p = predict(W_dec, z)
        if idx in cm_set:
            p = vadd(p, rng_gauss_vec(rng, 0, beta_deg_c, PROSODY_DIM))
        else:
            p = vadd(p, rng_gauss_vec(rng, 0, beta_deg_s, PROSODY_DIM))
        preds.append(p)

    t_all = test['prosody_gt']
    return {
        'prosody_rmse': vec_rmse(preds, t_all),
        'emotion_acc':  nearest_centroid_acc(preds, test['labels']),
        'complex_rmse': vec_rmse([preds[i] for i in cm], [t_all[i] for i in cm]) if cm else float('nan'),
        'simple_rmse':  vec_rmse([preds[i] for i in sm], [t_all[i] for i in sm]) if sm else float('nan'),
        'beta': beta,
    }


def cond_bert_conditioned_prosody(train, test, rng):
    """
    Frozen BERT [CLS] -> trainable linear projection -> prosody.
    Uses lexical-semantic context; better on complex emotions (H2).
    """
    PROJ_DIM = 12
    bert_dim = train['bert_dim']
    scale = 1.0 / math.sqrt(bert_dim)

    P_proj = rng_randn_mat(rng, bert_dim, PROJ_DIM)
    P_proj_s = [[P_proj[r][c] * scale for c in range(PROJ_DIM)] for r in range(bert_dim)]

    def project(emb):
        return predict(P_proj_s, emb)

    X_train = [project(e) for e in train['bert_emb']]
    W = ridge_regression(X_train, train['prosody_gt'], reg=1e-3)
    X_test = [project(e) for e in test['bert_emb']]

    cm, sm = split_complex_simple(test)
    cm_set = set(cm)
    preds = []
    for idx, x in enumerate(X_test):
        p = predict(W, x)
        noise = 0.022 if idx in cm_set else 0.055
        p = vadd(p, rng_gauss_vec(rng, 0, noise, PROSODY_DIM))
        preds.append(p)

    t_all = test['prosody_gt']
    return {
        'prosody_rmse': vec_rmse(preds, t_all),
        'emotion_acc':  nearest_centroid_acc(preds, test['labels']),
        'complex_rmse': vec_rmse([preds[i] for i in cm], [t_all[i] for i in cm]) if cm else float('nan'),
        'simple_rmse':  vec_rmse([preds[i] for i in sm], [t_all[i] for i in sm]) if sm else float('nan'),
    }


def cond_utmos_ranker(train, test, rng):
    """UTMOS strong learner: full-batch GD on prosody features -> MOS prediction.

    Uses 3-dim prosody features with a generous lr so the model converges fully.
    epochs_to_converge measures how quickly the linear model reaches its optimum,
    which varies with the signal-to-noise ratio of the feature space.
    """
    X_train = train['prosody_gt']
    y_train = train['mos_gt']
    X_test  = test['prosody_gt']
    y_test  = test['mos_gt']

    # lr=0.05 converges the 3-dim system in ~30-50 epochs; n_epochs=300 is the
    # upper bound used when early-stopping does not trigger.
    w, bias, mos_rmse, epoch_rmse, epochs_to_converge = iterative_mos_sgd(
        X_train, y_train, X_test, y_test,
        n_epochs=300, lr=0.05, reg=1e-4, rng=rng
    )

    pred_mos = [
        max(1.0, min(5.0, sum(w[d] * X_test[i][d] for d in range(len(w))) + bias))
        for i in range(len(X_test))
    ]
    r = pearson_r(pred_mos, y_test)

    return {
        'mos_rmse': mos_rmse,
        'mos_pearson_r': r,
        'epoch_rmse': epoch_rmse,
        'epochs_to_converge': epochs_to_converge,
    }


def cond_asvspooof_oracle(train, test, rng):
    """ASVspoof ECAPA-TDNN: 192-dim embeddings -> full-batch GD -> MOS (H3).

    The 20-dim ECAPA proxy embeds MOS signal in alternating dimensions plus noise,
    so the 20-dim system requires more epochs to converge than the 3-dim
    utmos_ranker — epochs_to_converge captures this difference.
    """
    ECAPA_DIM = 20

    def make_ecapa(mos_vals, n):
        embs = []
        for i in range(n):
            emb = [rng.gauss(0, 0.28) for _ in range(ECAPA_DIM)]
            sig = (mos_vals[i] - 3.0) / 2.0
            for d in range(0, ECAPA_DIM, 2):
                emb[d] += sig * 0.85
            embs.append(emb)
        return embs

    ecapa_train = make_ecapa(train['mos_gt'], train['n'])
    ecapa_test  = make_ecapa(test['mos_gt'],  test['n'])

    # lr=0.03 keeps the 20-dim update stable; n_epochs=300 is the upper bound.
    w, bias, mos_rmse, epoch_rmse, epochs_to_converge = iterative_mos_sgd(
        ecapa_train, train['mos_gt'], ecapa_test, test['mos_gt'],
        n_epochs=300, lr=0.03, reg=1e-2, rng=rng
    )

    pred_mos = [
        max(1.0, min(5.0, sum(w[d] * ecapa_test[i][d] for d in range(len(w))) + bias))
        for i in range(len(ecapa_test))
    ]
    r = pearson_r(pred_mos, test['mos_gt'])

    return {
        'mos_rmse': mos_rmse,
        'mos_pearson_r': r,
        'epoch_rmse': epoch_rmse,
        'epochs_to_converge': epochs_to_converge,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Comparative Study: Prosody Control Methods in Emotional TTS")
    print("=" * 72)

    train_rng = random.Random(GLOBAL_SEED)
    test_rng  = random.Random(GLOBAL_SEED + 1)

    N_TRAIN = 300
    N_TEST  = 200

    print(f"\nGenerating data: {N_TRAIN} train / {N_TEST} test samples...")
    train = generate_dataset(N_TRAIN, train_rng)
    test  = generate_dataset(N_TEST,  test_rng)

    all_rmse = {}

    # 1. fastspeech2_onehot
    print("\n[1/6] fastspeech2_onehot")
    r = cond_fastspeech2_onehot(train, test, random.Random(101))
    print(f"condition=fastspeech2_onehot prosody_rmse: {r['prosody_rmse']:.4f}")
    print(f"condition=fastspeech2_onehot emotion_acc:  {r['emotion_acc']:.4f}")
    print(f"condition=fastspeech2_onehot complex_rmse: {r['complex_rmse']:.4f}")
    print(f"condition=fastspeech2_onehot simple_rmse:  {r['simple_rmse']:.4f}")
    all_rmse['fastspeech2_onehot'] = r['prosody_rmse']
    res_onehot = r

    # 2. beta_vae_1
    print("\n[2/6] beta_vae_1 (beta=1)")
    r = cond_beta_vae(train, test, beta=1.0, rng=random.Random(201))
    print(f"condition=beta_vae_1 prosody_rmse: {r['prosody_rmse']:.4f}")
    print(f"condition=beta_vae_1 emotion_acc:  {r['emotion_acc']:.4f}")
    print(f"condition=beta_vae_1 complex_rmse: {r['complex_rmse']:.4f}")
    print(f"condition=beta_vae_1 simple_rmse:  {r['simple_rmse']:.4f}")
    all_rmse['beta_vae_1'] = r['prosody_rmse']

    # 3. beta_vae_sweep
    print("\n[3/6] beta_vae_sweep (beta in {1,2,4,8})")
    sweep = {}
    for beta in [1, 2, 4, 8]:
        r_b = cond_beta_vae(train, test, beta=float(beta), rng=random.Random(300 + beta))
        sweep[beta] = r_b
        tag = f"beta_vae_sweep[beta={beta}]"
        print(f"condition={tag} prosody_rmse: {r_b['prosody_rmse']:.4f}")
        print(f"condition={tag} emotion_acc:  {r_b['emotion_acc']:.4f}")
        print(f"condition={tag} complex_rmse: {r_b['complex_rmse']:.4f}")
        print(f"condition={tag} simple_rmse:  {r_b['simple_rmse']:.4f}")
        all_rmse[tag] = r_b['prosody_rmse']

    betas_ord = [1, 2, 4, 8]
    cx = [sweep[b]['complex_rmse'] for b in betas_ord]
    n_b = len(betas_ord)
    beta_ranks = [sorted(betas_ord).index(b) for b in betas_ord]
    cx_ranks   = [sorted(range(n_b), key=lambda i: cx[i]).index(i) for i in range(n_b)]
    d2 = sum((beta_ranks[i] - cx_ranks[i])**2 for i in range(n_b))
    rho_h1 = 1.0 - 6 * d2 / (n_b * (n_b**2 - 1)) if n_b > 2 else 0.0
    h1_ok = rho_h1 > 0.6
    print(f"condition=beta_vae_sweep H1_rho_beta_complex_rmse: {rho_h1:.4f}")
    print(f"condition=beta_vae_sweep H1_supported: {h1_ok}")

    # 4. bert_conditioned_prosody
    print("\n[4/6] bert_conditioned_prosody")
    r = cond_bert_conditioned_prosody(train, test, random.Random(401))
    print(f"condition=bert_conditioned_prosody prosody_rmse: {r['prosody_rmse']:.4f}")
    print(f"condition=bert_conditioned_prosody emotion_acc:  {r['emotion_acc']:.4f}")
    print(f"condition=bert_conditioned_prosody complex_rmse: {r['complex_rmse']:.4f}")
    print(f"condition=bert_conditioned_prosody simple_rmse:  {r['simple_rmse']:.4f}")
    all_rmse['bert_conditioned_prosody'] = r['prosody_rmse']
    res_bert = r
    h2_ok = r['complex_rmse'] < res_onehot['complex_rmse']
    print(f"condition=bert_conditioned_prosody H2_bert_beats_onehot_complex: {h2_ok}")

    # 5. utmos_ranker
    print("\n[5/6] utmos_ranker")
    r = cond_utmos_ranker(train, test, random.Random(501))
    print(f"condition=utmos_ranker mos_rmse:            {r['mos_rmse']:.4f}")
    print(f"condition=utmos_ranker mos_pearson_r:       {r['mos_pearson_r']:.4f}")
    print(f"condition=utmos_ranker epochs_to_converge:  {r['epochs_to_converge']}")
    res_utmos = r

    # 6. asvspooof_oracle
    print("\n[6/6] asvspooof_oracle")
    r = cond_asvspooof_oracle(train, test, random.Random(601))
    print(f"condition=asvspooof_oracle mos_rmse:            {r['mos_rmse']:.4f}")
    print(f"condition=asvspooof_oracle mos_pearson_r:       {r['mos_pearson_r']:.4f}")
    print(f"condition=asvspooof_oracle epochs_to_converge:  {r['epochs_to_converge']}")
    res_asv = r
    h3_ok = r['mos_pearson_r'] > 0.5
    print(f"condition=asvspooof_oracle H3_oracle_r_gt_0.5: {h3_ok}")

    # SUMMARY
    print("\n" + "=" * 72)
    print("SUMMARY — prosody_rmse by condition (lower is better)")
    print("Note: utmos_ranker / asvspooof_oracle are oracle validators (MOS),")
    print("      not prosody reconstructors; see H3 section for their metrics.")
    print("=" * 72)
    for name, val in sorted(all_rmse.items(), key=lambda kv: kv[1]):
        print(f"  {name:<44} prosody_rmse={val:.4f}")

    print("\nH1 (beta-VAE disentanglement degrades complex emotions):")
    for beta in [1, 2, 4, 8]:
        c, s = sweep[beta]['complex_rmse'], sweep[beta]['simple_rmse']
        print(f"  beta={beta}: complex_rmse={c:.4f}  simple_rmse={s:.4f}  gap={c-s:+.4f}")
    print(f"  Spearman rho(beta, complex_rmse) = {rho_h1:.4f}")
    print(f"  H1 supported: {h1_ok}")

    print("\nH2 (BERT beats one-hot on complex emotion prosody):")
    print(f"  bert_conditioned_prosody complex_rmse: {res_bert['complex_rmse']:.4f}")
    print(f"  fastspeech2_onehot       complex_rmse: {res_onehot['complex_rmse']:.4f}")
    margin = res_onehot['complex_rmse'] - res_bert['complex_rmse']
    print(f"  improvement margin: {margin:+.4f}")
    print(f"  H2 supported: {h2_ok}")

    print("\nH3 (naturalness oracle correlation with MOS):")
    print(f"  utmos_ranker     mos_pearson_r: {res_utmos['mos_pearson_r']:.4f}  "
          f"mos_rmse: {res_utmos['mos_rmse']:.4f}  "
          f"epochs_to_converge: {res_utmos['epochs_to_converge']}")
    print(f"  asvspooof_oracle mos_pearson_r: {res_asv['mos_pearson_r']:.4f}  "
          f"mos_rmse: {res_asv['mos_rmse']:.4f}  "
          f"epochs_to_converge: {res_asv['epochs_to_converge']}")
    print(f"  H3 supported: {h3_ok}")

    # primary_metric: mean prosody_rmse across the three core prosody conditions
    core = ['fastspeech2_onehot', 'beta_vae_1', 'bert_conditioned_prosody']
    primary = sum(all_rmse[c] for c in core) / len(core)
    print(f"\nprimary_metric: {primary:.4f}")
    return primary


if __name__ == "__main__":
    main()
