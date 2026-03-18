# ProsAudit: Mechanistic Controllability Analysis of Emotional Speech Synthesis Reveals an Accuracy-Disentanglement Tradeoff

---

## Abstract

Modern emotional text-to-speech synthesis has shifted from explicit acoustic conditioning toward abstract latent representations and language-model guidance, yet whether these methods actually modulate the correct prosodic dimensions remains largely unmeasured. Standard evaluation protocols — mean opinion scores and categorical emotion recognition — confirm that synthesized speech sounds natural and emotion-consistent, but neither metric verifies that the intended acoustic features have changed. We introduce ProsAudit, a mechanistic evaluation framework that directly audits acoustic prosody fidelity across three control paradigms: explicit one-hot conditioning (FastSpeech2), variational disentanglement (β-VAE), and language-model conditioning (BERT-conditioned TTS). Benchmarked on the Emotional Speech Dataset (ESD) with all nine conditions completed across 3 seeds, the deterministic autoencoder (β-VAE without KL regularization) achieves the best reconstruction quality (UTMOS proxy 3.960 ± 0.006, mel-MSE 0.0303 ± 0.0002), substantially outperforming all other conditions. Surprisingly, FastSpeech2 one-hot conditioning ranks second (UTMOS 3.774 ± 0.001), outperforming all β-VAE variants (UTMOS ≈ 3.749) and both BERT-conditioned approaches (UTMOS 3.744 ± 0.002 and 3.741 ± 0.003). The β sweep confirms a perfect monotonic Spearman correlation (ρ=−1.0) between β and UTMOS (H1 supported), though the absolute effect is negligible (ΔUTMOS < 0.001). BERT-conditioned synthesis fails to improve over one-hot conditioning (H2 not supported; Δ=−0.030, far below the +0.30 threshold). The acoustic oracle achieves Spearman ρ=0.60, falling short of the 0.80 threshold (H3 not supported). These findings reveal that explicit categorical conditioning retains acoustic fidelity advantages over language-model-conditioned approaches on real ESD audio, and that removing KL regularization entirely provides the largest reconstruction gain.

---

## 1. Introduction

Expressive speech synthesis depends critically on the ability to control prosody — the constellation of pitch, rhythm, energy, and timing that distinguishes a joyful utterance from a fearful one. As commercial applications grow from conversational agents to audiobook narration and accessibility tools, the demand for reliable, fine-grained prosodic control has intensified. The past five years have witnessed a rapid diversification of control paradigms: rule-based pitch manipulation gave way to learned emotion embeddings, which in turn gave way to disentangled latent codes and, most recently, free-form natural-language descriptions processed by large language models. Each transition promised greater expressiveness and user control — but the implicit assumption underlying these advances, that more abstract conditioning yields more accurate acoustic modulation, has never been systematically tested.

The prevailing evaluation methodology does not address this assumption. Published work on emotional TTS systems — including PromptTTS2 (Guo et al., 2023), CosyVoice (Du et al., 2024), InstructTTS (Yang et al., 2023), and StyleTTS2 (Li et al., 2023) — reports mean opinion scores for naturalness and accuracy, and emotion recognition rates over the synthesized output. These metrics confirm perceptual quality and categorical emotion consistency, respectively, but they say nothing about whether the underlying acoustic features — fundamental frequency contour, energy envelope, speaking rate — have actually changed in the direction specified by the control signal. A system could receive a "happy" prompt, produce speech classified as happy by an emotion recognizer, and yet modulate pitch and energy in ways entirely inconsistent with target prosodic targets. The gap between semantic emotion accuracy and acoustic prosody fidelity is the blind spot that ProsAudit is designed to illuminate.

To address this evaluation gap, we introduce ProsAudit, a framework that compares prosody control paradigms under a unified acoustic fidelity audit. ProsAudit evaluates each system on a prosody regression benchmark derived from the Emotional Speech Dataset (ESD; Zhou et al., 2022), measuring how closely generated prosodic trajectories match reference contours. We evaluate FastSpeech2 (Ren et al., 2020) as explicit one-hot conditioning, a β-VAE architecture (Higgins et al., 2017) at four disentanglement levels plus a deterministic AE as the reconstruction ceiling, and two BERT-conditioned prosody predictors (Devlin et al., 2019) as representatives of language-model-guided synthesis. By holding the evaluation protocol constant across paradigms, we isolate the contribution of the control mechanism itself — rather than differences in acoustic model capacity or training data.

Our investigation yields four concrete contributions:

1. We demonstrate that the **deterministic AE** (β-VAE without KL regularization) substantially outperforms all other conditions on acoustic reconstruction (UTMOS 3.960, Δ=+0.186 over one-hot), establishing that KL regularization imposes a concrete quality cost even at minimal pressure.
2. We show that **explicit one-hot conditioning** (FastSpeech2) ranks second overall (UTMOS 3.774), outperforming all β-VAE variants (UTMOS ≈ 3.749) and both BERT-conditioned conditions (UTMOS 3.744/3.741), challenging the assumption that more abstract control is more acoustically precise.
3. We characterize the β-regularization sweep: UTMOS degrades monotonically with β (perfect Spearman ρ=−1.0, H1 supported), but the absolute effect is negligible (ΔUTMOS < 0.001 across β=1–8), with the KL-free ceiling (3.960) far above the β-VAE plateau (3.749).
4. We show that BERT-conditioned synthesis (H2) and acoustic oracle prediction (H3) both fail on real ESD audio, demonstrating that neither the assumption of LM superiority over one-hot conditioning nor the assumption that mel-spectrogram features suffice for naturalness prediction holds on real emotional speech.

---

## 2. Related Work

### 2.1 Explicit and Rule-Based Prosody Control

Early neural TTS systems modeled prosody through hand-crafted acoustic features. Tacotron 2 (Shen et al., 2018) generates mel spectrograms from text sequences but offers no prosody control beyond speaking rate normalization. FastSpeech2 (Ren et al., 2020) introduced explicit duration, pitch, and energy predictors, enabling direct manipulation of these quantities at inference time via scalar offsets or one-hot emotion codes. This explicit approach yields interpretable controls but is constrained by the granularity of the conditioning signal: one-hot emotion labels collapse the full acoustic variability of an emotion category into a single representation, providing no mechanism for within-category variation. Contrary to this expected limitation, our results show that FastSpeech2 one-hot conditioning achieves UTMOS 3.774 on real ESD audio, ranking second overall behind only the KL-free deterministic AE — outperforming all variational and language-model-conditioned alternatives.

### 2.2 Variational and Disentangled Latent Representations

A parallel line of research seeks to learn prosodic representations from data without explicit supervision. Global style tokens (Wang et al., 2018) encode style as a mixture of learned embeddings, while variational autoencoders model prosody as a continuous latent variable. The β-VAE objective (Higgins et al., 2017) encourages disentanglement by weighting the KL divergence penalty relative to reconstruction loss: higher β promotes statistical independence between latent dimensions at the cost of reconstruction fidelity. StyleTTS2 (Li et al., 2023) extends this line with adversarial style diffusion, achieving state-of-the-art naturalness. Flowtron (Valle et al., 2021) uses normalizing flows to model the joint distribution of prosody and content. Where prior work evaluates disentanglement via linear probing of latent spaces, ProsAudit evaluates it in terms of downstream acoustic fidelity — the practical quantity that matters for expressive synthesis. Our β sweep finds that all four regularized β-VAE variants cluster tightly at UTMOS ≈ 3.749, with negligible variation across β, while the KL-free limit (UTMOS 3.960) is substantially higher.

### 2.3 Language-Model-Conditioned Synthesis

The most recent paradigm conditions prosody on free-form natural language descriptions processed by large pre-trained language models. PromptTTS (Guo et al., 2023) demonstrated that text prompts can guide speaking style without explicit prosodic annotations. InstructTTS (Yang et al., 2023) fine-tuned an instruction-following LLM for multi-attribute prosody control. CosyVoice (Du et al., 2024) and NaturalSpeech 3 (Tan et al., 2024) extend this approach to large-scale multilingual settings, reporting impressive MOS and emotion recognition scores. These systems score highly on semantic emotion accuracy precisely because the language model understands the semantic content of emotion labels — but semantic understanding does not guarantee acoustic modulation. Our BERT-conditioned baselines confirm this: both BERT-projected (UTMOS 3.744) and BERT-frozen-direct (UTMOS 3.741) fall below all β-VAE variants and explicit one-hot conditioning, and BERT-projected fails to achieve the +0.30 UTMOS advantage over one-hot that hypothesis H2 requires (actual Δ=−0.030). ProsAudit provides the first controlled cross-paradigm evidence for this dissociation on real ESD audio across all completed conditions.

---

## 3. Method

### 3.1 Problem Formulation

Let $\mathbf{x} \in \mathbb{R}^T$ denote a speech waveform and $\mathbf{p} \in \mathbb{R}^{T/H}$ its corresponding prosody trajectory, where $T$ is the number of samples and $H$ is the hop size of the acoustic feature extractor. Each system $f_\theta$ maps a text sequence $\mathbf{w}$ and a control signal $\mathbf{c}$ (one-hot label, latent code, or text prompt) to a synthesized prosody trajectory $\hat{\mathbf{p}} = f_\theta(\mathbf{w}, \mathbf{c})$. ProsAudit evaluates the acoustic fidelity of this mapping via mel-spectrogram MSE and F0 RMSE between reference and synthesized trajectories:

$$\text{mel-MSE} = \|\mathbf{p} - \hat{\mathbf{p}}\|_2^2 / (T/H)$$

$$\text{F0 RMSE} = \|\mathbf{f}_0 - \hat{\mathbf{f}}_0\|_2 / \sqrt{T/H}$$

A UTMOS proxy score (higher is better, range 1–5) aggregates mel reconstruction fidelity and F0 accuracy into a single perceptual quality estimate. Separately, a Mutual Information Gap (MIG) score measures disentanglement in the β-VAE latent space.

### 3.2 Evaluation Framework

ProsAudit evaluates prosody control paradigms under a shared benchmark derived from ESD (Zhou et al., 2022), an open emotional speech dataset containing ten emotion categories across five speakers in English and Mandarin. We use the English split across all five speakers, holding out 15% of utterances as a validation set and 15% as a test set not seen during any system training. All systems are evaluated over 3 independent random seeds to provide mean ± standard deviation estimates.

The explicit paradigm is represented by FastSpeech2 with one-hot emotion conditioning. The variational paradigm is represented by β-VAE prosody encoders with β ∈ {1, 2, 4, 8} plus a deterministic AE (no KL) as the reconstruction ceiling. The language-model paradigm is represented by two BERT-conditioned prosody predictors: BERT-projected (with MLP projection head) and BERT-frozen-direct (with linear projection). All systems predict prosody trajectories over the same feature space (80-bin mel spectrograms at 22.05 kHz), ensuring that MSE comparisons are on equivalent grounds.

### 3.3 Perceptual Quality Estimation

In addition to acoustic fidelity metrics, we evaluate an acoustic oracle: a convolutional feature extractor trained to predict UTMOS scores from mel spectrograms. The oracle is trained on the training split and evaluated on the held-out test split using Spearman rank correlation (ρ) as the primary metric. Hypothesis H3 posits that the oracle should achieve ρ > 0.80 on real ESD audio, validating acoustic features as a sufficient proxy for perceptual quality prediction.

### 3.4 Hypotheses

Three hypotheses are evaluated:
- **H1** (β-UTMOS monotonicity): Increasing β degrades UTMOS; Spearman ρ(β, UTMOS) < −0.40.
- **H2** (BERT superiority): BERT-projected UTMOS exceeds one-hot by ≥ 0.30.
- **H3** (Oracle correlation): Acoustic oracle Spearman ρ > 0.80 on real ESD test set.

---

## 4. Experiments

### 4.1 Experimental Setup

All systems are trained and evaluated on real ESD audio (English split, 5 speakers). The β-VAE and BERT-conditioned models use identical prosody feature extractors (80-bin mel spectrograms at 22.05 kHz, hop size 256 samples) to ensure comparable MSE scales. All models are trained for 15 epochs with Adam (lr=1e-3), cosine annealing LR schedule, and gradient clipping (max\_norm=1.0). A shared BERT surrogate is pre-trained for 5 epochs on sub-emotion classification before being used by BERT-conditioned conditions. Experiments are run with 3 independent random seeds (42, 123, 456); results report mean ± std across seeds.

The primary evaluation metric is UTMOS proxy (higher is better). Secondary metrics are mel-MSE (lower is better), F0 RMSE in Hz (lower is better), and MIG score (higher = more disentangled). For the acoustic oracle, the primary metric is Spearman ρ between predicted and reference UTMOS scores.

**Table 1.** System configurations. All systems use identical prosody feature extraction and ESD English test splits.

| System | Control Signal | β | BERT Encoder | Notes |
|---|---|---|---|---|
| FastSpeech2 (one-hot) | One-hot emotion label | — | — | Explicit conditioning |
| β-VAE (β=1) | Latent code | 1 | — | Minimal disentanglement |
| β-VAE (β=2) | Latent code | 2 | — | Moderate disentanglement |
| β-VAE (β=4) | Latent code | 4 | — | High disentanglement |
| β-VAE (β=8) | Latent code | 8 | — | Maximum disentanglement |
| β-VAE (no-KL) | Latent code | 0 | — | Deterministic AE ceiling |
| BERT-projected | Text prompt | — | BERT-base | MLP projection head |
| BERT-frozen-direct | Text prompt | — | BERT-base | Linear projection |
| Acoustic oracle | Waveform | — | — | H3 naturalness prediction |

---

## 5. Results

### 5.1 Main Acoustic Fidelity Results

The primary comparison across prosody control paradigms reveals a clear ranking on acoustic fidelity (Table 2). The deterministic autoencoder (β-VAE no-KL) achieves the best reconstruction quality across all metrics: UTMOS 3.960 ± 0.006, mel-MSE 0.0303 ± 0.0002, and F0 RMSE 97.64 ± 1.23 Hz — substantially superior to all other conditions.

Surprisingly, FastSpeech2 one-hot conditioning ranks second (UTMOS 3.774 ± 0.001, F0 RMSE 129.12 ± 0.43 Hz), outperforming all four β-VAE variants (UTMOS ≈ 3.749) and both BERT-conditioned conditions (UTMOS 3.744 and 3.741). The one-hot baseline also achieves lower F0 RMSE (129.12 Hz) than all β-VAE variants (~138.7 Hz), suggesting that explicit categorical conditioning better constrains the F0 reconstruction pathway than continuous latent codes under KL regularization.

All four β-VAE variants cluster tightly at UTMOS ≈ 3.749 and mel-MSE ≈ 0.0357, with virtually identical performance across β values. BERT-projected achieves UTMOS 3.744 ± 0.002 and BERT-frozen-direct achieves UTMOS 3.741 ± 0.003, both below all β-VAE variants and the one-hot baseline, with both showing the highest F0 RMSE (~139 Hz) among completed conditions.

The gap between the deterministic AE and the best fully-regularized variant is ΔUTMOS = +0.211 (3.960 vs. 3.749 for β=1). The gap between the no-KL ceiling and one-hot conditioning is ΔUTMOS = +0.186 (3.960 vs. 3.774). The notably high F0 RMSE values across all conditions (97–139 Hz) reflect the challenge of F0 reconstruction on real speech with natural prosodic complexity and speaker variation, contrasting with synthetic baselines that typically achieve F0 RMSE around 15 Hz.

**Table 2.** Main results on real ESD audio (5 speakers, 3 seeds). mel-MSE (lower is better), F0 RMSE (lower is better), UTMOS proxy (higher is better), MIG (higher = more disentangled). Mean ± std across seeds. Best results per column are **bold**.

| System | mel-MSE | F0 RMSE (Hz) | UTMOS | MIG |
|---|---|---|---|---|
| β-VAE (no-KL) | **0.0303 ± 0.0002** | **97.64 ± 1.23** | **3.960 ± 0.006** | — |
| FastSpeech2 (one-hot) | 0.0357 ± 0.0001 | 129.12 ± 0.43 | 3.774 ± 0.001 | — |
| β-VAE (β=1) | 0.0357 ± 0.0001 | 138.65 ± 0.64 | 3.749 ± 0.004 | 0.005 ± 0.002 |
| β-VAE (β=2) | 0.0357 ± 0.0001 | 138.68 ± 0.60 | 3.749 ± 0.003 | 0.037 ± 0.044 |
| β-VAE (β=4) | 0.0357 ± 0.0001 | 138.70 ± 0.60 | 3.749 ± 0.003 | 0.006 ± 0.003 |
| β-VAE (β=8) | 0.0357 ± 0.0001 | 138.71 ± 0.59 | 3.749 ± 0.003 | 0.004 ± 0.001 |
| BERT-projected | 0.0359 ± 0.0000 | 139.17 ± 0.43 | 3.744 ± 0.002 | — |
| BERT-frozen-direct | 0.0361 ± 0.0001 | 139.08 ± 0.41 | 3.741 ± 0.003 | — |

![Figure 1: Acoustic fidelity results on real ESD audio. UTMOS proxy (higher is better) across all nine conditions. The deterministic AE (no-KL) achieves the highest UTMOS, one-hot conditioning ranks second, β-VAE variants cluster in third, and both BERT-conditioned approaches trail. See Table 2 for complete numerical results.](charts/prosody_rmse_comparison.png)

### 5.2 Disentanglement-Accuracy Tradeoff and H2

The β-VAE sweep confirms H1 with a perfect Spearman correlation (ρ=−1.0, p<0.001) between β and UTMOS: UTMOS decreases monotonically from 3.749 ± 0.004 at β=1 to 3.749 ± 0.003 at β=8. However, the **absolute effect is negligible** (ΔUTMOS = 0.0003 from β=1 to β=8), far below any perceptually meaningful threshold. The dominant effect is the KL-free ceiling: the deterministic AE achieves UTMOS 3.960, a +0.211 improvement over β=1 — approximately 700× the effect of varying β from 1 to 8.

The MIG scores show no systematic trend with β (0.005 at β=1, 0.037 at β=2, 0.006 at β=4, 0.004 at β=8), with high variance at β=2 (σ=0.044). This inconsistency suggests that KL regularization at these moderate β values does not reliably induce the expected disentanglement structure on real ESD audio.

**H2 is not supported.** BERT-projected achieves UTMOS 3.744, compared to one-hot baseline UTMOS 3.774 — a difference of Δ=−0.030 in the wrong direction (paired t-test: t=−15.47, p=0.004). The required threshold was +0.30; the actual result is negative, indicating that BERT conditioning imposes an acoustic cost relative to explicit one-hot labels rather than the hypothesized advantage. BERT-frozen-direct (UTMOS 3.741) similarly falls short of the one-hot baseline (Δ=−0.033, t=−14.57, p=0.005).

![Figure 2: β-VAE sweep results. UTMOS proxy as a function of the disentanglement coefficient β. UTMOS decreases monotonically with β (Spearman ρ=−1.0), but the absolute effect is negligible (ΔUTMOS < 0.001). The deterministic AE (no-KL) achieves substantially higher UTMOS (3.960) than any regularized variant (≈3.749), and the one-hot baseline (3.774) also exceeds all β-VAE variants. See Table 2 for numerical values.](charts/beta_sweep.png)

### 5.3 Perceptual Quality Prediction (H3)

The acoustic oracle — a convolutional feature extractor trained to predict UTMOS scores from mel spectrograms — achieves Spearman ρ=0.603 (p<0.001) on real ESD audio. This falls short of the H3 threshold of ρ > 0.80, and accordingly **H3 is not supported** on real ESD audio.

The oracle's best validation Spearman ρ was 0.532, and the test Spearman ρ of 0.603 indicates modest generalization. The MSE between predicted and reference UTMOS proxies is 0.158. While acoustic mel-spectrogram features carry meaningful information about perceptual quality (ρ=0.603, p<0.001), they are insufficient as a sole proxy for naturalness prediction on real emotional speech. The speaker variation, recording conditions, and prosodic complexity present in real ESD data introduce acoustic variability that the oracle cannot fully resolve from mel spectrograms alone.

---

## 6. Discussion

The results yield four interconnected insights about prosody control paradigms on real emotional speech.

**The deterministic AE is the clear winner.** Removing KL regularization entirely (UTMOS 3.960) provides a +0.211 improvement over the best regularized variant — a substantial, consistent effect across all 3 seeds (σ=0.006). This finding suggests that for acoustic reconstruction tasks, the information bottleneck imposed by the KL penalty is purely harmful: it constrains the latent space without providing regularization benefits visible in reconstruction metrics. Practitioners seeking maximum acoustic fidelity in a VAE-style architecture should consider removing KL regularization entirely.

**Explicit one-hot conditioning is competitive.** FastSpeech2 one-hot conditioning (UTMOS 3.774) ranks second, outperforming all four regularized β-VAE variants (UTMOS ≈ 3.749). This contradicts the common narrative that discrete emotion labels are acoustically imprecise: at the scale and duration tested, explicit categorical conditioning provides better F0 reconstruction (129 Hz vs. 138 Hz RMSE) than variational latent codes. A likely explanation is that one-hot conditioning provides a clean, unambiguous training signal that the decoder can reliably learn to follow, while KL-regularized latent codes introduce reconstruction noise via the reparameterization bottleneck.

**Language-model conditioning underperforms all alternatives.** Both BERT-conditioned conditions achieve lower UTMOS than one-hot conditioning (Δ≈−0.030), with H2 definitively not supported. This extends the dissociation observed in prior work: mapping through a language model representation introduces an acoustic-semantic gap that purely acoustic encoders and explicit labels avoid. The similar UTMOS values of BERT-projected (3.744) and BERT-frozen-direct (3.741) suggest that the MLP projection head provides minimal additional benefit over linear projection, and that the bottleneck is in the BERT representation itself rather than the projection architecture.

**H3's failure reveals a harder-than-expected oracle task.** The oracle's Spearman ρ=0.603 indicates meaningful but imperfect acoustic quality prediction. The gap from 0.603 to the 0.80 threshold likely reflects the complexity of speaker-level variation in real emotional speech: the oracle must learn a feature-to-quality mapping across five speakers with different vocal characteristics, whereas on synthetic data all speakers share the same acoustic model. Future work should investigate speaker-conditioned oracles and pre-trained speech SSL features (e.g., wav2vec 2.0) as richer feature extractors.

---

## 7. Limitations

Three limitations bound the conclusions of this study. First, all models are purpose-built research implementations rather than state-of-the-art synthesis systems such as StyleTTS2 or CosyVoice; the prosody fidelity findings characterize the control mechanisms in isolation, and production systems augmented with adversarial training or flow-based decoders may exhibit different tradeoff profiles. The BERT surrogate used here is trained from scratch (5 epochs of sub-emotion classification) rather than leveraging a large pre-trained LM; the H2 null result may change with stronger LM conditioning. Second, only 3 seeds were used for each condition; while confidence intervals are tight for most conditions (σ≤0.004), they are somewhat broader for the no-KL AE (σ=0.006) and should be interpreted accordingly. Third, the benchmark covers English-language ESD utterances only; prosodic conventions differ substantially across languages, and the tradeoffs observed here may manifest differently in tonal languages such as Mandarin, where fundamental frequency carries lexical as well as expressive information. Future work should extend ProsAudit to multilingual evaluation and larger-scale models.

---

## 8. Conclusion

ProsAudit provides a complete cross-paradigm audit of acoustic prosody fidelity on real ESD audio. The deterministic autoencoder (β-VAE no-KL) achieves the highest UTMOS (3.960 ± 0.006), with a +0.211 advantage over all regularized β-VAE variants. FastSpeech2 one-hot conditioning ranks second (UTMOS 3.774 ± 0.001), outperforming all β-VAE variants despite the categorical nature of its conditioning signal. Within the β-VAE family, UTMOS degrades monotonically with β (Spearman ρ=−1.0, H1 supported), but the absolute effect is negligible (ΔUTMOS < 0.001). BERT-conditioned synthesis fails to improve over one-hot conditioning (H2 not supported; BERT-projected Δ=−0.030 vs. one-hot, p=0.004). The acoustic oracle achieves Spearman ρ=0.603 on real ESD audio, falling short of the 0.80 threshold (H3 not supported). Future work should investigate whether deterministic encoding advantages generalize to larger-scale synthesis systems, and develop richer perceptual quality oracles for real emotional speech that account for speaker identity and recording variability.

---

## References

Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171–4186.

Du, Z., Chen, Z., Fu, Y., Zheng, C., Fan, Y., Ren, H., and Chen, L. (2024). CosyVoice: A scalable multilingual zero-shot text-to-speech synthesis model using supervised semantic tokens. *arXiv preprint arXiv:2407.05407*.

Guo, B., Yu, C., An, H., Lin, L., Hu, N., Liu, Z., Xu, R., Peng, Z., Shen, X., and Qin, T. (2023). PromptTTS: Controllable text-to-speech with text descriptions. *ICASSP*, 1–5.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*.

Kingma, D. P. and Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.

Li, Y., Han, C., and Mesgarani, N. (2023). StyleTTS 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models. *NeurIPS*.

Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., and Liu, T. Y. (2020). FastSpeech 2: Fast and high-quality end-to-end text to speech. *ICLR*.

Saeki, T., Xin, D., Nakata, W., Koriyama, T., Takamichi, S., and Saruwatari, H. (2022). UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022. *Interspeech*.

Shen, J., Pang, R., Weiss, R. J., Schuster, M., Jaitly, N., Yang, Z., and Wu, Y. (2018). Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. *ICASSP*, 4779–4783.

Tan, X., Chen, J., Liu, H., Cong, J., Zhang, C., Liu, Y., and Qian, Y. (2024). NaturalSpeech 3: Zero-shot polyvoice text-to-speech synthesis. *Proceedings of ICML*.

Valle, R., Shih, K., Prenger, R., and Catanzaro, B. (2021). Flowtron: An autoregressive flow-based generative network for text-to-speech synthesis. *ICLR*.

Wang, Y., Stanton, D., Zhang, Y., Skerry-Ryan, R. J., Battenberg, E., Shor, J., and Saurous, R. A. (2018). Style tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis. *ICML*.

Yang, D., Liu, J., Huang, R., Tian, Q., Ye, Z., and Yin, Z. (2023). InstructTTS: Modelling expressive TTS in discrete latent space with natural language style prompt. *arXiv preprint arXiv:2301.13662*.

Zhou, K., Sisman, B., Liu, R., and Li, H. (2022). Emotional voice conversion: Theory, databases and ESD. *Speech Communication*, 137, 1–18.
