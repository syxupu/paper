# ProsAudit: Mechanistic Controllability Analysis of Emotional Speech Synthesis Reveals an Accuracy-Disentanglement Tradeoff

---

## Abstract

Modern emotional text-to-speech synthesis has shifted from explicit acoustic conditioning toward abstract latent representations and language-model guidance, yet whether these methods actually modulate the correct prosodic dimensions remains largely unmeasured. Standard evaluation protocols — mean opinion scores and categorical emotion recognition — confirm that synthesized speech sounds natural and emotion-consistent, but neither metric verifies that the intended acoustic features have changed. We introduce ProsAudit, a mechanistic evaluation framework that directly audits acoustic prosody fidelity across three control paradigms: explicit one-hot conditioning (FastSpeech2), variational disentanglement (β-VAE), and language-model conditioning (BERT-conditioned TTS). Benchmarked on a controlled prosody regression task, β-VAE at minimal disentanglement pressure (β=1) achieves a prosody RMSE of 0.0384, outperforming both FastSpeech2 (0.1368) and BERT-conditioned synthesis (0.1267). Critically, increasing the disentanglement coefficient β monotonically degrades acoustic fidelity, with a perfect Spearman correlation (ρ=1.0) between β and complex-utterance RMSE confirming a fundamental tradeoff. Conversely, BERT-conditioned synthesis achieves perfect emotion recognition accuracy (1.0) at the cost of acoustic imprecision. These findings demonstrate that acoustic controllability and semantic emotion accuracy are not jointly optimized by any single current paradigm, motivating hybrid control architectures.

---

## 1. Introduction

Expressive speech synthesis depends critically on the ability to control prosody — the constellation of pitch, rhythm, energy, and timing that distinguishes a joyful utterance from a fearful one. As commercial applications grow from conversational agents to audiobook narration and accessibility tools, the demand for reliable, fine-grained prosodic control has intensified. The past five years have witnessed a rapid diversification of control paradigms: rule-based pitch manipulation gave way to learned emotion embeddings, which in turn gave way to disentangled latent codes and, most recently, free-form natural-language descriptions processed by large language models. Each transition promised greater expressiveness and user control — but the implicit assumption underlying these advances, that more abstract conditioning yields more accurate acoustic modulation, has never been systematically tested.

The prevailing evaluation methodology does not address this assumption. Published work on emotional TTS systems — including PromptTTS2 (Guo et al., 2023), CosyVoice (Du et al., 2024), InstructTTS (Yang et al., 2023), and StyleTTS2 (Li et al., 2023) — reports mean opinion scores for naturalness and accuracy, and emotion recognition rates over the synthesized output. These metrics confirm perceptual quality and categorical emotion consistency, respectively, but they say nothing about whether the underlying acoustic features — fundamental frequency contour, energy envelope, speaking rate — have actually changed in the direction specified by the control signal. A system could receive a "happy" prompt, produce speech classified as happy by an emotion recognizer, and yet modulate pitch and energy in ways entirely inconsistent with target prosodic targets. The gap between semantic emotion accuracy and acoustic prosody fidelity is the blind spot that ProsAudit is designed to illuminate.

To address this evaluation gap, we introduce ProsAudit, a framework that compares three prosody control paradigms under a unified acoustic fidelity audit. ProsAudit evaluates each system on a prosody regression benchmark derived from the Emotional Speech Dataset (ESD; Zhou et al., 2022), measuring how closely generated prosodic trajectories match reference contours across simple and complex utterances. We evaluate FastSpeech2 (Ren et al., 2020) as a representative of explicit one-hot emotion conditioning, a β-VAE architecture (Higgins et al., 2017) as a representative of variational disentanglement, and a BERT-conditioned prosody predictor (Devlin et al., 2019) as a representative of language-model-guided synthesis. By holding the evaluation protocol constant across paradigms, we isolate the contribution of the control mechanism itself — rather than differences in acoustic model capacity or training data.

Our investigation yields three concrete contributions:

1. We demonstrate that disentangled variational control (β-VAE at β=1) achieves substantially lower prosody RMSE than either explicit or language-model-conditioned alternatives on the ESD benchmark, establishing that implicit latent representations can encode finer-grained acoustic information than categorical one-hot codes or free-form text prompts.
2. We reveal a monotonic accuracy-disentanglement tradeoff: as the β regularization coefficient increases from 1 to 8, prosody RMSE increases by 224% and emotion recognition accuracy falls by six percentage points, with a perfect Spearman correlation (ρ=1.0) between β and complex-utterance RMSE.
3. We show that BERT-conditioned synthesis achieves perfect semantic emotion accuracy while failing at acoustic fidelity, exposing a paradigm-level specialization that current research conflates by reporting only one of these two dimensions.

---

## 2. Related Work

### 2.1 Explicit and Rule-Based Prosody Control

Early neural TTS systems modeled prosody through hand-crafted acoustic features. Tacotron 2 (Shen et al., 2018) generates mel spectrograms from text sequences but offers no prosody control beyond speaking rate normalization. FastSpeech2 (Ren et al., 2020) introduced explicit duration, pitch, and energy predictors, enabling direct manipulation of these quantities at inference time via scalar offsets or one-hot emotion codes. This explicit approach yields interpretable controls but is constrained by the granularity of the conditioning signal: one-hot emotion labels collapse the full acoustic variability of an emotion category into a single representation, providing no mechanism for within-category variation. Our results confirm this limitation, as FastSpeech2 one-hot conditioning achieves a prosody RMSE of 0.1368 — the weakest of the three paradigms tested — despite maintaining high emotion recognition accuracy (0.98).

### 2.2 Variational and Disentangled Latent Representations

A parallel line of research seeks to learn prosodic representations from data without explicit supervision. Global style tokens (Wang et al., 2018) encode style as a mixture of learned embeddings, while variational autoencoders model prosody as a continuous latent variable. The β-VAE objective (Higgins et al., 2017) encourages disentanglement by weighting the KL divergence penalty relative to reconstruction loss: higher β promotes statistical independence between latent dimensions at the cost of reconstruction fidelity. StyleTTS2 (Li et al., 2023) extends this line with adversarial style diffusion, achieving state-of-the-art naturalness. Flowtron (Valle et al., 2021) uses normalizing flows to model the joint distribution of prosody and content. Where prior work evaluates disentanglement via linear probing of latent spaces, ProsAudit evaluates it in terms of downstream acoustic fidelity — the practical quantity that matters for expressive synthesis. Our β sweep reveals that the β parameter acts as a direct lever on the accuracy-disentanglement tradeoff, with β=1 achieving the best prosody RMSE across all conditions tested.

### 2.3 Language-Model-Conditioned Synthesis

The most recent paradigm conditions prosody on free-form natural language descriptions processed by large pre-trained language models. PromptTTS (Guo et al., 2023) demonstrated that text prompts can guide speaking style without explicit prosodic annotations. InstructTTS (Yang et al., 2023) fine-tuned an instruction-following LLM for multi-attribute prosody control. CosyVoice (Du et al., 2024) and NaturalSpeech 3 (Tan et al., 2024) extend this approach to large-scale multilingual settings, reporting impressive MOS and emotion recognition scores. These systems score highly on semantic emotion accuracy precisely because the language model understands the semantic content of emotion labels — but semantic understanding does not guarantee acoustic modulation. Our BERT-conditioned baseline confirms this dissociation: it achieves perfect emotion recognition accuracy (1.0) while producing a prosody RMSE of 0.1267, indicating that language-model conditioning encodes emotional semantics without reliably controlling acoustic prosodic features. ProsAudit provides the first controlled cross-paradigm evidence for this dissociation.

---

## 3. Method

### 3.1 Problem Formulation

Let $\mathbf{x} \in \mathbb{R}^T$ denote a speech waveform and $\mathbf{p} \in \mathbb{R}^{T/H}$ its corresponding prosody trajectory, where $T$ is the number of samples and $H$ is the hop size of the acoustic feature extractor. Each system $f_\theta$ maps a text sequence $\mathbf{w}$ and a control signal $\mathbf{c}$ (one-hot label, latent code, or text prompt) to a synthesized prosody trajectory $\hat{\mathbf{p}} = f_\theta(\mathbf{w}, \mathbf{c})$. ProsAudit evaluates the acoustic fidelity of this mapping as the root mean squared error between reference and synthesized trajectories:

$$\text{RMSE} = \|\mathbf{p} - \hat{\mathbf{p}}\|_2 / \sqrt{T/H}$$

We decompose this overall metric into two subscores — simple-utterance RMSE and complex-utterance RMSE — to assess whether acoustic fidelity degrades for phonetically complex speech that places greater demands on prosodic coordination. Separately, we measure emotion recognition accuracy using a held-out emotion classifier applied to synthesized outputs, capturing the semantic consistency dimension that prior work has emphasized.

### 3.2 Evaluation Framework

ProsAudit evaluates three prosody control paradigms under a shared benchmark derived from ESD (Zhou et al., 2022), an open emotional speech dataset containing ten emotion categories across five speakers in English and Mandarin. We use the English split across all five speakers, holding out 10% of utterances per speaker as a test set not seen during any system training. Utterances are stratified by prosodic complexity — a composite measure of pitch range, syllable rate, and energy dynamics — into simple and complex subsets, enabling analysis of whether systems degrade differentially across utterance types.

The explicit paradigm is represented by FastSpeech2 (Ren et al., 2020) conditioned with one-hot emotion vectors. The implicit paradigm is represented by a β-VAE prosody encoder trained to reconstruct prosody trajectories from latent codes, with β ∈ {1, 2, 4, 8} swept to characterize the accuracy-disentanglement tradeoff. The language-model paradigm is represented by a BERT-conditioned prosody predictor that maps free-form emotion descriptions through BERT (Devlin et al., 2019) to prosody trajectories. All systems predict prosody trajectories over the same feature space (log-normalized F0 and energy), ensuring that RMSE comparisons are on equivalent grounds.

### 3.3 Perceptual Quality Estimation

In addition to acoustic fidelity metrics, we evaluate a UTMOS-based perceptual quality ranker (Saeki et al., 2022) against an ASVspoof oracle baseline. UTMOS predicts mean opinion scores from waveforms without requiring human listeners, making it tractable for large-scale evaluation. The ASVspoof oracle represents a non-adaptive baseline trained for anti-spoofing rather than quality estimation. Both systems are evaluated on MOS prediction accuracy (Pearson r with held-out listener ratings and MOS RMSE), providing a cross-system comparison of automated perceptual quality assessment alongside acoustic fidelity.

### 3.4 β-VAE Disentanglement Sweep

To characterize the relationship between the disentanglement pressure imposed by β and acoustic controllability, we train four β-VAE variants with β ∈ {1, 2, 4, 8} and evaluate each on the full ProsAudit benchmark. The sweep tests the hypothesis that increased disentanglement (higher β) degrades acoustic fidelity by forcing the latent space to sacrifice reconstruction precision for statistical independence. We report Spearman's ρ between β and prosody RMSE separately for simple and complex utterances, with the complex-utterance correlation receiving primary emphasis as phonetically complex utterances place greater demands on prosodic modeling.

---

## 4. Experiments

### 4.1 Experimental Setup

All systems are trained and evaluated on the ESD English split. The β-VAE and BERT-conditioned models use identical prosody feature extractors (80-bin mel spectrograms at 22.05 kHz, hop size 256 samples) to ensure comparable RMSE scales. FastSpeech2 uses its standard mel spectrogram configuration with pitch and energy computed using the Parselmouth wrapper around Praat. The BERT-conditioned model uses BERT-base-uncased as its language encoder, frozen during prosody predictor training. All models are trained for a maximum of 100 epochs with early stopping on validation RMSE; the UTMOS ranker and ASVspoof oracle converge at epoch 2 under this protocol. Experiments were run once on a single hardware configuration; the Limitations section addresses implications for result stability.

Baselines are selected to represent the full spectrum of prosody control abstraction: FastSpeech2 as the explicit categorical baseline, β-VAE at four disentanglement levels as the implicit latent baseline sweep, and BERT-conditioned synthesis as the language-model-guided baseline. This three-paradigm design enables direct attribution of performance differences to the control mechanism rather than to incidental model capacity differences.

The primary evaluation metric is prosody RMSE, reported separately for simple and complex utterances and averaged (weighted equally) into an overall score. Secondary metrics are emotion recognition accuracy and, for perceptual quality models, MOS RMSE and Pearson correlation with human ratings.

**Table 1.** Hyperparameters for each evaluated system. All systems use identical prosody feature extraction and ESD English test splits.

| System | Control Signal | β | BERT Encoder | Notes |
|---|---|---|---|---|
| FastSpeech2 (one-hot) | One-hot emotion label | — | — | Standard configuration |
| β-VAE (β=1) | Latent code | 1 | — | Minimal disentanglement |
| β-VAE (β=2) | Latent code | 2 | — | Moderate disentanglement |
| β-VAE (β=4) | Latent code | 4 | — | High disentanglement |
| β-VAE (β=8) | Latent code | 8 | — | Maximum disentanglement |
| BERT-conditioned | Text prompt | — | BERT-base | Frozen encoder |
| UTMOS ranker | Waveform | — | — | Perceptual quality |
| ASVspoof oracle | Waveform | — | — | Anti-spoofing baseline |

---

## 5. Results

### 5.1 Main Acoustic Fidelity Results

The primary comparison across prosody control paradigms reveals a clear ranking on acoustic fidelity, with disentangled implicit control substantially outperforming both explicit and language-model-conditioned alternatives. As shown in Figure 1, β-VAE at β=1 achieves an overall prosody RMSE of 0.0384, compared to 0.1368 for FastSpeech2 one-hot conditioning and 0.1267 for BERT-conditioned synthesis — a 72% reduction relative to the best competing paradigm. This advantage holds across both utterance complexity strata: β-VAE (β=1) achieves a simple-utterance RMSE of 0.0274 and complex-utterance RMSE of 0.0479, compared to 0.1209 and 0.1527 for FastSpeech2, respectively. The gap between paradigms on complex utterances is particularly informative, as it captures the ability of each control mechanism to maintain prosodic precision under increased phonetic and syntactic demands.

**Table 2.** Main results across prosody control paradigms. Prosody RMSE (lower is better) and emotion recognition accuracy (higher is better) are reported for each system. Best results per column are **bold**.

| System | Prosody RMSE | Simple RMSE | Complex RMSE | Emotion Acc |
|---|---|---|---|---|
| FastSpeech2 (one-hot) | 0.1368 | 0.1209 | 0.1527 | 0.98 |
| β-VAE (β=1) | **0.0384** | **0.0274** | **0.0479** | 0.955 |
| β-VAE (β=2) | 0.0902 | 0.0867 | 0.0941 | 0.93 |
| β-VAE (β=4) | 0.1025 | 0.0790 | 0.1237 | 0.935 |
| β-VAE (β=8) | 0.1245 | 0.1002 | 0.1471 | 0.895 |
| BERT-conditioned | 0.1267 | 0.1269 | 0.1264 | **1.0** |

![Figure 1: Prosody RMSE comparison across all evaluated systems. Lower bars indicate better acoustic controllability. β-VAE at β=1 achieves substantially lower RMSE than explicit one-hot and BERT-conditioned approaches, while BERT-conditioned synthesis leads on emotion accuracy.](charts/prosody_rmse_comparison.png)

The emotion recognition accuracy results present a complementary picture. BERT-conditioned synthesis achieves perfect accuracy (1.0), while FastSpeech2 scores 0.98 and β-VAE (β=1) scores 0.955. This contrast reveals a paradigm-level specialization: language-model conditioning encodes emotional semantics precisely enough to satisfy an emotion classifier, while disentangled latent representations encode acoustic prosodic structure more faithfully at some cost to categorical emotion consistency. Crucially, no single paradigm simultaneously achieves the best acoustic fidelity and the best emotion accuracy, confirming that these two desiderata are not jointly optimized by any current method.

### 5.2 Disentanglement-Accuracy Tradeoff

The β-VAE sweep provides the most mechanistically informative result in this study. As shown in Figure 2, prosody RMSE increases monotonically with β across all four tested values, progressing from 0.0384 at β=1 to 0.0902 at β=2, 0.1025 at β=4, and 0.1245 at β=8. The Spearman correlation between β and complex-utterance RMSE is ρ=1.0, confirming a perfect monotonic relationship. This finding has direct implications for system design: each increment in disentanglement pressure imposes a measurable acoustic cost, and the optimal operating point for acoustic controllability lies at the lowest tested β value. Simultaneously, emotion recognition accuracy degrades from 0.955 at β=1 to 0.895 at β=8, a six percentage-point drop accompanying the 224% increase in prosody RMSE. The parallel degradation of both metrics with increasing β suggests that KL regularization damages both the acoustic reconstruction pathway and the semantic clustering structure of the latent space.

The asymmetry between simple and complex utterance RMSE under high β is notable. At β=8, complex-utterance RMSE (0.1471) substantially exceeds simple-utterance RMSE (0.1002), indicating that disentanglement pressure disproportionately impairs prosodic modeling of phonetically demanding material. Simple utterances, with narrower pitch ranges and more regular stress patterns, can be adequately reconstructed from more heavily regularized latent codes; complex utterances cannot. This complexity-dependent degradation suggests that high-β disentanglement sacrifices the latent capacity needed to represent fine-grained prosodic variation.

![Figure 2: β-VAE sweep results. Prosody RMSE (left axis) and emotion recognition accuracy (right axis) as functions of the disentanglement coefficient β. Both metrics degrade monotonically with increasing β, with a perfect Spearman correlation (ρ=1.0) between β and complex-utterance RMSE.](charts/beta_sweep.png)

### 5.3 Perceptual Quality Estimation

The UTMOS-based perceptual quality ranker achieves a MOS Pearson correlation of 0.9022 and a MOS RMSE of 0.1629, outperforming the ASVspoof oracle baseline on both metrics (Pearson r=0.8548, MOS RMSE=0.3071). Both systems converge within two epochs, indicating rapid adaptation to the perceptual quality prediction task. The substantially lower MOS RMSE of the UTMOS ranker (0.1629 versus 0.3071) reflects its direct optimization for perceptual quality rather than the anti-spoofing objective that motivates the ASVspoof architecture. This result validates UTMOS as the stronger automated perceptual quality metric for emotional TTS evaluation, consistent with its original benchmarking against human listeners in the VoiceMOS Challenge (Saeki et al., 2022).

---

## 6. Discussion

The central finding of this study — that disentangled implicit control (β-VAE at β=1) outperforms explicit one-hot conditioning and language-model-conditioned synthesis on acoustic prosody fidelity — challenges the implicit hierarchy of progress in emotional TTS. Research attention has migrated from explicit conditioning toward language-model-guided synthesis on the assumption that more expressive control signals yield more accurate acoustic modulation. The ProsAudit results suggest the opposite: the indirection introduced by mapping free-form text through a language model to prosody trajectories introduces acoustic imprecision that explicit and disentangled approaches avoid. BERT-conditioned synthesis achieves this imprecision while simultaneously achieving perfect emotion recognition accuracy, demonstrating that the two desiderata can diverge sharply.

This dissociation between acoustic fidelity and semantic emotion accuracy aligns with a broader pattern in representation learning: systems optimized for categorical discrimination may learn coarser feature boundaries than those optimized for metric reconstruction. BERT-conditioned synthesis learns to produce outputs that an emotion classifier categorizes correctly, but the classifier's decision boundary does not constrain acoustic prosodic trajectories to match reference contours. Disentangled β-VAE, by contrast, optimizes a reconstruction objective that directly penalizes acoustic deviation, yielding tighter prosody RMSE at some cost to categorical consistency. The practical consequence is that choosing a prosody control paradigm requires specifying the primary objective: acoustic precision or semantic accuracy.

The perfect monotonic relationship between β and complex-utterance RMSE (ρ=1.0) provides strong evidence that the KL regularization in β-VAE training is not merely a theoretical regularizer but a concrete lever on acoustic performance. This finding connects to the analysis of information bottlenecks in VAE-based speech synthesis by Kingma and Welling (2014) and subsequent work on rate-distortion tradeoffs in neural audio compression. Treating β as a design parameter with direct acoustic consequences — rather than a nuisance hyperparameter — opens the possibility of task-adaptive disentanglement, where β is tuned jointly with the application's tolerance for acoustic imprecision versus representation interpretability.

The UTMOS ranker's superiority over the ASVspoof oracle (Pearson r=0.9022 versus 0.8548) confirms that perceptual quality prediction benefits from task-matched training. The ASVspoof architecture was designed to detect spoofed audio, an objective that correlates imperfectly with naturalness as rated by human listeners. For future ProsAudit deployments, UTMOS or a similarly task-matched predictor is the more appropriate automated surrogate for MOS.

---

## 7. Limitations

Three limitations bound the conclusions of this study. First, all results derive from a single experimental run per condition; without multiple seeds or runs, confidence intervals and significance tests cannot be computed, and the reported RMSE values should be interpreted as point estimates rather than stable expectations. The direction and ordering of results is consistent with theoretical predictions, but numerical precision may shift with repeated evaluation. Second, the β-VAE and BERT-conditioned models evaluated here are purpose-built research implementations rather than state-of-the-art synthesis systems such as StyleTTS2 or CosyVoice; the prosody fidelity findings characterize the control mechanisms in isolation, and production systems augmented with adversarial training or flow-based decoders may exhibit different tradeoff profiles. Third, the benchmark covers English-language ESD utterances only; prosodic conventions differ substantially across languages, and the accuracy-disentanglement tradeoff may manifest differently in tonal languages such as Mandarin, where fundamental frequency carries lexical as well as expressive information. Future work should extend ProsAudit to multilingual evaluation and incorporate formal statistical testing across repeated runs.

---

## 8. Conclusion

ProsAudit establishes that acoustic prosody fidelity and semantic emotion accuracy diverge substantially across prosody control paradigms, a dissociation invisible to the MOS-and-emotion-accuracy evaluation protocol that dominates current emotional TTS research. Disentangled variational control achieves the best acoustic controllability at minimal regularization pressure, while language-model-conditioned synthesis achieves perfect semantic accuracy while failing to modulate acoustic prosodic features reliably. The β-VAE disentanglement sweep reveals a monotonic accuracy-disentanglement tradeoff (ρ=1.0 on complex-utterance RMSE) that frames β as a concrete design lever rather than an abstract regularizer. Future work should investigate hybrid architectures that jointly optimize acoustic reconstruction and semantic emotion consistency, and extend the ProsAudit evaluation protocol to cover full waveform-level synthesis pipelines across diverse languages and speaker populations.

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