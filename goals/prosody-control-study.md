# Research Goal: Prosody Control Methods in Emotional Speech Synthesis

---

## Topic

Comparative evaluation of prosody control paradigms in emotional text-to-speech (TTS) synthesis — specifically examining whether different control methods (explicit, implicit, and LLM-conditioned) modulate the *acoustically correct* prosodic features associated with target emotions.

---

## Novel Angle

### What has NOT been well-studied

The field now contains at least three distinct paradigms for prosody control in emotional TTS:

1. **Explicit control** — directly conditioning on predicted pitch (F0), energy, and duration sequences (FastSpeech2-style, 2021–2022)
2. **Implicit/reference-encoder control** — learning a style embedding from a reference utterance (GST, VAE-based; Tacotron-GST, 2018–2022)
3. **LLM/natural-language conditioned** — using text prompts ("speak with angry, fast pace") via a language model encoder to condition synthesis (InstructTTS 2023, PromptTTS2 ICLR 2024, CosyVoice 2024)

Each paradigm is individually well-studied. However, **there is no systematic cross-paradigm study that evaluates whether these methods actually produce the expected *acoustic-level* prosodic signatures for each emotion** — as opposed to merely producing perceptually "different" or "emotional-sounding" speech.

Existing comparisons rely almost exclusively on:
- Mean Opinion Scores (MOS) for naturalness — a holistic perceptual judgment
- Downstream emotion classifier accuracy — a black-box label-match, not a mechanistic check

**The gap**: No paper has asked "when I say *synthesize angry speech*, does each paradigm actually modulate F0 range, speaking rate, and energy level in the directions known from the phonetics literature on emotional prosody — and by how much, and how precisely?"

This matters because:
- LLM-conditioned systems could produce high emotion-classifier accuracy by learning spurious acoustic shortcuts (e.g., slight pitch elevation) without producing the full prosodic profile of anger (high F0, wide range, increased energy, faster rate)
- Explicit control may be more acoustically faithful but perceptually less natural
- The *controllability-naturalness Pareto frontier* across paradigms has never been mapped

### Why this is timely NOW (2024–2026)

Three developments create this opportunity simultaneously:

1. **LLM-TTS proliferation**: The year 2024 saw a wave of LLM-conditioned TTS systems (CosyVoice, PromptTTS2, InstructTTS, VoiceCraft) all claiming "natural language prosody control." The claims are untested at the acoustic feature level.
2. **Open-source accessibility**: Pre-trained checkpoints for StyleTTS2, CosyVoice, and FastSpeech2 are publicly available — a fair comparison is now feasible without training.
3. **Acoustic analysis tooling maturity**: Parselmouth (Python/Praat wrapper), openSMILE, and wav2vec2-based emotion classifiers now make automated acoustic feature extraction and emotion recognition straightforward at scale.

### How this differs from standard approaches

Standard approaches either:
- Train a new model and compare its MOS to prior work (single-paradigm contribution), or
- Conduct listener studies without acoustic grounding

This work conducts a **mechanistic controllability audit**: for each paradigm and each emotion, it measures *which acoustic dimensions change, by how much, and whether the direction matches phonetic ground truth for that emotion* — decoupling "sounds emotional" from "is acoustically emotional."

---

## Scope

A single-paper study covering:
- **3 paradigms**: explicit (FastSpeech2 + emotion label), implicit (StyleTTS2 or GST-Tacotron with reference), LLM-conditioned (CosyVoice or InstructTTS)
- **4–6 emotions**: angry, happy, sad, neutral (core), optionally fearful, surprised
- **1 benchmark**: ESD (Emotional Speech Dataset, English subset)
- **Dual evaluation axis**: (1) acoustic feature profiles vs. phonetic ground truth, (2) perceptual naturalness via UTMOS + human MOS
- **No model training** — inference-only comparison using publicly released checkpoints

Out of scope: cross-lingual transfer, speaker adaptation, real-time synthesis, multilingual evaluation.

---

## SMART Goal

**Specific**: Conduct an inference-only, cross-paradigm comparison of three emotional TTS prosody control methods (explicit token-based, implicit reference-encoder, LLM-conditioned), measuring both acoustic prosody fidelity (F0, energy, speaking rate per emotion vs. phonetic expectations) and perceptual naturalness (UTMOS, emotion classifier accuracy) on the ESD benchmark.

**Measurable**: For each system × emotion combination, report:
- F0 mean and range (Hz), energy (dB RMS), speaking rate (syllables/sec) — compared to ESD ground-truth reference values
- Prosody direction accuracy: % of emotions where the system's acoustic profile matches the expected phonetic direction (e.g., anger → higher F0, higher energy, faster rate)
- UTMOS naturalness score (0–5)
- Emotion classification accuracy (%) using a held-out wav2vec2-based emotion classifier
- Controllability precision score: standard deviation of acoustic features across 50 synthesized utterances per emotion (lower = more consistent/controllable)

**Achievable**: All models have publicly available pre-trained checkpoints. ESD is freely downloadable. Acoustic analysis uses Parselmouth (open-source). UTMOS is a pre-trained MOS predictor. Total compute: ~6–12 GPU-hours on a single A100 or RTX 3090 (inference only).

**Relevant**: As LLM-conditioned TTS becomes dominant, the community needs to understand whether natural language control actually produces acoustically grounded emotional prosody — or merely emotion-classifier-detectable artifacts. This directly informs future system design and evaluation protocol design.

**Time-bound**: 8 weeks total.
- Weeks 1–2: Environment setup, model checkpoints, ESD preprocessing, baseline acoustic feature extraction from ground truth
- Weeks 3–4: Run inference for all three systems across all emotions, extract acoustic features
- Weeks 5–6: UTMOS scoring, emotion classification, statistical analysis, controllability precision metrics
- Weeks 7–8: Write-up, figures, ablation (e.g., prompt phrasing sensitivity for LLM-conditioned system)

---

## Constraints

- **Compute**: Single GPU (NVIDIA RTX 3090 24GB or A100 40GB), inference-only — no training
- **Data**: ESD English subset (~14,000 utterances, freely available at [github.com/HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data))
- **Models**: Open-source pre-trained checkpoints only
  - FastSpeech2: espnet or coqui-ai pre-trained
  - StyleTTS2: official repo (ljspeech/LibriTTS checkpoints)
  - CosyVoice or InstructTTS: official Alibaba/open-source release
- **Tooling**: Parselmouth (F0/energy/duration), openSMILE (feature extraction), UTMOS (MOS predictor), wav2vec2-emotion (SER classifier)
- **No proprietary APIs**: All evaluation must be reproducible with open-source tools

---

## Success Criteria

**Publishable at Interspeech 2025 / ICASSP 2026 / SLT 2026** if the study demonstrates:

1. **Quantitative differentiation**: The three paradigms produce statistically significantly different acoustic profiles (ANOVA p < 0.01) for at least 3 of 4 emotions on at least 2 acoustic dimensions (F0 range, energy, rate)

2. **Mechanistic insight**: At least one non-obvious finding — e.g., LLM-conditioned systems achieve high emotion classifier accuracy (~85%) but low prosody direction accuracy (<60%) for certain emotions, revealing a gap between "sounds emotional" and "is acoustically correct"

3. **Actionable recommendation**: A clear, evidence-based recommendation for when each paradigm should be preferred (e.g., explicit for prosody precision, LLM for naturalness, implicit for reference-based transfer)

4. **Reproducibility**: Full code and evaluation pipeline released publicly, enabling future comparisons

**Minimum bar**: MOS-N ≥ 3.8 for all systems under test (confirming they produce intelligible speech), plus at least one cross-paradigm metric where the difference is statistically significant.

---

## Benchmark

### Primary: ESD — Emotional Speech Dataset

| Property | Detail |
|---|---|
| **Name** | ESD (Emotional Speech Dataset) |
| **Source** | Zhou et al., 2022 — [github.com/HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| **Size** | 350 utterances × 10 emotions × 10 speakers (EN+ZH); use English subset (5 speakers) |
| **Emotions** | Neutral, happy, angry, sad, surprised |
| **Primary Metrics** | MOS-N (naturalness), MOS-E (emotion expressiveness), emotion classification accuracy (%) |
| **Secondary Metrics** | F0 mean/range (Hz), energy RMS (dB), speaking rate (syl/sec), prosody direction accuracy (%), controllability std. dev. |
| **Current SOTA MOS-N** | ~4.2–4.3/5 (StyleTTS2-based systems, 2024) |
| **Current SOTA Emotion Acc.** | ~84–88% (4-class, LLM-conditioned systems, 2024) |
| **Notes** | Ground-truth recordings provide reference acoustic profiles per emotion, enabling direction-accuracy calculation |

### Secondary: Phonetic Ground Truth Reference

Emotion-to-acoustic-direction mappings sourced from the psychoacoustics / affective computing literature (Banse & Scherer 1996; Schröder 2001; Juslin & Laukka 2003):

| Emotion | F0 Mean | F0 Range | Energy | Speaking Rate |
|---|---|---|---|---|
| Angry | ↑ High | ↑ Wide | ↑ High | ↑ Fast |
| Happy | ↑ High | ↑ Wide | ↑ High | ↑ Fast |
| Sad | ↓ Low | ↓ Narrow | ↓ Low | ↓ Slow |
| Neutral | Baseline | Baseline | Baseline | Baseline |
| Surprised | ↑↑ Very high | ↑↑ Very wide | ↑ High | Variable |

Angry and happy are deliberately included as a *hard case*: they have similar F0/energy profiles but must be differentiated through spectral texture and rhythm — testing whether systems over-rely on gross acoustic features.

---

## Trend Validation

### Supporting Papers (2024–2026)

1. **PromptTTS 2** (Leng et al., Microsoft Research, ICLR 2024)
   — Demonstrates LLM-conditioned zero-shot style control; evaluates only on MOS and style-consistency, not acoustic feature analysis. Directly motivates the gap this work fills.

2. **CosyVoice** (Du et al., Alibaba DAMO, arXiv 2024)
   — State-of-the-art LLM + flow-matching TTS with instruction following; claims fine-grained prosody control; evaluated on MOS and speaker similarity only, no F0/energy audit. Confirms recency and applicability.

3. **StyleTTS 2** (Li et al., NeurIPS 2023 / follow-ups 2024)
   — Adversarial reference-encoder approach achieving near-human MOS; strongest implicit baseline. Widely used in 2024 comparisons, making it an ideal representative of the implicit paradigm.

---

## Generated

2026-03-18T00:00:00Z
