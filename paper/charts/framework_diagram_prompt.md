# Framework Diagram Prompt

**Paper**: ProsAudit: Mechanistic Controllability Analysis of Emotional Speech Synthesis Reveals an Accuracy-Disentanglement Tradeoff

## Image Generation Prompt

Flat-design academic architecture diagram, white background (#F8F8F8), vector art style with subtle drop shadows, suitable for a top-tier ML conference paper. Left-to-right data flow layout, high information density, clean sans-serif labels.

Top row: a single input node labeled "Emotional Text + Emotion Label" in a rounded rectangle (#E8EEF4, dark border #4477AA). Three downward arrows branch from this node into three parallel vertical pipeline columns, each representing a control paradigm.

Left column (label: "Explicit Conditioning"): stack of three rounded boxes — "Text Encoder" (#4477AA, white text), "One-Hot Emotion Embed" (#4477AA), "FastSpeech2 Decoder" (#4477AA) — connected by downward arrows.

Center column (label: "Variational Disentanglement"): stack — "VAE Encoder" (#44AA99, white text), "β-Weighted KL Loss" (#44AA99, annotate with β=1…8), "Latent Decoder" (#44AA99) — connected by downward arrows. Small annotation bubble beside β box: "β↑ → fidelity↓".

Right column (label: "LM Conditioning"): stack — "BERT Encoder" (#AA3377, white text), "Cross-Attention Bridge" (#AA3377), "Neural TTS Decoder" (#AA3377) — connected by downward arrows.

All three columns converge downward into a wide shared evaluation block "ProsAudit Evaluation Framework" (#CCBB44, dark text), subdivided horizontally into three labeled sub-boxes: "Acoustic Fidelity (F0/Energy RMSE)", "Emotion Classification Accuracy", "Prosody Regression Task". Below this block, two output nodes connected by a bidirectional arrow labeled "Accuracy–Disentanglement Tradeoff": left node "#4477AA: Controllability" and right node "#AA3377: Semantic Accuracy". All arrows are dark grey (#333333), 2px weight, filled arrowheads. Minimal text, no decorative elements, no gradients.

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
