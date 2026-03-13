# Open Source AI/ML Models for Healthcare

_Updated 13.03.2026 · added MedGemma 1.5 and MedASR_

---

<details>
<summary><strong>🧠 LLMs</strong></summary>

### General purpose
> Can be used for summarization, drafting patient letters, agentic/RAG systems. Some include VLM (visual language models).

| Model | Description |
|-------|-------------|
| [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B) | General usage model for summarization, drafting patient letters, retrieval-augmented QA over guidelines/policies |
| [Mistral](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) | Open-weight option for summarization + structured extraction |
| [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25) | Multilingual text + vision-language variants; useful for multilingual clinical operations and multimodal workflows |

### Clinical use
> Can be used for named entity recognition (NER), text classification and other tasks when medical-specific terms are highly involved.

| Model | Description |
|-------|-------------|
| [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it) | Multimodal LLM fine-tuned on instruction tasks in EHR analysis and medical imaging |
| [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) | Clinical-note representations for NER, phenotyping, cohort selection, readmission/risk models, etc. |
| [BioGPT](https://huggingface.co/docs/transformers/en/model_doc/biogpt) | Biomedical text generation/mining; useful for literature-focused tasks (entity linking, relation extraction prototypes, etc.) |
| [MedAlpaca](https://huggingface.co/medalpaca/medalpaca-7b) | LLaMA fine-tuned on medical texts (WikiDoc, ChatDoctor, etc.) |

</details>

---

<details>
<summary><strong>👁️ Vision</strong></summary>

### General purpose

| Model | Description |
|-------|-------------|
| [YOLO](https://www.ultralytics.com/yolo) | Real-time detection for patient-room activity, PPE detection, fall-risk monitoring prototypes ⚠️ requires licence for commercial use |
| [Florence-2](https://huggingface.co/microsoft/Florence-2-large) | Promptable vision-language model for captioning, detection/grounding-style tasks, image understanding; useful for multimodal triage/QA or dataset bootstrapping |

### Medical imaging

| Model | Description |
|-------|-------------|
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Open-source segmentation baseline; often a first-choice for CT/MRI segmentation projects |
| [MONAI](https://github.com/Project-MONAI/MONAI) | Healthcare imaging toolkit used to build/train/deploy many imaging models (segmentation, classification, self-supervised, etc.) |
| [MedSAM](https://huggingface.co/wanglab/medsam-vit-base) | "Segment Anything" adapted to medical images; great for semi-automatic annotation and generalized segmentation workflows |

</details>

---

<details>
<summary><strong>🎙️ Speech</strong></summary>

| Model | Description |
|-------|-------------|
| [MedASR](https://huggingface.co/google/medasr) | Speech recognition model with Conformer architecture, pre-trained for medical dictation |
| [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3) | Robust transcription for clinician dictation, visit audio summaries, patient interviews (consent + governance needed) |
| [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) | Open framework + pretrained ASR checkpoints (Conformer, etc.); powerful for streaming and on-device deployment |

</details>

---

<details>
<summary><strong>📊 Tabular data</strong></summary>

> Algorithm implementations; need to be trained on your data.

| Model | Description |
|-------|-------------|
| [XGBoost](https://github.com/dmlc/xgboost) | Gradient boosting |
| [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | Random Forest and its variations |
| [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) | LSTM architectures for sequential/temporal healthcare data |

</details>
