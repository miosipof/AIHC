# Open Source AI/ML models for Healthcare
_______________________________________

Updated 06.02.2026

## LLMs

### General purpose

Can be used for summarization, drafting patient letters, agentic/RAG systems. Some include VLM (visual language models).

* [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B) General usage model for summarization, drafting patient letters, retrieval-augmented QA over guidelines/policies
* [Mistral](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) open-weight option for summarization + structured extraction
* [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25) multilingual text + vision-language variants; useful for multilingual clinical operations and multimodal workflows.


### Clinical use

Can be used for named entity recognition (NER), text classification and other tasks when medical-specific terms are highly involved.

* [ClinicalBERT](https://huggingface.co/medicalai/ClinicalBERT) clinical-note representations for NER, phenotyping, cohort selection, readmission/risk models, etc
* [BioGPT](https://huggingface.co/docs/transformers/en/model_doc/biogpt) biomedical text generation/mining; useful for literature-focused tasks (entity linking, relation extraction prototypes, etc.)



## Vision

### General purpose

* [YOLO](https://www.ultralytics.com/yolo) real-time detection for patient-room activity, Personal Protective Equipment (PPE) detection, fall-risk monitoring prototypes (requires licence for commercial use)
* [Florence-2](https://huggingface.co/microsoft/Florence-2-large) promptable vision-language model for captioning, detection/grounding-style tasks, image understanding; useful for multimodal triage/QA or dataset bootstrapping.

### Medical imaging

* [nnU-Ne](https://github.com/MIC-DKFZ/nnUNet) open-source segmentation baseline; often a first-choice for CT/MRI segmentation projects
* [MONAI](https://github.com/Project-MONAI/MONAI) healthcare imaging toolkit used to build/train/deploy many imaging models (segmentation, classification, self-supervised, etc.)
* [MedSAM](https://huggingface.co/wanglab/medsam-vit-base) “segment anything” adapted to medical images; great for semi-automatic annotation and generalized segmentation workflows.


## Speech

* [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3) robust transcription for clinician dictation, visit audio summaries, patient interviews (consent + governance needed).
* [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) (Conformer, etc) open framework + pretrained ASR checkpoints; powerful for streaming and on-device deployment.

## Tabular data

Algorithm implementations; need to be trained.

* [XGBoost](https://github.com/dmlc/xgboost) Gradient boosting
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and its variations
* [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) architectures
