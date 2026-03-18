# machine-translation-minor-languages
This is for collaboration and sharing the code for a paper working on machine translation for minor languages

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-LoRA%20Model-blue)](https://huggingface.co/abhinandansamal/nllb-200-distilled-600M-LoRA-finetuned-odia-german-bidirectional)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-FFT%20Model-blue)](https://huggingface.co/abhinandansamal/nllb-200-distilled-600M-full-finetuned-odia-german-bidirectional)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c.svg)]()
[![Transformers](https://img.shields.io/badge/Transformers-4.57.3-blue.svg)]()
[![SentencePiece](https://img.shields.io/badge/SentencePiece-0.2.1-orange.svg)]()
[![BitsAndBytes](https://img.shields.io/badge/bitsandbytes-0.49.0-yellow.svg)]()
[![PEFT](https://img.shields.io/badge/PEFT-0.18.0-red.svg)]()
[![Gradio](https://img.shields.io/badge/Gradio-5.50.0-ff69b4.svg)]()
[![Huggingface Hub](https://img.shields.io/badge/Huggingface%20Hub-0.36.0-blueviolet.svg)]()
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Pro-8E75B2.svg)]()
[![OpenAI GPT](https://img.shields.io/badge/OpenAI-GPT--5-412991.svg)]()
[![Anthropic Claude](https://img.shields.io/badge/Anthropic-Claude%20Sonnet%204.5-D97757.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## 📖 Overview

This repository contains the code, data pipelines, and evaluation frameworks for a comprehensive research study on Bidirectional Neural Machine Translation (NMT) between **Odia (Oriya)**—a low-resource Indian language—and **German**. 

The study explores the fine-tuning of the **NLLB-200-distilled-600M** (No Language Left Behind) model, directly comparing **Full Fine-Tuning (FFT)** against Parameter-Efficient Fine-Tuning using **Low-Rank Adaptation (LoRA)**. 

### 🏆 Key Highlights
* **The "David vs. Goliath" Result:** The compact 0.6B parameter LoRA-optimized model outperformed massive proprietary LLMs (Gemini 2.5 Pro, Claude Sonnet 4.5, and GPT-5) in the German → Odia translation direction.
* **Storage Efficiency:** By mathematically merging the LoRA adapter weights back into the base model, we achieved an **87.43% reduction** in storage footprint (2.33 GB → 299 MB) with zero loss in inference quality.
* **Leakage-Free Corpus:** Built a highly curated parallel corpus from scratch using web-scraped data from prominent Odia newspapers, implementing strict pre-augmentation dataset splitting to completely eliminate data leakage.

---

## 💾 Dataset Creation & Curation
Because Odia-German is a rare language pair, a high-quality parallel corpus was built from the ground up:
1. **Scraping & Filtering:** Extracted 298 articles from leading Odia news platforms ("Sambad" and "Dharitri"). Implemented strict Unicode filtering (U+0B00 to U+0B7F) to isolate authentic Odia script.
2. **Translation & Validation:** Translated the cleaned corpus into German (2,000 lines via human translation/validation, and 1,676 lines via machine translation + strict human correction), resulting in **3,676 perfect 1:1 aligned sentence pairs**.
3. **Leakage-Free Splitting:** The unique sentence pairs were split (80% Train, 10% Val, 10% Test) *before* applying bidirectional task prefixes (`translate Odia to German: ` / `translate German to Odia: `). This strict order prevents the model from memorizing A → B in training and being tested on B → A.

---

## ⚙️ Methodology

### 1. Full Fine-Tuning (FFT)
* Trained all 600M parameters using 8-bit quantization for memory efficiency.
* **Optimizer:** Adafactor (Learning Rate: **2e-5**).
* Achieved statistically significant improvements over the NLLB baseline, verified via Paired Bootstrap Resampling (10,000 iterations).

### 2. Low-Rank Adaptation (LoRA)
* Targeted Attention (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and Feed-Forward (`fc1`, `fc2`) layers, training only **~10% of the model's parameters**.
* **Hyperparameter Optimization:** Utilized **Optuna** (Bayesian Optimization) to find the ideal configuration:
  * **Learning Rate:** **5e-4**
  * **Rank (r):** **128**
  * **Dropout:** **0.10**

### 3. Ablation Studies
Conducted isolated training loops to study the individual impacts of Learning Rate, Rank, and Dropout on model performance and training time stability. Visualized the trade-offs between computational cost and BLEU score efficiency.

---

## 📊 Benchmarking & Results

The fine-tuned models were rigorously benchmarked against the NLLB baseline and state-of-the-art LLMs. The LoRA-optimized model consistently demonstrated superior performance, particularly in the computationally difficult German → Odia direction.

### Final Performance Comparison (German → Odia)
| Model | Scale | BLEU | chrF++ | TER (↓) | COMET |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NLLB-200 (LoRA)** | **Compact (0.6B)** | **17.50** | **46.47** | **68.42** | 82.31 |
| Gemini 2.5 Pro | Massive LLM | 16.07 | 46.30 | 69.73 | **83.61** |
| Claude Sonnet 4.5 | Massive LLM | 14.87 | 44.83 | 73.45 | 82.61 |
| GPT-5 | Massive LLM | 8.07 | 37.73 | 80.10 | 80.27 |

*(Note: A comprehensive suite of visualizations, including Heatmaps, Radar Charts, and Probability Distribution graphs, can be found in Notebook 11 and Notebook 13).*

---

## 🚀 Quick Start: Using the Model

The best-performing, standalone merged LoRA model is hosted on the Hugging Face Hub and can be loaded seamlessly using the `transformers` pipeline.

### Installation
```bash
!pip install transformers==4.57.3 torch==2.9.0 sentencepiece==0.2.1 accelerate==1.12.0
```

```python
from transformers import pipeline
import torch

# Load the merged LoRA model
model_id = "abhinandansamal/nllb-200-distilled-600M-LoRA-finetuned-odia-german-bidirectional"

translator = pipeline(
    "translation",
    model=model_id,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 1. Translate Odia to German
odia_text = "ଆଜି ପାଗ ବହୁତ ଭଲ ଅଛି।"
prompt = "translate Odia to German: " + odia_text

german_out = translator(prompt, src_lang="ory_Orya", tgt_lang="deu_Latn", max_length=512)
print("German:", german_out[0]['translation_text'])

# 2. Translate German to Odia
german_text = "Die Feuerwehr musste zahlreiche Menschen mit Booten in Sicherheit bringen."
prompt = "translate German to Odia: " + german_text

odia_out = translator(prompt, src_lang="deu_Latn", tgt_lang="ory_Orya", max_length=512)
print("Odia:", odia_out[0]['translation_text'])
```

---

## 🖥️ Web Application (Gradio)

Interactive UI wrappers were built using **Gradio** to test and interact with both the Full Fine-Tuned (FFT) and LoRA-optimized models. The application features:
* **Automatic Language Detection**: Incorporates a custom Odia Unicode heuristic alongside `langdetect` to instantly route Odia or German inputs.
* **Dynamic Prompt Injection**: Automatically prepends the exact training prefixes (`translate Odia to German: ` / `translate German to Odia: `) required by the NLLB architecture to trigger the correct translation head.

**To run the Web Apps locally:**
Open the respective notebooks in your environment (or Google Colab) and run all cells. A public Gradio share link will be generated.
* **For the LoRA Model:** Run `notebooks/19_Adapter_based_fine_tuned_model_web_application.ipynb`
* **For the FFT Model:** Run `notebooks/18_fully_fine_tuned_model_web_application.ipynb`

---

## 📂 Repository Directory Structure

```text
├── data/
│   ├── raw/                           # Raw web-scraped Odia and translated German text files
│   └── transformed/                   # Processed, aligned, and split JSONL parallel corpora
├── eval/
│   ├── ablation_dropout.csv           # Raw results from dropout ablation
│   ├── ablation_lr.csv                # Raw results from learning rate ablation
│   ├── fft_sentence_scores_updated.pkl# Serialized sentence-level metrics for bootstrap testing
│   ├── final_sota_comparison_full_testset.csv # Benchmark results vs. SOTA LLMs
│   └── qualitative_examples_german_to_odia.csv # Sample translation outputs for error analysis
├── images/                            # Workflow architecture and next-token distribution charts
├── notebooks/                         
│   ├── 01_odia_news_article_web_scraping.ipynb
│   ├── 02_check_num_of_lines.ipynb
│   ├── 03_data_instance_creation.ipynb
│   ├── 04_bidirectional_corpus_create.ipynb
│   ├── 05_bidirectional_full_fine_tuning_evaluation_NLLB.ipynb
│   ├── 06_bidirectional_LoRA_fine_tuning_evaluation_NLLB.ipynb
│   ├── 07_ablation_study.ipynb
│   ├── 08_ablation_study_plot.ipynb
│   ├── 09_model_benchmarking.ipynb
│   ├── 10_NLLB_LoRA_Hyperparameter_Analysis.ipynb
│   ├── 11_Model_Performance_Visualization.ipynb
│   ├── 12_final_model_save_storage_footprint.ipynb
│   ├── 13_sampling_analysis_full_fine_tuning_LoRA_NLLB.ipynb
│   ├── 14_model_deployment.ipynb
│   ├── 15_fully_fine_tuned_nllb_model_load_test.ipynb
│   ├── 16_LoRA_fine_tuned_nllb_model_load_test.ipynb
│   ├── 17_data_upload_hf.ipynb
│   ├── 18_fully_fine_tuned_model_web_application.ipynb
│   └── 19_Adapter_based_fine_tuned_model_web_application.ipynb
├── plots/                             # Publication-ready visualizations (heatmaps, radar charts, etc.)
├── clean_notebooks.py                 # Utility script for cleaning notebook metadata/outputs
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 📓 Notebook Pipeline Walkthrough

If you wish to reproduce this research step-by-step, follow the notebooks in their numbered order:

* **Data Engineering (`01` - `04`):** Covers custom web scraping with Unicode filtering, strict 1:1 parallel corpus validation, JSONL formatting, and leakage-free bidirectional dataset splitting.
* **Model Training (`05` - `06`):** Contains the Full Fine-Tuning (FFT) pipeline and the Parameter-Efficient Fine-Tuning (LoRA) pipeline featuring Optuna hyperparameter optimization.
* **Ablation Studies (`07` - `08`):** Execution and visualization of hyperparameter ablation (Learning Rate, Rank, Dropout).
* **Evaluation & Benchmarking (`09` - `11`):** Automated metric generation (BLEU, chrF++, TER, COMET) against proprietary LLMs (Gemini, Claude, GPT) and performance plotting.
* **Model Export & Analysis (`12` - `14`):** Mathematical merging of LoRA weights, storage footprint analysis, next-token probability sampling, and Hugging Face Hub deployment.
* **Deployment & UI (`15` - `19`):** Local inference load testing (`15`, `16`), Hugging Face dataset uploads (`17`), and interactive Gradio web interface generation (`18`, `19`).

👉 `01_odia_news_article_web_scraping.ipynb`: Custom web scraping and Unicode filtering.

👉 `02_check_num_of_lines.ipynb`: Validation of 1:1 parallel corpus alignment.

👉 `03_data_instance_creation.ipynb` & `04_bidirectional_corpus_create.ipynb`: JSONL formatting and leakage-free bidirectional data splitting.

👉 `05_bidirectional_full_fine_tuning_evaluation_NLLB.ipynb`: FFT training pipeline and baseline evaluation.

👉 `06_bidirectional_LoRA_fine_tuning_evaluation_NLLB.ipynb`: PEFT LoRA training and Optuna hyperparameter tuning.

👉 `07_ablation_study.ipynb` & `08_ablation_study_plot.ipynb`: Hyperparameter ablation execution and visualization.

👉 `09_model_benchmarking.ipynb` to `11_Model_Performance_Visualization.ipynb`: SOTA LLM API inference and performance plotting.

👉 `12_final_model_save_storage_footprint.ipynb`: Mathematical merging of LoRA weights and storage footprint analysis.

👉 `13_sampling_analysis_full_fine_tuning_LoRA_NLLB.ipynb`: Next-token probability distribution analysis.

👉 `14_model_deployment.ipynb`: Hugging Face Hub deployment automation.

👉 `15_fully_fine_tuned_nllb_model_load_test.ipynb` & `16_LoRA_fine_tuned_nllb_model_load_test.ipynb`: Inference load testing.

👉 `18_fully_fine_tuned_model_web_application.ipynb` & `19_Adapter_based_fine_tuned_model_web_application.ipynb`: Gradio UI development.

#### Models deployed in HuggingFace: https://huggingface.co/abhinandansamal/models
