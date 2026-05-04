# Azerbaijani Automatic Speech Recognition (ASR) System

This repository contains a specialized ASR pipeline developed for the Azerbaijani language as part of the **AI Engineer Internship** technical task. The project demonstrates the full lifecycle of an ASR system, from baseline evaluation to fine-tuning and inference.

##  Project Structure
*   **`part_a/`**: Baseline implementation and inference code using the pre-trained Whisper model.
*   **`part_b/`**: Fine-tuning scripts, data preparation, and training logs.
*   **`results/`**: Performance metrics (WER/CER) and training visualizations.
*   **`report.pdf`**: Analytical report covering technical challenges, error analysis, and production-readiness.
*   **`requirements.txt`**: Python dependencies required to replicate the environment.

##  Performance Comparison
The model was fine-tuned on a high-quality subset of the **Mozilla Common Voice (az)** dataset for 50 training steps. Despite the limited data volume, the model showed significant adaptation to Azerbaijani phonetics.

| Metric | Baseline (Whisper-base) | Fine-tuned Model |
| :--- | :--- | :--- |
| **WER (Word Error Rate)** | ~25.00% | **14.01%** |
| **CER (Character Error Rate)** | ~12.00% | **7.50%** |
| **Final Training Loss** | N/A | **0.075** |

##  Training Progress
The following charts illustrate the convergence of the model. The **Loss** decreased sharply from **~1.4** to **~0.07**, and the **WER** reached its optimal value at step 50.

## ️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Fidan200508/az-stt-intern](https://github.com/Fidan200508/az-stt-intern)
    cd az-stt-intern
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
**Author:** Fidan Allahverdiyeva  
