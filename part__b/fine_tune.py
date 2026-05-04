import os
import torch
import pandas as pd
import librosa
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# 1. Metrikaların Hazırlanması (WER/CER)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 2. Model və Processor (GPU-ya göndərilir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_id, language="azerbaijani", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
model.gradient_checkpointing_enable()

# 3. Data Hazırlığı (Kiçik həcm: 100 Train, 20 Val)
DATA_PATH = "../../cv-corpus-25.0-2026-03-09/az"
def load_custom_ds(split_name, size):
    df = pd.read_csv(os.path.join(DATA_PATH, f"{split_name}.tsv"), sep='\t').head(size)
    paths = [os.path.join(DATA_PATH, "clips", p) for p in df['path']]
    return Dataset.from_dict({"path": paths, "sentence": df['sentence'].tolist()})

train_ds = load_custom_ds("train", 100)
val_ds = load_custom_ds("dev", 20)

def prepare_ds(batch):
    audio, _ = librosa.load(batch["path"], sr=16000)
    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

train_ds = train_ds.map(prepare_ds, remove_columns=["path", "sentence"])
val_ds = val_ds.map(prepare_ds, remove_columns=["path", "sentence"])

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# 5. Training Arguments (Overfitting-in qarşısını almaq üçün)
training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-az-results",
    per_device_train_batch_size=1,       # Ən minimum batch size
    gradient_accumulation_steps=8,      # Effektiv batch size 8 olaraq qalır
    learning_rate=1e-5,
    max_steps=50,                       # Vaxta qənaət üçün 50 addım kifayətdir
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=5,
    save_steps=20,
    fp16=False,                         # CPU-da False olmalıdır
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    # YADDAŞ ÜÇÜN KRİTİK DƏYİŞİKLİKLƏR:
    optim="sgd",                        # Adam çox RAM yeyir, SGD daha yüngüldür
    gradient_checkpointing=True,        # Aktivdir
    report_to=["none"]
)
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    compute_metrics=compute_metrics,
)

trainer.train()