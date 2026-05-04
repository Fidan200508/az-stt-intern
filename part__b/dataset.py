import os
import pandas as pd
import torch
from transformers import pipeline
from jiwer import wer, cer

# 1. Dataset Yolları
# Layihə strukturuna əsasən yolu tənzimləyirik
# Əgər dataset.py 'az-stt-intern' qovluğundadırsa, bir pillə yuxarı çıxırıq
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cv-corpus-25.0-2026-03-09", "az"))
clips_dir = os.path.join(base_path, "clips")
test_tsv = os.path.join(base_path, "test.tsv")


def load_local_cv(tsv_path):
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Fayl tapılmadı: {tsv_path}")

    df = pd.read_csv(tsv_path, sep='\t')
    # Lazımi sütunlar: 'path' (audio adı) və 'sentence' (orijinal mətn)
    df = df[['path', 'sentence']].dropna()
    # Audio fayllarının tam yolunu yaradırıq
    df['audio_path'] = df['path'].apply(lambda x: os.path.join(clips_dir, x))
    return df


try:
    # 2. Datanı Yükləyirik
    print(f"Dataset yüklənir: {test_tsv}")
    test_data = load_local_cv(test_tsv)
    # İlk 50 nümunə üzərində test edirik (Performans qiymətləndirməsi üçün)
    subset = test_data.head(50)
    print(f"Uğurla yükləndi! {len(subset)} nümunə test ediləcək.")

    # 3. Model Seçimi (Hissə A2)
    # Whisper-tiny həm sürətlidir, həm də Azərbaycan dilini baza səviyyəsində tanıyır
    device = 0 if torch.cuda.is_available() else -1
    print("Model yüklənir (OpenAI Whisper Tiny)...")
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)

    # 4. İnferens və Metrikaların Hesablanması (Hissə A3)
    results = []
    print("İnferens prosesi başladı...")

    for idx, row in subset.iterrows():
        try:
            # İnferens zamanı dili Azərbaycan dili olaraq təyin edirik
            prediction = asr_pipe(row['audio_path'], generate_kwargs={"language": "azerbaijani"})["text"]

            # WER və CER hesablanması
            w_err = wer(row['sentence'], prediction)
            c_err = cer(row['sentence'], prediction)

            results.append({
                "reference": row['sentence'],
                "prediction": prediction,
                "wer": w_err,
                "cer": c_err
            })
            if (idx + 1) % 10 == 0:
                print(f"Proqres: {idx + 1}/50")

        except Exception as e:
            print(f"Xəta (Fayl: {row['path']}): {e}")

    # Nəticələri analiz üçün DataFrame-ə çeviririk
    results_df = pd.DataFrame(results)

    # 5. Yekun Nəticələrin Çapı
    avg_wer = results_df['wer'].mean()
    avg_cer = results_df['cer'].mean()

    print("\n" + "=" * 40)
    print(f"ORTALAMA WER: {avg_wer:.2%}")
    print(f"ORTALAMA CER: {avg_cer:.2%}")
    print("=" * 40)

    # Ən yaxşı və ən pis 5 nümunə (WER-ə görə)
    print("\n--- ƏN YAXŞI 5 NÜMÜNƏ (Lower WER) ---")
    print(results_df.nsmallest(5, 'wer')[['reference', 'prediction', 'wer']])

    print("\n--- ƏN PİS 5 NÜMÜNƏ (Higher WER) ---")
    print(results_df.nlargest(5, 'wer')[['reference', 'prediction', 'wer']])

    # Nəticələri qeyd edək (Hesabatda istifadə üçün)
    os.makedirs("../results", exist_ok=True)
    results_df.to_csv("../results/baseline_results.csv", index=False)
    print("\nNəticələr 'results/baseline_results.csv' faylına qeyd edildi.")

except Exception as e:
    print(f"\nKritik Xəta: {e}")