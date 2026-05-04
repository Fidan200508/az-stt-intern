import os
import pandas as pd
import librosa
import torch
import numpy as np
from datasets import Dataset
from transformers import pipeline
from jiwer import wer, cer
from tqdm import tqdm


def prepare_dataset(base_path, limit=50):
    """
    Common Voice TSV faylını oxuyur və audio faylları massivə çevirir.
    """
    test_tsv = os.path.join(base_path, "test.tsv")
    clips_dir = os.path.join(base_path, "clips")

    if not os.path.exists(test_tsv):
        raise FileNotFoundError(f"TSV faylı tapılmadı: {test_tsv}")

    # Dataseti limitləyirik (Tapşırıq tələbi: 50 nümunə)
    df = pd.read_csv(test_tsv, sep='\t').head(limit)

    audio_paths = [os.path.join(clips_dir, p) for p in df['path']]
    ds = Dataset.from_dict({
        "audio_path": audio_paths,
        "sentence": df['sentence'].tolist()
    })

    def load_audio(batch):
        # Librosa vasitəsilə 16kHz-də yükləyirik (Whisper üçün standart)
        audio_array, _ = librosa.load(batch["audio_path"], sr=16000)
        batch["audio"] = {"array": audio_array, "sampling_rate": 16000}
        return batch

    print("Audio faylları massivə (array) çevrilir...")
    return ds.map(load_audio)


if __name__ == "__main__":
    # Layihə strukturuna görə dataset yolu (Hissə A skriptindən bir pillə yuxarı)
    DATA_PATH = "../../cv-corpus-25.0-2026-03-09/az"

    try:
        # 1. Datasetin hazırlanması
        ds = prepare_dataset(DATA_PATH)

        # 2. Cihazın təyini (GPU və ya CPU)
        device = 0 if torch.cuda.is_available() else -1
        print(f"Model yüklənir (Cihaz: {'GPU' if device == 0 else 'CPU'})...")

        # Whisper modelinin yüklənməsi
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=device
        )

        # 3. İnferens prosesi
        all_preds = []
        all_refs = ds["sentence"]

        print("Transkripsiya (ASR) başlayır... Bu proses CPU-da bir neçə dəqiqə çəkə bilər.")
        for i in tqdm(range(len(ds))):
            # Tip uyğunsuzluğunu həll etmək üçün NumPy-a çeviririk
            audio_array = np.array(ds[i]["audio"]["array"])

            audio_input = {
                "array": audio_array,
                "sampling_rate": ds[i]["audio"]["sampling_rate"]
            }

            # Azərbaycan dili üçün proqnozun alınması
            res = asr_pipe(audio_input, generate_kwargs={"language": "azerbaijani"})
            all_preds.append(res["text"])

        # 4. Metrikaların hesablanması (Jiwer üçün list formatına çevrilmə)
        all_refs_list = list(all_refs)

        total_wer = wer(all_refs_list, all_preds) * 100
        total_cer = cer(all_refs_list, all_preds) * 100

        print(f"\n" + "=" * 40)
        print(f"--- HİSSƏ A: BAZA MODEL NƏTİCƏLƏRİ ---")
        print(f"WER (Word Error Rate): {total_wer:.2f}%")
        print(f"CER (Character Error Rate): {total_cer:.2f}%")
        print("=" * 40)

        # 5. Analiz üçün nəticələrin yadda saxlanılması
        os.makedirs("../results", exist_ok=True)
        results_df = pd.DataFrame({
            "Reference": all_refs_list,
            "Prediction": all_preds
        })

        output_path = "../results/base_model_results.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nNəticələr '{output_path}' faylına uğurla yazıldı.")

    except Exception as e:
        print(f"\nXəta baş verdi: {e}")