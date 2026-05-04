import pandas as pd
import os

# Əsas qovluq yolu
base_path = "cv-corpus-25.0-2026-03-09/az/"
clips_dir = os.path.join(base_path, "clips")


def load_local_cv(tsv_name):
    # TSV faylını oxuyuruq (tab ilə ayrılıb)
    df = pd.read_csv(os.path.join(base_path, tsv_name), sep='\t')

    # Səs fayllarının tam yolunu sütun olaraq əlavə edirik
    df['audio_path'] = df['path'].apply(lambda x: os.path.join(clips_dir, x))

    return df


# Nümunə üçün 'test.tsv' faylını yükləyək
test_data = load_local_cv("test.tsv")

print(f"Yüklənən sətir sayı: {len(test_data)}")
print("\nİlk 5 nümunə:")
print(test_data[['sentence', 'audio_path']].head())

# İlk səs faylının mövcudluğunu yoxlayaq
if os.path.exists(test_data['audio_path'].iloc[0]):
    print("\nSəs faylları uğurla tapıldı!")