
import re
import time

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

input_file = "metrics_and_datasets/cefr_leveled_texts.csv"
output_file = "metrics_and_datasets/temp_translations.csv"
df = pd.read_csv(input_file)
print(f"Загружено {len(df)} строк")
print(f"Колонки: {df.columns.tolist()}")
if 'text' not in df.columns:
    if 'content' in df.columns:
        df.rename(columns={'content': 'text'}, inplace=True)
    elif 'sentence' in df.columns:
        df.rename(columns={'sentence': 'text'}, inplace=True)
    else:
        print("Не найдена колонка с текстом! Доступные колонки:", df.columns.tolist())
        exit()
def clean_text(text):
    """Очистка текста от мусора"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+B2\s*$', '', text)
    text = re.sub(r'^B2\s+', '', text)
    text = text.strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)
df = df[df['text_clean'].str.len() > 10]
print(f"После очистки: {len(df)} строк")
gt = GoogleTranslator(source='en', target='de')

def translate_with_retry(text, max_retries=3):
    """Перевод с повторными попытками"""
    for attempt in range(max_retries):
        try:
            return gt.translate(text)
        except Exception as e:
            if "timed out" in str(e).lower():
                wait = (attempt + 1) * 3
                print(f"Таймаут, пауза {wait}с...")
                time.sleep(wait)
            else:
                time.sleep(1)
    return ""
print("\n" + "="*60)
print("Начинаю перевод...")
print("="*60 + "\n")

translations = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Перевод"):
    text = row['text_clean']

    if len(text) > 5000:
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        translated_chunks = []
        for chunk in chunks:
            trans = translate_with_retry(chunk)
            translated_chunks.append(trans)
            time.sleep(0.3)
        german = " ".join(translated_chunks)
    else:
        german = translate_with_retry(text)

    translations.append({
        'original_text': text,
        'german_translation': german
    })
    if (idx + 1) % 20 == 0:
        temp_df = pd.DataFrame(translations)
        temp_df.to_csv("temp_translations.csv", index=False)
        print(f"\n💾 Чекпоинт сохранен")

    time.sleep(0.5)

result_df = pd.DataFrame(translations)
result_df.to_csv(output_file, index=False, encoding='utf-8')

print("\n" + "="*60)
print(f"✅ ГОТОВО!")
print(f"   Файл сохранен: {output_file}")
print(f"   Переведено строк: {len(result_df)}")
print("="*60)
print("\n📝 Пример перевода:")
sample = result_df.iloc[0]
print(f"EN: {sample['original_text'][:150]}...")
print(f"DE: {sample['german_translation'][:150]}...")