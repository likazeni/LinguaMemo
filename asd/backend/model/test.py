import time

import pandas as pd
from deep_translator import GoogleTranslator
from sacrebleu import sentence_bleu
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "checkpoint_515"  # Путь к вашей новой доученной модели
TEST_SIZE = 250
SOURCE_LANG = 'en'
TARGET_LANG = 'de'

print("="*60)
print("ЗАГРУЗКА МОДЕЛИ")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
print("✅ Модель загружена")

gt = GoogleTranslator(source=SOURCE_LANG, target=TARGET_LANG)

def translate_my_model(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_google(text):
    try:
        return gt.translate(text)
    except Exception as e:
        print(f"Google API ошибка: {e}")
        return ""

test_sentences = [
    "Hi! I've been meaning to write for ages and finally today I'm actually doing something about it.",
    "It was not so much how hard people found the challenge but how far they would go to avoid it.",
    "Keith recently came back from a trip to Chicago, Illinois.",
    "The Griffith Observatory is a planetarium, and an exhibit hall located in Los Angeles's Griffith Park.",
    "The weather is nice today.",
    "I need to finish this report by Friday.",
    "The significant weather forced the cancellation of 6 flights.",
    "According to the news report, the product launch will take place next October.",
    "The patient was diagnosed with a rare condition.",
    "Scientists have discovered a new species in the Amazon rainforest.",
]

while len(test_sentences) < TEST_SIZE:
    test_sentences.extend(test_sentences[:TEST_SIZE - len(test_sentences)])

test_sentences = test_sentences[:TEST_SIZE]

print(f"\n{'='*60}")
print(f"Запуск тестирования на {len(test_sentences)} предложениях...")
print(f"{'='*60}\n")

results = []

for i, src in enumerate(tqdm(test_sentences, desc="Перевод")):
    my_trans = translate_my_model(src)
    google_trans = translate_google(src)

    if my_trans and google_trans:
        try:
            bleu_score = sentence_bleu(my_trans, [google_trans]).score
        except:
            bleu_score = 0
    else:
        bleu_score = 0

    results.append({
        'id': i+1,
        'source': src,
        'my_model': my_trans,
        'google': google_trans,
        'bleu_vs_google': bleu_score
    })

    time.sleep(0.5)

df = pd.DataFrame(results)

print(f"\n{'='*60}")
print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
print(f"{'='*60}")
print(f"Всего предложений: {len(df)}")
print(f"Средний BLEU vs Google: {df['bleu_vs_google'].mean():.2f}")
print(f"Медианный BLEU: {df['bleu_vs_google'].median():.2f}")
print(f"Минимальный BLEU: {df['bleu_vs_google'].min():.2f}")
print(f"Максимальный BLEU: {df['bleu_vs_google'].max():.2f}")

print(f"\n--- Распределение качества ---")
bins = [0, 20, 40, 60, 80, 100]
labels = ['Ужасно (0-20)', 'Плохо (20-40)', 'Средне (40-60)', 'Хорошо (60-80)', 'Отлично (80-100)']
df['quality'] = pd.cut(df['bleu_vs_google'], bins=bins, labels=labels)
print(df['quality'].value_counts().sort_index())

print(f"\n--- ТОП-5 лучших совпадений с Google ---")
top5 = df.nlargest(5, 'bleu_vs_google')[['id', 'source', 'my_model', 'google', 'bleu_vs_google']]
for _, row in top5.iterrows():
    print(f"\n[{row['bleu_vs_google']:.1f} BLEU]")
    print(f"EN: {row['source'][:80]}")
    print(f"MY: {row['my_model']}")
    print(f"GT: {row['google']}")

print(f"\n--- ТОП-5 худших расхождений с Google ---")
bottom5 = df.nsmallest(5, 'bleu_vs_google')[['id', 'source', 'my_model', 'google', 'bleu_vs_google']]
for _, row in bottom5.iterrows():
    print(f"\n[{row['bleu_vs_google']:.1f} BLEU]")
    print(f"EN: {row['source'][:80]}")
    print(f"MY: {row['my_model']}")
    print(f"GT: {row['google']}")

df.to_csv('test_results_finetuned.csv', index=False)
print(f"\n✅ Полный отчет сохранен в 'test_results_finetuned.csv'")

avg_bleu = df['bleu_vs_google'].mean()
if avg_bleu > 70:
    print("\n🎉 Отлично! Ваша доученная модель почти не уступает Google Translate!")
elif avg_bleu > 55:
    print("\n👍 Хорошо! Дообучение дало результат!")
elif avg_bleu > 40:
    print("\n📈 Средний результат. Дообучение немного помогло.")
else:
    print("\n⚠️ Результат ниже ожидаемого. Возможно, нужно больше данных или эпох.")