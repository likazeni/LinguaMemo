
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

MODEL_PATH = "fine-tuned-marian/checkpoint-5626"

print("Загружаем модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
df = pd.read_csv('metrics_and_datasets/temp_translations.csv')
print(f"Загружено {len(df)} примеров")
df = df.rename(columns={
    'original_text': 'en',
    'german_translation': 'de'
})
df = df[(df['en'].str.len() > 20) & (df['en'].str.len() < 2000)]
print(f"После фильтрации: {len(df)} примеров")
dataset = Dataset.from_pandas(df[['en', 'de']])

def preprocess_function(examples):
    inputs = examples['en']
    targets = examples['de']

    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding='max_length'
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding='max_length'
        )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print("Токенизация...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']
training_args = TrainingArguments(
    output_dir="./marianmt_finetuned_1494",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\n" + "="*60)
print("🚀 НАЧИНАЕМ ДООБУЧЕНИЕ!")
print(f"   Train примеров: {len(train_dataset)}")
print(f"   Validation: {len(eval_dataset)}")
print("="*60 + "\n")

trainer.train()
model.save_pretrained("./marianmt_finetuned_1494")
tokenizer.save_pretrained("./marianmt_finetuned_1494")

print("\n✅ МОДЕЛЬ ГОТОВА! Путь: ./marianmt_finetuned_1494")