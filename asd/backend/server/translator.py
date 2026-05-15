import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "checkpoint_515"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, max_memory={
    0: "400MB",   # GPU 0 limit
    "cpu": "400MB"  # CPU limit
}, device_map="cpu",
torch_dtype=torch.float16,     # половинная точность (экономия памяти)
low_cpu_mem_usage=True )
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, max_memory={
    0: "400MB",   # GPU 0 limit
    "cpu": "400MB"  # CPU limit
}, device_map="cpu",              # Только CPU
torch_dtype=torch.float16,     # половинная точность (экономия памяти)
low_cpu_mem_usage=True )

def translate(text):
    """Переводит текст с английского на немецкий"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test = translate("Hello, how are you? I`m your translator ;)")
print(test)