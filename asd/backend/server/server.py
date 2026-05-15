# async_example.py
import os

import gdown
from fastapi import FastAPI
from pydantic import BaseModel

from .translator import translate

app = FastAPI()

@app.on_event("startup")
async def load_model():
    folder_id = "1cMfG4KCt_ZJdpaHaaKCbAu6f0IYiG6EC?usp=sharing"  # Замените на ID вашей папки

    # Ссылка на папку
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    # Путь куда сохранять модели
    output_dir = "checkpoint_515"

    # Создаём папку, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Скачиваем всю папку
    gdown.download_folder(url=folder_url, output=output_dir, quiet=False)
class Item(BaseModel):
    text_trans: str = ''
@app.post("/async")
async def async_endpoint(item: Item):
    text_for_translate = item.text_trans
    res = translate(text_for_translate)
    return {"trans": res}
