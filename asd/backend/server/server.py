# async_example.py
import os
from contextlib import asynccontextmanager

import gdown
from fastapi import FastAPI
from pydantic import BaseModel

FOLDER_ID = "1cMfG4KCt_ZJdpaHaaKCbAu6f0IYiG6EC"
MODEL_DIR = "checkpoint_515"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Загружаем модель...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{FOLDER_ID}",
        output=MODEL_DIR,
        quiet=False,
        use_cookies=False
    )
    print("✅ Модель загружена!")
    yield
    print("👋 Сервер останавливается")

app = FastAPI(lifespan=lifespan)
from .translator import translate
class Item(BaseModel):
    text_trans: str = ''


@app.get("/")
async def root():
    return {"status": "ok", "message": "LinguaMemo API is running"}


@app.post("/async")
async def async_endpoint(item: Item):
    text_for_translate = item.text_trans
    res = translate(text_for_translate)
    return {"trans": res}
