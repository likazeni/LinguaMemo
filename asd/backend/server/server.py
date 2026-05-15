# async_example.py
from fastapi import FastAPI
import asyncio
from pydantic import BaseModel
from translator import translate
app = FastAPI()
class Item(BaseModel):
    text_trans: str = ''
@app.post("/async")
async def async_endpoint(item: Item):
    text_for_translate = item.text_trans
    res = translate(text_for_translate)
    return {"trans": res}
