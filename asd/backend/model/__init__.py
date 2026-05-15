import os

import gdown

# ID папки с вашими моделями
folder_id = "1cMfG4KCt_ZJdpaHaaKCbAu6f0IYiG6EC?usp=sharing"  # Замените на ID вашей папки

# Ссылка на папку
folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

# Путь куда сохранять модели
output_dir = "checkpoint_515"

# Создаём папку, если её нет
os.makedirs(output_dir, exist_ok=True)

# Скачиваем всю папку
gdown.download_folder(url=folder_url, output=output_dir, quiet=False)
try:
    a = os.path.dirname(os.path.abspath(__file__))
    os.listdir(a + '/' + 'checkpoint_515')
except ModuleNotFoundError:
    raise FileNotFoundError("In directory not found package 'backend' or directory 'frontend'") from None
