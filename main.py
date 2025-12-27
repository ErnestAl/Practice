import os
import glob
from pathlib import Path
from silero import silero_tts

# Настройки
input_dir = "texts"
output_dir = "WAVs"

# Загрузка модели Silero TTS
model, _ = silero_tts(language='ru', speaker='v5_ru')

# Список дикторов
speakers = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']

# Частота дискретизации
sample_rate = 8000

# Получаем все файлы вида text_XXXXX.txt
all_files = sorted(glob.glob(os.path.join(input_dir, "text_*.txt")))

if not all_files:
    print("Не найдено исходных файлов по шаблону text_*.txt")
    exit(1)

file_records = []

for f in all_files:
    basename = Path(f).name
    if basename.startswith("text_") and basename.endswith(".txt"):
        try:
            # Извлекаем XXXXX из text_XXXXX.txt
            num_part = basename[5:-4]  
            text_number = int(num_part)
            file_records.append((text_number, f))
        except ValueError:
            print(f"Пропускаем файл с некорректным именем: {basename}")

# Сортируем по номеру текста
file_records.sort(key=lambda x: x[0])

print(f"Найдено {len(file_records)} текстовых файлов.")

os.makedirs(output_dir, exist_ok=True)

total_generated_files = 0

for text_number, input_path in file_records:
    print(f"\nОбработка файла: {Path(input_path).name}")

    with open(input_path, 'r') as f:
        ssml_sample = f.read()

    # Генерация речи для каждого диктора
    for speaker_index, speaker_name in enumerate(speakers, start=1):
        output_filename = f"audio_{text_number:05d}_{speaker_index:05d}.wav"
        output_path = os.path.join(output_dir, output_filename)

        print(f"  Озвучка диктором '{speaker_name}' файла {output_filename}")

        try:
            model.save_wav(
                ssml_text=ssml_sample,
                speaker=speaker_name,
                sample_rate=sample_rate,
                audio_path=output_path
            )
            total_generated_files += 1
        except Exception as e:
            print(f"  Ошибка при генерации: {e}")

print(f"\nГотово!")
print(f"   Обработано текстов: {len(file_records)}")
print(f"   Создано аудиофайлов: {total_generated_files}")