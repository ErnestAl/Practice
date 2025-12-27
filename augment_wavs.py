import os
import subprocess
import glob
from pathlib import Path

# Настройки 
input_dir = "WAVs" 
output_dir = "WAVs_mod"  

# Диапазоны
tempo_values = list(range(-20, 201, 20))      
pitch_values = list(range(-5, 11, 3))       
combinations_per_file = len(tempo_values) * len(pitch_values)  

# Получаем все файлы вида audio_XXXXX_YYYYY.wav
all_files = sorted(glob.glob(os.path.join(input_dir, "audio_*_*.wav")))

if not all_files:
    print("Не найдено исходных файлов по шаблону audio_*_*.wav")
    exit(1)

# Группируем файлы по префиксу: audio_00001, audio_00002,
from collections import defaultdict
groups = defaultdict(list)

for f in all_files:
    basename = Path(f).name
    if basename.startswith("audio_") and "_" in basename:
        prefix = basename.rsplit("_", 1)[0]  # audio_00001
        groups[prefix].append(f)

# Сортируем группы по номеру
sorted_groups = sorted(groups.items(), key=lambda x: x[0])

print(f"Найдено {len(sorted_groups)} групп.")

os.makedirs(output_dir, exist_ok=True)

total_new_files = 0

for prefix, file_list in sorted_groups:
    print(f"\nОбработка группы: {prefix} (найдено файлов: {len(file_list)})")

    # Для этой группы начинаем нумерацию с 00006
    local_counter = 6  # будет превращаться в 00006, 00007

    # Обрабатываем каждый файл в группе (например, 00001..00005)
    for src_file in file_list:
        with open(src_file, "rb") as test:  # проверка, что файл не повреждён
            pass

        print(f"  → обработка файла: {os.path.basename(src_file)}")

        for tempo in tempo_values:
            for pitch in pitch_values:
                new_suffix = f"{local_counter:05d}"
                output_path = os.path.join(output_dir, f"{prefix}_{new_suffix}.wav")

                cmd = [
                    "soundstretch_x64.exe",
                    src_file,
                    output_path,
                    f"-tempo={tempo}",
                    f"-pitch={pitch}"
                ]

                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    local_counter += 1
                    total_new_files += 1
                except subprocess.CalledProcessError as e:
                    print(f"    Ошибка обработки: {e}")
                except FileNotFoundError:
                    print("Утилита 'soundstretch_x64.exe' не найдена.")
                    exit(1)

    print(f"  → создано файлов для {prefix}: {local_counter - 6}")

print(f"\nГотово!")
print(f"   Обработано групп: {len(sorted_groups)}")
print(f"   Создано новых файлов: {total_new_files}")
print(f"   Для каждой группы: суффиксы от 00006 до {local_counter - 1:05d}")