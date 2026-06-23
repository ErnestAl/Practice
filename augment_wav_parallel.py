"""
augment_wav_parallel.py
Параллельная аугментация WAV-файлов с помощью soundstretch (SoundTouch).
Поддерживает tempo, pitch, rate и оптимизацию для речи (--speech).
"""

import sys
import argparse
import logging
import subprocess
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Константы
FILE_NUMBERING_FORMAT = "08d"
DEFAULT_INPUT_PATTERN = "audio_*_*.wav"
DEFAULT_PATTERN_PREFIX = "audio_"
START_IF_EMPTY = 1


def setup_logging(verbose=False, log_file=None, log_mode="a"):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8", mode=log_mode))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )

logger = logging.getLogger(__name__)


def get_next_number_for_prefix(output_dir, prefix):
    """Определяет следующий доступный номер для файлов с заданным префиксом."""
    existing = list(output_dir.glob(f"{prefix}_*.wav"))
    max_num = 0
    for ef in existing:
        try:
            num_str = ef.name.rsplit("_", 1)[-1][:-4]
            num = int(num_str)
            if num > max_num:
                max_num = num
        except (ValueError, IndexError):
            continue
    return max_num + 1 if max_num > 0 else START_IF_EMPTY


def group_files_by_prefix(files, pattern_prefix):
    """Группирует файлы по префиксу (части имени до последнего подчёркивания)."""
    groups = defaultdict(list)
    for f in files:
        basename = Path(f).name
        if basename.startswith(pattern_prefix) and "_" in basename:
            prefix = basename.rsplit("_", 1)[0]
            groups[prefix].append(f)
    return dict(sorted(groups.items()))


def parse_range(range_str):
    """Парсит строку диапазона 'start:stop:step' в список чисел."""
    if range_str is None:
        return None
    parts = list(map(float, range_str.split(":")))
    if len(parts) != 3:
        raise ValueError("Формат: start:stop:step")
    start, stop, step = parts
    if step == 0:
        raise ValueError("Шаг не может быть нулевым")
    result = []
    current = start
    while current <= stop + 1e-9:
        result.append(current)
        current += step
    return result


def generate_tasks(groups, output_dir, tempo_values, pitch_values, rate_values, use_speech):
    """Генерирует список задач с детерминированной нумерацией."""
    tasks = []
    for prefix, file_list in groups.items():
        next_num = get_next_number_for_prefix(output_dir, prefix)
        
        for src_file in file_list:
            # Комбинации tempo и pitch
            if tempo_values is not None and pitch_values is not None:
                for tempo in tempo_values:
                    for pitch in pitch_values:
                        suffix = f"{next_num:{FILE_NUMBERING_FORMAT}}"
                        output_path = output_dir / f"{prefix}_{suffix}.wav"
                        tasks.append((src_file, output_path, tempo, pitch, None, use_speech))
                        next_num += 1
            
            # Комбинации rate (если указан)
            if rate_values is not None:
                for rate in rate_values:
                    suffix = f"{next_num:{FILE_NUMBERING_FORMAT}}"
                    output_path = output_dir / f"{prefix}_{suffix}.wav"
                    tasks.append((src_file, output_path, None, None, rate, use_speech))
                    next_num += 1
    
    return tasks


def build_soundstretch_command(src_path, out_path, tempo, pitch, rate, use_speech):
    """Строит команду для soundstretch с учётом всех параметров."""
    cmd = [
        "soundstretch_x64.exe",
        str(src_path),
        str(out_path),
    ]
    
    if tempo is not None:
        cmd.append(f"-tempo={tempo:g}")
    if pitch is not None:
        cmd.append(f"-pitch={pitch:g}")
    if rate is not None:
        cmd.append(f"-rate={rate:g}")
    
    # Флаг оптимизации для речи
    if use_speech:
        cmd.append("-speech")
    
    return cmd


def run_soundstretch_task(args):
    """Выполняет одну задачу аугментации через soundstretch."""
    src_path, out_path, tempo, pitch, rate, use_speech = args
    cmd = build_soundstretch_command(src_path, out_path, tempo, pitch, rate, use_speech)
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path, True, None
    except subprocess.CalledProcessError as e:
        return out_path, False, f"код возврата {e.returncode}"
    except FileNotFoundError:
        return out_path, False, "soundstretch_x64.exe не найден в PATH"
    except Exception as e:
        return out_path, False, str(e)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Параллельная аугментация WAV-файлов через soundstretch (SoundTouch)."
    )
    parser.add_argument("input_dir", help="Директория с исходными WAV-файлами")
    parser.add_argument("output_dir", help="Директория для сохранения результатов")
    
    # Параметры аугментации
    parser.add_argument("--tempo-range", type=str, default="-20:200:20",
                        help="Диапазон изменения темпа в %%: start:stop:step (по умолчанию: -20:200:20)")
    parser.add_argument("--pitch-range", type=str, default="-5:10:3",
                        help="Диапазон изменения тона в полутонах: start:stop:step (по умолчанию: -5:10:3)")
    parser.add_argument("--rate-range", type=str, default=None,
                        help="Диапазон изменения rate (меняет И темп, И высоту): start:stop:step")
    
    # Флаг для речи
    parser.add_argument("--speech", action="store_true",
                        help="Использовать алгоритм, оптимизированный для обработки речевого сигнала")
    
    # Прочее
    parser.add_argument("--pattern", type=str, default=DEFAULT_INPUT_PATTERN,
                        help=f"Шаблон имён файлов (по умолчанию: {DEFAULT_INPUT_PATTERN})")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Количество потоков (по умолчанию: auto)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument("--log-file", type=str, default=None, help="Путь к файлу лога")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.is_dir():
        logger.error(f"Директория не найдена: {input_dir}")
        sys.exit(1)
    
    # Парсинг диапазонов
    try:
        tempo_values = parse_range(args.tempo_range)
        pitch_values = parse_range(args.pitch_range)
        rate_values = parse_range(args.rate_range)
    except ValueError as e:
        logger.error(f"Ошибка парсинга диапазона: {e}")
        sys.exit(1)
    
    if tempo_values is None and rate_values is None:
        logger.error("Должен быть указан хотя бы один из диапазонов: --tempo-range или --rate-range")
        sys.exit(1)
    
    # Поиск исходных файлов
    all_files = sorted(input_dir.glob(args.pattern))
    if not all_files:
        logger.error(f"Не найдено файлов по шаблону {args.pattern}")
        sys.exit(1)
    
    # Группировка и генерация задач
    groups = group_files_by_prefix(all_files, DEFAULT_PATTERN_PREFIX)
    logger.info(f"Найдено {len(groups)} групп файлов")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(groups, output_dir, tempo_values, pitch_values, rate_values, args.speech)
    
    if not tasks:
        logger.warning("Задачи не сгенерированы")
        sys.exit(0)
    
    # Настройка параллелизма
    max_workers = args.jobs or min(6, os.cpu_count() or 4)
    speech_mode = "ВКЛЮЧЁН" if args.speech else "ВЫКЛЮЧЕН"
    logger.info(f"Сформировано {len(tasks)} задач. Режим --speech: {speech_mode}. Запуск в {max_workers} потоков...")
    
    # Параллельное выполнение
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_soundstretch_task, t): t for t in tasks}
        
        for future in as_completed(futures):
            out_path, success, error = future.result()
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.error(f"Ошибка {out_path.name}: {error}")
    
    logger.info(f"Готово! Успешно: {success_count}, Ошибок: {error_count}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
