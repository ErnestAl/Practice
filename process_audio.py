"""
process_audio.py
CLI инструмент для пакетной обработки аудиофайлов.
Использует библиотеку audio_effects.
"""


import sys
import argparse
import logging
import random
from pathlib import Path
from fnmatch import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
import audio_effects as audio_effects


# Константы
DEFAULT_ARTIFACTS_PROBABILITY = 0.1
DEFAULT_ARTIFACTS_SEGMENT_DURATION = 50
DEFAULT_RADIO_SNR = 12
DEFAULT_OVERLAY_PERCENT = 100.0
DEFAULT_OVERLAY_ATTINUATION = -20.0
TARGET_SAMPLE_RATE = 8000
DEFAULT_MAX_JOBS = 4


# Настройка логирования
def setup_logging(verbose=False, log_file=None):
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8', mode='w'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )
    
    # Снижаем уровень для scipy/numpy, чтобы не засорять вывод
    logging.getLogger('scipy').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def matches_mask(filename, mask):
    """Проверяет, соответствует ли имя файла маске."""
    return fnmatch(filename, mask)


def process_single_file_task(input_file, output_file, args):
    """Задача для обработки одного файла (для ThreadPoolExecutor)."""
    try:
        # Определяем, какой файл накладывать
        overlay_to_use = None
        
        # Режим 1: один фиксированный файл + маска
        if args.overlay and matches_mask(input_file.name, args.overlay_mask):
            overlay_to_use = args.overlay
        
        # Режим 2: случайный файл из папки + маска
        elif args.overlay_dir and matches_mask(input_file.name, args.overlay_mask):
            overlay_dir = Path(args.overlay_dir)
            if overlay_dir.is_dir():
                # Получаем все WAV-файлы в папке
                overlay_candidates = list(overlay_dir.glob("*.wav"))
                if overlay_candidates:
                    # Фильтруем по проценту
                    sample_size = max(1, int(len(overlay_candidates) * args.overlay_percent / 100))
                    if args.overlay_seed is not None:
                        # Используем локальный random с seed для воспроизводимости
                        rng = random.Random(args.overlay_seed + hash(str(input_file)))
                    else:
                        rng = random
                    sampled = rng.sample(overlay_candidates, sample_size)
                    # Случайно выбираем один из отобранных
                    overlay_to_use = rng.choice(sampled)
                    logger.debug(f"Для {input_file.name} выбран overlay: {overlay_to_use.name}")

        audio_effects.process_audio_pipeline(
            input_path=input_file,
            output_path=output_file,
            apply_artifacts=args.artifacts,
            artifact_prob=args.artifact_prob,
            artifact_seg_ms=args.artifact_seg,
            apply_radio=args.radio,
            radio_snr=args.radio_snr,
            overlay_path=overlay_to_use,
            overlay_att_db=args.overlay_att,
            target_sample_rate=args.target_sr
        )
        return input_file, True, None
    except Exception as e:
        return input_file, False, str(e)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Обработка WAV файлов: артефакты, радио-эффекты, наложение шума."
    )
    
    # Основные пути
    parser.add_argument("input", help="Входной файл или директория с WAV файлами")
    parser.add_argument("output", help="Выходной файл или директория для результатов")
    
    # Эффекты
    parser.add_argument("--artifacts", action="store_true", help="Добавить прерывания сигнала")
    parser.add_argument("--artifact-prob", type=float, default=DEFAULT_ARTIFACTS_PROBABILITY, 
                        help=f"Вероятность артефакта (0.0-1.0, по умолчанию: {DEFAULT_ARTIFACTS_PROBABILITY})")
    parser.add_argument("--artifact-seg", type=float, default=DEFAULT_ARTIFACTS_SEGMENT_DURATION, 
                        help=f"Длительность сегмента в мс (по умолчанию: {DEFAULT_ARTIFACTS_SEGMENT_DURATION})")
    
    parser.add_argument("--radio", action="store_true", 
                        help="Применить эффект рации (фильтр + шум)")
    parser.add_argument("--radio-snr", type=float, default=DEFAULT_RADIO_SNR, 
                        help=f"SNR для радио-шума в дБ (по умолчанию: {DEFAULT_RADIO_SNR})")
    
    # Наложение (взаимоисключающая группа)
    overlay_group = parser.add_mutually_exclusive_group()
    overlay_group.add_argument("--overlay", type=str, default=None, 
                            help="Путь к одному файлу для наложения (шум/фон)")
    overlay_group.add_argument("--overlay-dir", type=str, default=None,
                            help="Директория с WAV-файлами для случайного выбора при наложении")
    parser.add_argument("--overlay-percent", type=float, default=DEFAULT_OVERLAY_PERCENT,
                        help=f"Процент файлов из --overlay-dir для формирования пула (1-100, по умолчанию: {DEFAULT_OVERLAY_PERCENT})")
    parser.add_argument("--overlay-seed", type=int, default=None,
                        help="Seed для случайного выбора (для воспроизводимости)")
    parser.add_argument("--overlay-att", type=float, default=DEFAULT_OVERLAY_ATTINUATION, 
                        help=f"Ослабление наложения в дБ (по умолчанию:{DEFAULT_OVERLAY_ATTINUATION})")
    parser.add_argument("--overlay-mask", type=str, default="*", 
                        help="Маска файлов для наложения (например, 'voice_*')")
    
    # Технические параметры
    parser.add_argument("--target-sr", type=int, default=TARGET_SAMPLE_RATE,
                        help=f"Целевая частота дискретизации в Гц (по умолчанию: {TARGET_SAMPLE_RATE})")
    parser.add_argument("--jobs", type=int, default=DEFAULT_MAX_JOBS,
                        help=f"Количество потоков для параллельной обработки (по умолчанию: {DEFAULT_MAX_JOBS})")
    parser.add_argument("--recursive", action="store_true", 
                        help="Рекурсивный поиск файлов в подпапках")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Подробный вывод (режим отладки)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Путь к файлу лога (опционально)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Проверка измений параметров без явного влючения флагов
    if args.artifact_prob != DEFAULT_ARTIFACTS_PROBABILITY or args.artifact_seg != DEFAULT_ARTIFACTS_SEGMENT_DURATION:
        if not args.artifacts:
            logger.info("Обнаружены настройки артефактов без флага --artifacts. Включаю автоматически.")
            args.artifacts = True
    
    if args.radio_snr != DEFAULT_RADIO_SNR:
        if not args.radio:
            logger.info("Обнаружены настройки радио-шума буз флага --radio. Включаю автоматически.")
            args.radio = True

    # Подготовка списка задач
    tasks = []
    
    if input_path.is_file():
        if input_path.suffix.lower() != '.wav':
            logger.error("Входной файл должен быть в формате WAV")
            sys.exit(1)
        tasks.append((input_path, output_path))
        logger.info(f"Запланирована обработка 1 файла")
        
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        pattern = "**/*.wav" if args.recursive else "*.wav"
        
        wav_files = list(input_path.glob(pattern))
        if not wav_files:
            logger.error(f"Файлы WAV не найдены в {input_path}")
            sys.exit(1)
            
        for src_file in wav_files:
            if args.recursive:
                rel_path = src_file.relative_to(input_path)
                dst_file = output_path / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst_file = output_path / src_file.name
            tasks.append((src_file, dst_file))
        
        logger.info(f"Найдено {len(tasks)} файлов для обработки")
    else:
        logger.error(f"Путь не найден: {input_path}")
        sys.exit(1)
    
    # Многопоточное выполнение
    logger.info(f"Запуск обработки в {args.jobs} потоков...")
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_single_file_task, src, dst, args): (src, dst)
            for src, dst in tasks
        }
        
        for future in as_completed(futures):
            src, dst = futures[future]
            try:
                input_file, success, error = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    logger.error(f"Ошибка {input_file.name}: {error}")
            except Exception as e:
                error_count += 1
                logger.error(f"Необработанная ошибка {src.name}: {e}")
    
    # Итоговый отчёт
    logger.info(f"Обработка завершена. Успешно: {success_count}, Ошибок: {error_count}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()