"""
tts_pipeline.py
Единый скрипт для озвучки текстовых файлов через Silero или Piper TTS.
Генерирует WAV-файлы с частотой дискретизации 8000 Гц.
"""

import sys
import argparse
import logging
import wave
from pathlib import Path
import chardet
from concurrent.futures import ThreadPoolExecutor, as_completed

# Константы
FILE_NUMBERING_FORMAT = "08d"
DEFAULT_ENCODING = "utf-8"
DETECTION_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_MAX_JOBS = 4
TEMP_SUBDIR = ".temp_tts"

# Silero
SILERO_LANGUAGE = "ru"
SILERO_SPEAKER_VERSION = "v5_ru"
SILERO_SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]

# Piper
PIPER_DEFAULT_MODELS = [
    "piper_models/ru_RU-denis-medium.onnx",
    "piper_models/ru_RU-dmitri-medium.onnx",
    "piper_models/ru_RU-irina-medium.onnx",
    "piper_models/ru_RU-ruslan-medium.onnx"
]


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
    logging.getLogger("scipy").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def detect_file_encoding(file_path, read_bytes=100000):
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(read_bytes)
        result = chardet.detect(raw_data)
        return result.get("encoding"), result.get("confidence", 0.0)
    except Exception as e:
        logger.warning(f"Не удалось определить кодировку {file_path}: {e}")
        return None, 0.0


def read_text_with_encoding(file_path, target_encoding=DEFAULT_ENCODING):
    detected_enc, confidence = detect_file_encoding(file_path)
    if detected_enc is None:
        detected_enc = target_encoding
    
    logger.debug(f"Кодировка {file_path.name}: {detected_enc} ({confidence:.0%})")
    
    if detected_enc.lower().replace("-", "_") == target_encoding.lower().replace("-", "_") and confidence >= DETECTION_CONFIDENCE_THRESHOLD:
        with open(file_path, "r", encoding=target_encoding, errors="replace") as f:
            return f.read()
    try:
        with open(file_path, "r", encoding=detected_enc, errors="replace") as f:
            return f.read()
    except (UnicodeDecodeError, LookupError) as e:
        logger.warning(f"Ошибка чтения в {detected_enc}, пробуем {target_encoding}: {e}")
        with open(file_path, "r", encoding=target_encoding, errors="replace") as f:
            return f.read()


def load_text_files(input_dir, pattern="text_*.txt"):
    files = []
    for f in input_dir.glob(pattern):
        if not f.is_file():
            continue
        try:
            num = int(f.name[5:-4])
            files.append((num, f))
        except (ValueError, IndexError):
            logger.warning(f"Пропущен файл с некорректным именем: {f.name}")
    files.sort(key=lambda x: x[0])
    return files


def get_next_speaker_index_for_text(output_dir, text_number):
    """
    Находит следующий доступный номер диктора (speaker_index) 
    для заданного номера текста в выходной папке.
    """
    pattern = f"audio_{text_number:{FILE_NUMBERING_FORMAT}}_*.wav"
    existing = list(output_dir.glob(pattern))
    
    if not existing:
        return 1
    
    max_speaker_num = 0
    for f in existing:
        try:
            # Формат: audio_XXXXX_YYYYY.wav
            parts = f.name.split("_")
            if len(parts) >= 3:
                speaker_str = parts[2].replace(".wav", "")
                speaker_num = int(speaker_str)
                if speaker_num > max_speaker_num:
                    max_speaker_num = speaker_num
        except (ValueError, IndexError):
            continue
    
    return max_speaker_num + 1


# Silero TTS
def silero_text_to_ssml(text):
    import re
    def replace_punctuation(match):
        char = match.group(0)
        if char in ",;:-":
            return f'{char}<break strength="weak" />'
        elif char in ".!?":
            return f'{char}<break strength="strong" />'
        return char
    
    result = "<speak>\n<p>\n"
    processed = re.sub(r'[,;:\-.!?]', replace_punctuation, text)
    result += f"   {processed}\n</p>\n</speak>"
    return result


def silero_generate_task(args):
    text_content, output_path, speaker_name, sample_rate, model = args
    try:
        if output_path.exists():
            return output_path, True, "Уже существует"
            
        ssml_text = silero_text_to_ssml(text_content)
        model.save_wav(
            ssml_text=ssml_text,
            speaker=speaker_name,
            sample_rate=sample_rate,
            audio_path=str(output_path)
        )
        return output_path, True, None
    except Exception as e:
        return output_path, False, str(e)


def run_silero_pipeline(text_files, output_dir, sample_rate, input_encoding, jobs):
    from silero import silero_tts
    
    logger.info("Загрузка модели Silero TTS...")
    model, _ = silero_tts(language=SILERO_LANGUAGE, speaker=SILERO_SPEAKER_VERSION)
    logger.info(f"Дикторы: {', '.join(SILERO_SPEAKERS)}")
    
    tasks = []
    
    for text_number, input_path in text_files:
        try:
            text_content = read_text_with_encoding(input_path, target_encoding=input_encoding)
            if not text_content.strip():
                logger.warning(f"Файл пуст: {input_path.name}")
                continue
        except Exception as e:
            logger.error(f"Ошибка чтения {input_path}: {e}")
            continue
        
        # Определяем, с какого номера диктора начинать для этого текста
        start_speaker_idx = get_next_speaker_index_for_text(output_dir, text_number)
        current_speaker_idx = start_speaker_idx
        
        for speaker_name in SILERO_SPEAKERS:
            output_filename = f"audio_{text_number:{FILE_NUMBERING_FORMAT}}_{current_speaker_idx:{FILE_NUMBERING_FORMAT}}.wav"
            output_path = output_dir / output_filename
            
            if output_path.exists():
                logger.debug(f"Файл уже существует, пропускаем: {output_filename}")
                current_speaker_idx += 1
                continue
                
            tasks.append((text_content, output_path, speaker_name, sample_rate, model))
            current_speaker_idx += 1
    
    if not tasks:
        logger.warning("Нет задач для выполнения")
        return 0, 0
    
    logger.info(f"Запуск генерации {len(tasks)} файлов в {jobs} потоков...")
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(silero_generate_task, t): t for t in tasks}
        for future in as_completed(futures):
            output_path, success, error = future.result()
            if success:
                success_count += 1
                logger.debug(f"Создан: {output_path.name}")
            else:
                error_count += 1
                logger.error(f"Ошибка генерации {output_path.name}: {error}")
    
    return success_count, error_count


# Piper TTS 

def piper_synthesize_task(args):
    input_path, temp_path, output_path, voice, input_encoding = args
    try:
        if output_path.exists():
            return output_path, True, "Уже существует"
            
        text = read_text_with_encoding(input_path, target_encoding=input_encoding)
        if not text.strip():
            return output_path, False, "Пустой текст"
        
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(temp_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        
        return temp_path, True, None
    except Exception as e:
        return output_path, False, str(e)


def piper_resample_task(args):
    from audio_effects import load_audio, save_audio
    temp_path, output_path, target_sr = args
    try:
        audio, sr = load_audio(temp_path, target_sr=target_sr)
        save_audio(output_path, sr, audio)
        return output_path, True, None
    except Exception as e:
        return output_path, False, str(e)


def run_piper_pipeline(text_files, output_dir, sample_rate, input_encoding, jobs, models, keep_temp):
    from piper import PiperVoice
    
    # Загружаем все модели заранее
    loaded_voices = []
    for model_path in models:
        try:
            voice = PiperVoice.load(model_path)
            loaded_voices.append(voice)
        except Exception as e:
            logger.error(f"Ошибка загрузки {model_path}: {e}")
    
    if not loaded_voices:
        logger.error("Не удалось загрузить ни одной модели Piper")
        return 0, len(text_files) * len(models)
    
    all_temp_files = []
    tts_errors = 0
    temp_dir = output_dir / TEMP_SUBDIR
    temp_dir.mkdir(exist_ok=True)
    
    logger.info("=== Этап 1: Синтез (Piper TTS) ===")
    
    for txt_num, txt_path in text_files:
        try:
            text_content = read_text_with_encoding(txt_path, target_encoding=input_encoding)
            if not text_content.strip():
                logger.warning(f"Файл пуст: {txt_path.name}")
                continue
        except Exception as e:
            logger.error(f"Ошибка чтения {txt_path}: {e}")
            tts_errors += len(loaded_voices)
            continue
        
        # Определяем следующий номер диктора для этого текста
        start_speaker_idx = get_next_speaker_index_for_text(output_dir, txt_num)
        current_speaker_idx = start_speaker_idx
        
        tasks_for_text = []
        for voice in loaded_voices:
            output_filename = f"audio_{txt_num:{FILE_NUMBERING_FORMAT}}_{current_speaker_idx:{FILE_NUMBERING_FORMAT}}.wav"
            output_path = output_dir / output_filename
            
            if output_path.exists():
                logger.debug(f"Файл уже существует, пропускаем: {output_filename}")
                current_speaker_idx += 1
                continue
            
            temp_name = f"temp_{txt_num:{FILE_NUMBERING_FORMAT}}_{current_speaker_idx:{FILE_NUMBERING_FORMAT}}.wav"
            temp_path = temp_dir / temp_name
            
            tasks_for_text.append((txt_path, temp_path, output_path, voice, input_encoding))
            current_speaker_idx += 1
        
        if not tasks_for_text:
            continue
            
        logger.info(f"Синтез для текста {txt_num}: {len(tasks_for_text)} файлов в {jobs} потоков...")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(piper_synthesize_task, t): t for t in tasks_for_text}
            for future in as_completed(futures):
                temp_path, success, error = future.result()
                if success:
                    all_temp_files.append((temp_path, futures[future][2])) # (temp_path, output_path)
                else:
                    tts_errors += 1
                    logger.error(f"Синтез для {futures[future][2].name}: {error}")
    
    if not all_temp_files:
        logger.error("Не создано ни одного временного файла")
        return 0, tts_errors
    
    logger.info(f"Этап 1 завершён. Временных файлов: {len(all_temp_files)}, Ошибок: {tts_errors}")
    
    logger.info("=== Этап 2: Ресемплинг через audio_effects ===")
    resample_tasks = []
    for temp_path, output_path in all_temp_files:
        resample_tasks.append((temp_path, output_path, sample_rate))
    
    resample_success = 0
    resample_errors = 0
    logger.info(f"Ресемплинг {len(resample_tasks)} файлов в {jobs} потоков...")
    
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(piper_resample_task, t): t for t in resample_tasks}
        for future in as_completed(futures):
            out_path, success, error = future.result()
            if success:
                resample_success += 1
            else:
                resample_errors += 1
                logger.error(f"Ресемплинг {out_path.name}: {error}")
    
    logger.info(f"Этап 2 завершён. Успешно: {resample_success}, Ошибок: {resample_errors}")
    
    if not keep_temp:
        logger.info("Очистка временных файлов...")
        deleted = 0
        for temp_path, _ in all_temp_files:
            try:
                if temp_path.exists():
                    temp_path.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Не удалось удалить {temp_path.name}: {e}")
        try:
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception:
            pass
        logger.info(f"Удалено временных файлов: {deleted}")
    else:
        logger.info(f"Временные файлы сохранены в: {temp_dir}")
    
    return resample_success, tts_errors + resample_errors


# === Main ===

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Озвучка текстовых файлов через Silero или Piper TTS с частотой 8000 Гц."
    )
    parser.add_argument("input_dir", help="Директория с текстовыми файлами (text_*.txt)")
    parser.add_argument("output_dir", help="Директория для сохранения WAV-файлов")
    parser.add_argument("--engine", type=str, choices=["silero", "piper"], default="silero",
                        help="Движок TTS: silero или piper (по умолчанию: silero)")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Частота дискретизации (по умолчанию: {DEFAULT_SAMPLE_RATE})")
    parser.add_argument("--input-encoding", type=str, default=DEFAULT_ENCODING,
                        help=f"Кодировка входных файлов (по умолчанию: {DEFAULT_ENCODING})")
    parser.add_argument("--jobs", type=int, default=DEFAULT_MAX_JOBS,
                        help=f"Количество потоков (по умолчанию: {DEFAULT_MAX_JOBS})")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Пути к моделям Piper (только для --engine piper)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Сохранить временные файлы (только для --engine piper)")
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
    
    text_files = load_text_files(input_dir)
    if not text_files:
        logger.error(f"Не найдено text_*.txt в {input_dir}")
        sys.exit(1)
    
    logger.info(f"Текстов: {len(text_files)}, Движок: {args.engine}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.engine == "silero":
        success, errors = run_silero_pipeline(
            text_files, output_dir, args.sample_rate, args.input_encoding, args.jobs
        )
    else:  # piper
        models = args.models if args.models else PIPER_DEFAULT_MODELS
        if not models:
            logger.error("Список моделей не может быть пустым")
            sys.exit(1)
        
        for m in models:
            if not Path(m).is_file():
                logger.error(f"Модель не найдена: {m}")
                sys.exit(1)
        
        logger.info(f"Моделей: {len(models)}")
        success, errors = run_piper_pipeline(
            text_files, output_dir, args.sample_rate, args.input_encoding, args.jobs, models, args.keep_temp
        )
    
    logger.info(f"Готово! Создано файлов: {success}, Ошибок: {errors}")
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
