"""
create_input_texts.py
Разбивает исходный текст на чанки фиксированной длины и сохраняет в отдельные файлы.
Автоматически определяет и конвертирует кодировку входного файла в UTF-8.
"""

import sys
import argparse
import logging
from pathlib import Path
import chardet
import nltk

# Константы
FILE_NUMBERING_FORMAT = "08d"
DEFAULT_CHARS_LIMIT = 300
DEFAULT_ENCODING = "utf-8"
DETECTION_CONFIDENCE_THRESHOLD = 0.7
LANGUAGE = "russian"

# Настройка логирования
def setup_logging(verbose=False, log_file=None):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8", mode="w"))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )

logger = logging.getLogger(__name__)


def detect_file_encoding(file_path, read_bytes=100000):
    """
    Определяет кодировку файла с помощью chardet.
    
    Параметры:
        file_path: путь к файлу
        read_bytes: сколько байт прочитать для анализа (по умолчанию: 100 КБ)
    
    Возвращает:
        (encoding, confidence) или (None, 0.0) при ошибке
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(read_bytes)
        result = chardet.detect(raw_data)
        return result.get("encoding"), result.get("confidence", 0.0)
    except Exception as e:
        logger.warning(f"Не удалось определить кодировку {file_path}: {e}")
        return None, 0.0


def read_text_with_encoding(file_path, target_encoding=DEFAULT_ENCODING):
    """
    Читает текстовый файл, автоматически определяя кодировку и конвертируя в целевую.
    
    Параметры:
        file_path: путь к файлу
        target_encoding: целевая кодировка для возврата (по умолчанию: 'utf-8')
    
    Возвращает:
        Список строк текста
    """
    detected_enc, confidence = detect_file_encoding(file_path)
    
    if detected_enc is None:
        logger.warning(f"Кодировка не определена, пробуем {target_encoding}")
        detected_enc = target_encoding
    
    logger.info(f"Определена кодировка: {detected_enc} (уверенность: {confidence:.0%})")
    
    # Если кодировка уже целевая и уверенность высокая — читаем напрямую
    if detected_enc.lower().replace("-", "_") == target_encoding.lower().replace("-", "_") and confidence >= DETECTION_CONFIDENCE_THRESHOLD:
        logger.debug("Файл уже в целевой кодировке")
        with open(file_path, "r", encoding=target_encoding, errors="replace") as f:
            return [line.rstrip("\n\r") for line in f if line.strip()]
    
    # Иначе читаем с детектированной кодировкой и возвращаем как unicode
    try:
        with open(file_path, "r", encoding=detected_enc, errors="replace") as f:
            content = f.read()
        # Возвращаем строки, уже в Unicode (внутреннее представление Python)
        return [line for line in content.splitlines() if line.strip()]
    except (UnicodeDecodeError, LookupError) as e:
        logger.warning(f"Ошибка чтения в кодировке {detected_enc}, пробуем {target_encoding}: {e}")
        with open(file_path, "r", encoding=target_encoding, errors="replace") as f:
            return [line.rstrip("\n\r") for line in f if line.strip()]


def split_text_into_chunks(text_lines, chars_limit, language=LANGUAGE):
    """
    Разбивает текст на чанки заданной длины по границам предложений.
    
    Возвращает:
        Генератор чанков (строк)
    """
    chunk = ""
    for line in text_lines:
        sentences = nltk.sent_tokenize(line, language=language)
        for sentence in sentences:
            if len(chunk) + len(sentence) + 1 <= chars_limit:
                chunk += sentence + " "
            else:
                if chunk.strip():
                    yield chunk.strip()
                chunk = sentence + " "
        if chunk.strip():
            yield chunk.strip()
            chunk = ""


def get_next_file_number(output_dir, prefix="text_", suffix=".txt"):
    """Определяет следующий доступный номер файла в директории."""
    existing = list(output_dir.glob(f"{prefix}*{suffix}"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num_part = f.name[len(prefix):-len(suffix)]
            numbers.append(int(num_part))
        except (ValueError, IndexError):
            continue
    return max(numbers) + 1 if numbers else 1


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Разбивает текстовый файл на чанки и сохраняет в отдельные файлы."
    )
    parser.add_argument("input_file", help="Путь к исходному текстовому файлу")
    parser.add_argument("output_dir", help="Директория для сохранения чанков")
    parser.add_argument("--chars-limit", type=int, default=DEFAULT_CHARS_LIMIT,
                        help=f"Максимальная длина чанка в символах (по умолчанию: {DEFAULT_CHARS_LIMIT})")
    parser.add_argument("--output-encoding", type=str, default=DEFAULT_ENCODING,
                        help=f"Кодировка выходных файлов (по умолчанию: {DEFAULT_ENCODING})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument("--log-file", type=str, default=None, help="Путь к файлу лога")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    if not input_path.is_file():
        logger.error(f"Файл не найден: {input_path}")
        sys.exit(1)
    
    # Загрузка nltk-ресурсов при необходимости
    try:
        nltk.data.find(f"tokenizers/punkt_tab/{LANGUAGE}")
    except LookupError:
        logger.info("Загрузка токенизатора nltk...")
        nltk.download("punkt_tab", quiet=True)
    
    # Чтение исходного файла с авто-детекцией кодировки
    try:
        lines = read_text_with_encoding(input_path, target_encoding=args.output_encoding)
    except Exception as e:
        logger.error(f"Ошибка чтения файла {input_path}: {e}")
        sys.exit(1)
    
    if not lines:
        logger.warning("Исходный файл пуст или не содержит строк")
        sys.exit(0)
    
    logger.info(f"Загружено {len(lines)} непустых строк")
    
    # Создание выходной директории
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Определение стартового номера
    start_number = get_next_file_number(output_dir)
    logger.info(f"Начальный номер файла: {start_number:{FILE_NUMBERING_FORMAT}}")
    
    # Генерация и сохранение чанков
    files_created = 0
    current_number = start_number
    
    for chunk in split_text_into_chunks(lines, args.chars_limit):
        output_path = output_dir / f"text_{current_number:{FILE_NUMBERING_FORMAT}}.txt"
        try:
            output_path.write_text(chunk, encoding=args.output_encoding)
            files_created += 1
            current_number += 1
            if files_created % 10 == 0:
                logger.debug(f"Создано {files_created} файлов...")
        except Exception as e:
            logger.error(f"Ошибка записи {output_path}: {e}")
    
    end_number = current_number - 1
    logger.info(f"Готово! Создано файлов: {files_created}")
    logger.info(f"Диапазон номеров: {start_number:{FILE_NUMBERING_FORMAT}} – {end_number:{FILE_NUMBERING_FORMAT}}")


if __name__ == "__main__":
    main()