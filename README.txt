Сторонние библиотеки:
    numpy scipy.io scipy.signal chardet ntlk silero piper-tts

------------------------------------------------
audio_effects.py - библиотека для обработки аудио файлов
augment_wav_parallel.py - аугментация аудио файлов с возможность параллелилизации
create_input_texts.py - создание отдельных текстовых файлов из исходного длинного текста
create_wav_from_text.py - озвучивание текстового файла
process_audio.py - обработка аудио файлов с использованием библиотеки audio_effects.py

------------------------------------------------
create_input_texts.py
Разбивает текстовый файл на чанки и сохраняет в отдельные файлы.
Параметры:
    input_file          Путь к исходному текстовому файлу
    output_dir          Директория для сохранения чанков
    --chars-limit       Максимальная длина чанка в символах (по умолчанию: 300)
    --output-encoding   Кодировка выходного файла (по умолчанию: "UTF-8")
    -v, --verbose       Подробный вывод логов
    --log-file          Путь к файлу для вывода логов (перезаписывается, если файл существует)

Примеры использования:
    1) Изменение длины чанка для выходного файла
    python.exe create_input_texts.py text.txt .\texts\ --chars-limit 200

    2) Подробный вывод логов
    python.exe create_input_texts.py text.txt .\texts\ --verbose

------------------------------------------------
create_wav_from_text.py
Генерация WAV-файлов из текстов.
Параметры:
    input_dir           Директория с текстовыми файлами
    output_dir          Директория для сохранения WAV-файлов
    --engine            Движок TTS: silero или piper (по умолчанию: silero)
    --sample-rate       Частота дискретизации выходного WAV-файла (по умолчанию: 8000)
    --input-encoding    Ожидаемая кодировка входных файлов (по умолчанию: "UTF-8")
    --jobs              Количество потоков для параллельной работы (по умолчанию: 4)
    --models            Пути к моделям Piper (только для --engine piper)
    --keep-temp         Сохранить временные файлы (только для --engine piper)
    -v, --verbose       Подробный вывод логов
    --log-file          Путь к файлу для вывода логов (перезаписывается, если файл существует)

Примеры использованием:
    1) Генерация WAV-файлов из текстовых файлов с колировкой Win-1251, использую 5 потоков
    python.exe create_wav_from_text.py .\texts\ .\wavs\ --input-encoding "cp1251" --jobs 5

------------------------------------------------
augment_wav_parallel.py
Параллельная аугментация WAV-файлов через soundstretch.
Параметры:
    input_dir           Директория с исходными WAV-файлами
    output_dir          Директория для сохранения новых WAV-файлов
    --tempo-range       Диапазон изменение темпа ("начало:конец:шаг") (по умолчанию: "-20:200:20")
    --pitch-range       Диапазон изменение тона ("начало:конец:шаг") (по умолчанию: "-5:10:3")
    --pattern           Шаблон имён входных файлов (по умолчанию: "silero_audio_*_*.wav")
    --jobs              Количество потоков для параллельной работы (по умолчанию: 4)
    -v, --verbose       Подробный вывод логов
    --log-file          Путь к файлу для вывода логов (перезаписывается, если файл существует)

Примеры использованием:
    1) Изменение диапазона изменение темпа
    python.exe augment_wav_parallel.py .\wavs\ .\augment_wavs\ --tempo-range "-40:200:10"

------------------------------------------------
process_audio.py
Обработка WAV файлов: артефакты, радио-эффекты, наложение шума.
Параметры:
    input               Входной файл или директоиия с WAV файлами
    output              Выходной файл или директоиия с новыми WAV файлами
    --artifacts         Добавление прерывания согнала (артефактов)
    --artifact-prob     Вероятность появления артефакта в секгмента (0.0-1.0, по умолчанию: 0.1)
    --artifact-seg      Длительность сегмента в мс (по умолчанию: 50)
    --radio             Применить эффект радио (фильтр + шум)
    --radio-snr         SNR для радио-шума в дБ (по умолчанию: 12)
    --overlay           Путь к одному файлу для наложения поверх другого
    --overlay-dir       Директория с WAV-файлами для случайного выбора при наложении
    --overlay-percent   Процент файлов из --overlay-dir для формирования пула (1-100, по умолчанию: 100)
    --overlay-seed      Seed для случайного выбора (по умолчанию: None)
    --overlay-att       Ослабление наложения в дБ (по умолчанию: -20.0)
    --overlay-mask      Маска файлов, на которые будет проихожить наложение
    --target-sr         Целевая частота дискретизации выходного файла (по умолчанию: 8000)
    --jobs              Количество потоков для параллельной работы (по умолчанию: 4)
    --recursive         Рекурсивный поиск файлов в подпапках
    -v, --verbose       Подробный вывод логов
    --log-file          Путь к файлу для вывода логов (перезаписывается, если файл существует)

Примеры использованием:
    1) Добавление аудио артефактов в файлы
    python.exe process_audio.py .\augment_wavs\ .\wavs_with_artifacts\ --artifacts
    
    2) Изменение длины сегмента с артефатом до 100 мс
    python.exe process_audio.py .\augment_wavs\ .\wavs_with_artifacts\ --artifacts --artifact-seg 100
    или
    python.exe process_audio.py .\augment_wavs\ .\wavs_with_artifacts\ --artifact-seg 100

    3) Наложение радио-шума
    python.exe process_audio.py .\augment_wavs\ .\wavs_with_radio\ --radio

    4) Наложение радио-шума и изменение SNR
    python.exe process_audio.py .\augment_wavs\ .\wavs_with_radio\ --radio --radio-snr 15
    или
    python.exe process_audio.py .\augment_wavs\ .\wavs_with_radio\ --radio-snr 15

    5) Наложение одного файла поверх другого
    python.exe process_audio.py .\main.wav .\new.wav --overlay .\noise.wav 

    6) Наложение файлов из одной директории поверх других файлов из директории 
    python.exe process_audio.py .\augment_wavs\ .\wavs_with_overlay\ --overlay-dir .\noises\

    7) Наложение аудио-артефактов, радио шума и второго аудио-файла, использую 8 процессов
    python.exe process_audio.py .\augment_wavs\ .\bad_augment_wavs\ --artifact --radio --overlay-dir .\overlay_wavs\ --jobs 8  


