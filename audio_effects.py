import numpy as np
from fractions import Fraction
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, resample_poly
from pathlib import Path
import random
import logging

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 8000


def resample_audio(audio, original_sr, target_sr=TARGET_SAMPLE_RATE):
    """
    Ресемплирует аудио до целевой частоты дискретизации.
    Использует polyphase filtering для качества.
    """
    if original_sr == target_sr:
        return audio, target_sr
    
    logger.debug(f"Ресемплинг: {original_sr} Гц -> {target_sr} Гц")
    
    # Упрощаем отношение частот для resample_poly
    frac = Fraction(target_sr, original_sr).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    
    resampled = resample_poly(audio, up, down)
    return resampled, target_sr


def load_audio(file_path, target_sr=TARGET_SAMPLE_RATE):
    """
    Загружает WAV файл, приводит к моно, float32 и целевой частоте дискретизации.
    Возвращает (audio_data, actual_sample_rate).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    sr, data = wavfile.read(path)
    logger.debug(f"Загружен {path.name}: {sr} Гц, {data.dtype}, shape={data.shape}")
    
    # Конвертация в float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Конвертация в моно
    if data.ndim > 1:
        data = data.mean(axis=1)
        logger.debug("Конвертировано в моно")
    
    # Нормализация [-1, 1]
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data /= max_val
    
    # Ресемплинг при необходимости
    if sr != target_sr:
        data, sr = resample_audio(data, sr, target_sr)
    
    return data, sr


def save_audio(file_path, sample_rate, audio_data):
    """
    Сохраняет аудио в формате int16 WAV.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Клиппинг и конвертация в int16
    audio_clipped = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    
    wavfile.write(path, sample_rate, audio_int16)
    logger.debug(f"Сохранено: {path} ({sample_rate} Гц)")


def add_realistic_dropouts(audio, sample_rate, dropout_probability=0.1, segment_duration_ms=50):
    """
    Добавляет реалистичные прерывания (затухание -> тишина -> нарастание).
    """
    segment_samples = int(sample_rate * segment_duration_ms / 1000)
    if segment_samples < 4:
        segment_samples = 4

    audio_length = len(audio)
    output_audio = audio.copy()

    for start in range(0, audio_length, segment_samples):
        end = min(start + segment_samples, audio_length)
        current_length = end - start

        if current_length < 4:
            continue

        if random.random() < dropout_probability:
            fade_samples = min(max(1, int(current_length * 0.3)), current_length // 2)
            
            # Затухание
            fade_out_curve = np.linspace(1.0, 0.0, fade_samples)
            output_audio[start:start + fade_samples] *= fade_out_curve

            # Тишина
            mid_start = start + fade_samples
            mid_end = end - fade_samples
            if mid_end > mid_start:
                output_audio[mid_start:mid_end] = 0.0

            # Нарастание
            fade_in_curve = np.linspace(0.0, 1.0, fade_samples)
            output_audio[end - fade_samples:end] *= fade_in_curve

    logger.debug(f"Добавлены артефакты: prob={dropout_probability}, seg={segment_duration_ms}ms")
    return output_audio


def apply_radio_effect(audio, sample_rate, snr_db=12.0, lowcut=300, highcut=3000):
    """
    Имитация рации: полосовой фильтр + белый шум + легкий клиппинг.
    """
    # Полосовой фильтр
    nyq = 0.5 * sample_rate
    sos = butter(5, [lowcut/nyq, highcut/nyq], btype='band', output='sos')
    filtered = sosfilt(sos, audio)

    # Добавление шума по SNR
    noise = np.random.normal(0, 1, filtered.shape)
    sig_pow = np.mean(filtered ** 2)
    if sig_pow < 1e-12:
        sig_pow = 1e-12
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(noise_pow / (np.mean(noise ** 2) + 1e-12))
    noisy = filtered + noise_scaled

    # Легкий клиппинг
    clipped = np.clip(noisy, -0.95, 0.95)
    
    logger.debug(f"Применён радио-эффект: фильтр {lowcut}-{highcut} Гц, SNR={snr_db} dB")
    return clipped


def overlay_audio(main_sig, overlay_sig, attenuation_db=-20.0):
    """
    Накладывает overlay_sig на main_sig.
    Если длины разные: overlay обрезается или вставляется в случайное место.
    """
    L_main = len(main_sig)
    L_ovr = len(overlay_sig)
    
    # Ослабление громкости
    gain_factor = 10 ** (attenuation_db / 20.0)
    overlay_att = overlay_sig * gain_factor

    if L_ovr > L_main:
        overlay_used = overlay_att[:L_main]
        logger.debug(f"Overlay обрезан с {L_ovr} до {L_main} отсчётов")
    elif L_ovr < L_main:
        max_start = L_main - L_ovr
        pos = random.randint(0, max_start) if max_start > 0 else 0
        overlay_used = np.zeros(L_main, dtype=overlay_att.dtype)
        overlay_used[pos:pos + L_ovr] = overlay_att
        logger.debug(f"Overlay вставлен с позиции {pos}")
    else:
        overlay_used = overlay_att

    return main_sig + overlay_used


def process_audio_pipeline(input_path, output_path, 
                          apply_artifacts=False, artifact_prob=0.1, artifact_seg_ms=50,
                          apply_radio=False, radio_snr=12.0,
                          overlay_path=None, overlay_att_db=-20.0,
                          target_sample_rate=TARGET_SAMPLE_RATE):
    """
    Полный пайплайн обработки одного файла.
    Порядок: Загрузка/ресемплинг -> Артефакты -> Радио -> Наложение -> Сохранение.
    """
    logger.info(f"Обработка: {input_path}")
    
    # Загрузка с автоматическим ресемплингом
    audio, sr = load_audio(input_path, target_sr=target_sample_rate)
    
    # Артефакты
    if apply_artifacts:
        audio = add_realistic_dropouts(audio, sr, artifact_prob, artifact_seg_ms)
    
    # Радио-эффект
    if apply_radio:
        audio = apply_radio_effect(audio, sr, radio_snr)
    
    # Наложение второго файла (с ресемплингом если нужно)
    if overlay_path and Path(overlay_path).is_file():
        overlay_audio_data, overlay_sr = load_audio(overlay_path, target_sr=sr)
        audio = overlay_audio(audio, overlay_audio_data, overlay_att_db)
    
    # Финальная нормализация
    max_val = np.max(np.abs(audio))
    if max_val > 1e-12:
        audio /= max_val
    
    # Сохранение
    save_audio(output_path, sr, audio)
    logger.info(f"Завершено: {output_path}")
    return True