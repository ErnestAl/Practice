import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import matplotlib.pyplot as plt


def design_equalizer_filter(fft_size, frequency_bands, gain_factors, sample_rate):
    """
    Создаёт частотную маску для эквалайзера.
    
    Параметры:
        fft_size: размер окна FFT
        frequency_bands: список частотных диапазонов [(f_start, f_end), ...] в Гц
        gain_factors: список коэффициентов усиления (линейных, не в dB)
        sample_rate: частота дискретизации (в Гц)
    
    Возвращает:
        frequency_response_mask: Частотная маска длины fft_size // 2 + 1
    """
    positive_frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    frequency_response_mask = np.ones_like(positive_frequencies)

    for (band_start_hz, band_end_hz), gain in zip(frequency_bands, gain_factors):
        within_band = (positive_frequencies >= band_start_hz) & (positive_frequencies <= band_end_hz)
        frequency_response_mask[within_band] = gain

    return frequency_response_mask


def apply_overlap_add_filter(input_signal, frequency_response_mask, fft_size=2048, hop_length=50):
    """
    Применяет частотный фильтр с использованием Overlap-Add.
    
    Параметры:
        input_signal: входной аудиосигнал (1D массив)
        frequency_response_mask: маска АЧХ длины fft_size // 2 + 1
        fft_size: размер FFT-окна
        hop_length: шаг сдвига 
    
    Возвращает:
        output_signal: обработанный сигнал той же длины, что и вход
    """

    window_function = get_window('hann', fft_size)
    signal_length = len(input_signal)

    # Выходной буфер и буфер для суммы весов окон
    output_signal = np.zeros(signal_length + fft_size)
    window_sum = np.zeros(signal_length + fft_size)

    # Обработка по блокам
    for start_index in range(0, signal_length, hop_length):
        end_index = start_index + fft_size
        segment = input_signal[start_index:end_index]
        if len(segment) < fft_size:
            segment = np.pad(segment, (0, fft_size - len(segment)))

        windowed_segment = segment * window_function

        # FFT → фильтрация → обратный FFT
        segment_spectrum = np.fft.rfft(windowed_segment)
        filtered_spectrum = segment_spectrum * frequency_response_mask
        reconstructed_segment = np.fft.irfft(filtered_spectrum, n=fft_size)

        # Накопление сигнала и весов окна
        output_signal[start_index:end_index] += reconstructed_segment
        window_sum[start_index:end_index] += window_function

    # Обрезаем до исходной длины
    output_signal = output_signal[:signal_length]
    window_sum = window_sum[:signal_length]

    # Точная нормализация: делим на сумму оконных весов
    # Защищаемся от деления на ноль
    normalization_safe = np.where(window_sum > 1e-12, window_sum, 1.0)
    output_signal = output_signal / normalization_safe

    return output_signal


input_file_path = 'WAVs/audio_00001_00001.wav'
output_file_path = 'output_eq.wav'
overlap_percent = 75  # Процент перекрытия

# Загрузка аудио
sample_rate_hz, audio_data = wavfile.read(input_file_path)
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)  # моно

# Преобразуем в float32 и нормализуем по амплитуде
audio_data = audio_data.astype(np.float32)
audio_data = audio_data / np.max(np.abs(audio_data))


# Параметры обработки
fft_size = 2048
sample_rate_hz = float(sample_rate_hz)

# Вычисляем hop_length из процента перекрытия
overlap_ratio = overlap_percent / 100.0
hop_length = int(fft_size * (1.0 - overlap_ratio))
if hop_length < 1:
    hop_length = 1

# Параметры эквалайзера
frequency_bands_hz = [
    (20, 200),     # саббас
    (200, 800),    # низкие средние
    (800, 2000),   # средние
    (2000, 6000),  # высокие средние
    (6000, 20000)  # высокие частоты
]
gain_in_decibels = [-6, 3, 6, -6, 8]  # усиление в dB
gain_factors_linear = [10 ** (gain_db / 10.0) for gain_db in gain_in_decibels]

# Создаём частотную маску эквалайзера
frequency_response_mask = design_equalizer_filter(
    fft_size=fft_size,
    frequency_bands=frequency_bands_hz,
    gain_factors=gain_factors_linear,
    sample_rate=sample_rate_hz
)

# Применяем эквалайзер к сигналу
processed_audio = apply_overlap_add_filter(
    input_signal=audio_data,
    frequency_response_mask=frequency_response_mask,
    fft_size=fft_size,
    hop_length=hop_length
)

# Финальная нормализация и сохранение
processed_audio = np.clip(processed_audio, -1.0, 1.0)
processed_audio_int16 = (processed_audio * 32767).astype(np.int16)
wavfile.write('output_eq.wav', int(sample_rate_hz), processed_audio_int16)

print("Обработка завершена. Результат сохранён в 'output_eq.wav'")