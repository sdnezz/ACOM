import numpy as np
import soundfile as sf
from scipy.fft import fft, ifft
from scipy.signal import get_window
import matplotlib.pyplot as plt


def spectral_noise_suppression(
    input_audio_path: str,
    output_audio_path: str,
    segment_length: int = 2048,
    overlap_ratio: float = 0.5,
    noise_begin_ms: int = 0,
    noise_end_ms: int = 3000,
    attenuation_coefficient: float = 1.0,
    floor_coefficient: float = 0.05,
    visualize_results: bool = True):
    # Загрузка и предобработка аудио
    audio_data, sample_rate = sf.read(input_audio_path)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)  # Конвертация в моно
    audio_data = audio_data.astype(np.float32)

    # Нормализация амплитуды
    peak_amplitude = np.max(np.abs(audio_data))
    if peak_amplitude > 0:
        audio_data /= peak_amplitude

    # Параметры оконной обработки
    hop_size = int(segment_length * (1 - overlap_ratio))
    analysis_window = get_window('hamming', segment_length)  # Окно Хэмминга вместо Ханна

    # Построение шумового профиля
    start_idx = int(noise_begin_ms * sample_rate / 1000)
    end_idx = min(int(noise_end_ms * sample_rate / 1000), len(audio_data) - segment_length)

    noise_spectra = [
        np.abs(fft(audio_data[pos:pos + segment_length] * analysis_window))
        for pos in range(start_idx, end_idx, hop_size)
    ]
    noise_template = np.mean(noise_spectra, axis=0) if noise_spectra else np.zeros(segment_length)

    # Обработка сигнала с перекрытием и суммированием
    processed_signal = np.zeros(len(audio_data))
    window_accumulator = np.zeros(len(audio_data))

    for start_pos in range(0, len(audio_data) - segment_length, hop_size):
        # Взятие сегмента и применение окна
        segment = audio_data[start_pos:start_pos + segment_length] * analysis_window
        
        # Прямое преобразование Фурье
        spectrum = fft(segment)
        
        # Разделение на амплитуду и фазу
        magnitude = np.abs(spectrum)
        phase_component = np.angle(spectrum)
        
        # Спектральное вычитание
        cleaned_magnitude = np.maximum(
            magnitude - attenuation_coefficient * noise_template,
            floor_coefficient * magnitude
        )
        
        # Восстановление комплексного спектра
        restored_spectrum = cleaned_magnitude * np.exp(1j * phase_component)
        
        # Обратное преобразование Фурье
        cleaned_segment = np.real(ifft(restored_spectrum))
        
        # Наложение с перекрытием
        processed_signal[start_pos:start_pos + segment_length] += cleaned_segment * analysis_window
        window_accumulator[start_pos:start_pos + segment_length] += analysis_window ** 2

    # Коррекция амплитуды
    processed_signal /= (window_accumulator + 1e-12)
    
    # Квантование для 16-битного формата
    processed_signal = np.clip(processed_signal * 32767, -32768, 32767).astype(np.int16)
    
    sf.write(output_audio_path, processed_signal, sample_rate)

    if visualize_results:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        axes[0].specgram(audio_data, NFFT=segment_length, Fs=sample_rate, 
                        noverlap=hop_size, cmap='viridis', vmin=-100, vmax=0)
        axes[0].set_title('Исходный аудиосигнал')
        axes[0].set_ylabel('Частота (Гц)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].specgram(processed_signal / 32767.0, NFFT=segment_length, 
                        Fs=sample_rate, noverlap=hop_size, cmap='viridis', 
                        vmin=-100, vmax=0)
        axes[1].set_title(f'Очищенный сигнал (α={attenuation_coefficient}, β={floor_coefficient})')
        axes[1].set_ylabel('Частота (Гц)')
        axes[1].grid(True, alpha=0.3)
        
        freq_bins = np.fft.fftfreq(segment_length, 1/sample_rate)[:segment_length//2]
        axes[2].plot(freq_bins, 20*np.log10(noise_template[:segment_length//2] + 1e-12))
        axes[2].set_title('Шумовой профиль (средний амплитудный спектр)')
        axes[2].set_xlabel('Частота (Гц)')
        axes[2].set_ylabel('Уровень (дБ)')
        axes[2].set_xlim(0, sample_rate/2)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    return processed_signal, sample_rate


if __name__ == '__main__':
    spectral_noise_suppression(
        input_audio_path='danik.mp3',
        output_audio_path='audio_output.mp3',
        attenuation_coefficient=1.0,  # Меньше значение = менее агрессивное подавление
        floor_coefficient=0.05,
        visualize_results=True
    )