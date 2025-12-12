import numpy as np
import soundfile as sf
from scipy.fft import fft, ifft
from scipy.signal import get_window
import matplotlib.pyplot as plt


def noise_reduction(
    input_file: str,
    output_file: str,
    frame_size: int = 4096,
    overlap: float = 0.5,
    noise_start_ms: int = 0,
    noise_end_ms: int = 3000,
    suppression_factor: float = 1.2,
    protection_factor: float = 0.02,
    show_plots: bool = True,
):
    audio, sr = sf.read(input_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)      # моно
    audio = audio.astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak # нормализация по пиковой амплитуде

    hop = int(frame_size * (1 - overlap))   # шаг сдвига окна
    window = get_window('hann', frame_size) # окно Ханна

    # Оценка спектра шума
    start_sample = int(noise_start_ms * sr / 1000)
    end_sample = min(int(noise_end_ms * sr / 1000), len(audio) - frame_size)

    noise_frames = [
        np.abs(fft(audio[i:i + frame_size] * window))  # ДПФ → амплитудный спектр
        for i in range(start_sample, end_sample, hop)
    ]
    noise_profile = np.mean(noise_frames, axis=0) if noise_frames else np.zeros(frame_size)

    # Обработка по фреймам и сборка сигнала
    out_signal = np.zeros(len(audio))
    win_sum = np.zeros(len(audio))

    for i in range(0, len(audio) - frame_size, hop):

        frame = audio[i:i + frame_size] * window

        X = fft(frame)

        mag = np.abs(X)
        phase = np.angle(X)

        clean_mag = np.maximum(
            mag - suppression_factor * noise_profile,
            protection_factor * mag
        )

        # Восстановление спектра с исходной фазой
        clean_X = clean_mag * np.exp(1j * phase)

        # Обратное ДПФ → очищенный фрейм
        clean_frame = np.real(ifft(clean_X))

        # Наложение фреймов
        out_signal[i:i + frame_size] += clean_frame * window
        win_sum[i:i + frame_size] += window ** 2

    out_signal /= (win_sum + 1e-12)

    out_signal = np.clip(out_signal * 32767, -32768, 32767).astype(np.int16)
    sf.write(output_file, out_signal, sr)

    if show_plots:
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        plt.specgram(audio, NFFT=frame_size, Fs=sr, noverlap=hop,
                     cmap='plasma', vmin=-120, vmax=0)
        plt.title('Оригинальный сигнал')
        plt.ylabel('Частота, Гц')
        plt.colorbar(label='Уровень (дБ)')

        plt.subplot(2, 1, 2)
        plt.specgram(out_signal / 32767.0, NFFT=frame_size, Fs=sr, noverlap=hop,
                     cmap='plasma', vmin=-120, vmax=0)
        plt.title(f'После шумоподавления (α={suppression_factor}, β={protection_factor})')
        plt.xlabel('Время, с')
        plt.ylabel('Частота, Гц')
        plt.colorbar(label='Уровень (дБ)')

        plt.tight_layout()
        plt.show()

    return out_signal, sr


if __name__ == '__main__':

    noise_reduction(
        input_file='danik.mp3',
        output_file='output.mp3',
    )