import math
import numpy as np
from matplotlib import pyplot as plt

# Частоты двухканальных сигналов
freqs_dtmf = [
    (941.0, 1336.0),
    (697.0, 1209.0),
    (697.0, 1336.0),
    (697.0, 1477.0),
    (770.0, 1209.0),
    (770.0, 1336.0),
    (770.0, 1477.0),
    (852.0, 1209.0),
    (852.0, 1336.0),
    (852.0, 1477.0),
]


def signal_dtmf(freq, sample_rate=8000, duration=1, window_size=205, A=1):
    t = np.linspace(0, duration, sample_rate)[:window_size]
    return A * np.sin(2.0 * np.pi * freq[0] * t) + A * np.sin(2.0 * np.pi * freq[1] * t)


def goertzel(samples, sample_rate, target_freqs):
    window_size = len(samples)
    f_step_normalized = 1.0 / window_size

    n_range = range(0, window_size)
    freqs = []
    results = []

    bins = np.around(np.array(target_freqs) * (window_size / sample_rate))
    # bins = (18, 20, 22, 24, 31, 34, 38)
    for k in bins:

        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        d1, d2 = 0.0, 0.0
        for n in n_range:
            y = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2 ** 2 + d1 ** 2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results


def n_largest_indices_sorted(arr, n):
    indices = np.argpartition(arr, -n)[-n:]
    return indices[np.argsort(indices)]


def normal_noise_overlay(signal, t):
    noise_amplitude = np.max(signal) / 2
    noise = np.random.normal(0, noise_amplitude, len(t))
    return signal + noise


def show_signal_after_noise_fft(signal, signal_with_noise, t, harmonic_threshold=None):
    fft_result = np.fft.fft(signal_with_noise)
    if harmonic_threshold is not None:
        fft_result = np.where(2 * np.abs(fft_result) / len(signal) > harmonic_threshold, fft_result, 0)
    ifft_result = np.fft.ifft(fft_result)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Оригинальный сигнал')

    plt.subplot(2, 1, 2)
    plt.plot(t, ifft_result.real, label='Восстановленный сигнал')
    plt.title('Результат обратного FFT')
    plt.legend()

    plt.tight_layout()
    plt.show()


def decoder_all_dtmf():
    # Пример использования алгоритма Герцеля на декодирование сигнала DTMF
    freqs_list = []
    results_list = []
    symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    target_freqs = [697.0, 770.0, 852.0, 941.0, 1209.0, 1336.0, 1477.0]
    sample_rate = 8000

    plt.figure(figsize=(16, 12), dpi=70)

    for i in range(10):
        plt.subplot(5, 2, (i + 1))
        sine_wave = signal_dtmf(
            freq=freqs_dtmf[i],
            sample_rate=sample_rate,
        )
        freqs, results = goertzel(
            samples=sine_wave,
            sample_rate=sample_rate,
            target_freqs=target_freqs,
        )
        freqs_list.append(freqs)
        results_list.append(results)

        plt.plot(sine_wave, 'b-', linewidth=1.5)
        plt.grid()
        plt.ylabel('$s(n)$')
        plt.xlabel('Samples')
        plt.title(f'Символ: "{symbols[i]}", $f$={freqs_dtmf[i]} Гц')
    plt.tight_layout()

    plt.figure(figsize=(16, 12), dpi=70)
    for i in range(10):
        plt.subplot(5, 2, (i + 1))
        plt.stem(freqs_list[i], np.array(results_list[i])[:, 2], linefmt='b', markerfmt='bo')
        plt.ylabel('$|S(k)|$')
        plt.xlabel('$f(k)$ Гц')
        plt.title(f'Символ: "{symbols[i]}", $f$={freqs_dtmf[i]} Гц')
        plt.grid()
    plt.tight_layout()

    plt.show()


def decode_sequence_dtmf():
    # Декодирования последовательности DTMF
    nums = list(map(int, input("Введите последовательность чисел:\n").split()))
    target_freqs = [697.0, 770.0, 852.0, 941.0, 1209.0, 1336.0, 1477.0]
    sample_rate = 8000
    for num in nums:
        sine_wave = signal_dtmf(
            freq=freqs_dtmf[num],
            sample_rate=sample_rate,
        )
        freqs, results = goertzel(
            samples=sine_wave,
            sample_rate=sample_rate,
            target_freqs=target_freqs,
        )
        freq_ind = n_largest_indices_sorted(np.array(results)[:, 2], 2)
        for i, freq in enumerate(freqs_dtmf):
            if freq[0] == target_freqs[freq_ind[0]] and freq[1] == target_freqs[freq_ind[1]]:
                print(f"Ввели значение: {num}, декодировали значение: {i}")


def decode_am_signal():
    # Применение алгоритма Герцеля для АМ сигнала
    A = 15 * 10  # Амплитуда несущего сигнала
    f1 = 30000  # Частота несущего сигнала
    f2 = 1000  # Частота информационного сигнала
    sample_rate = 3e5  # Число временных отсчетов
    modulation_index = 0.5  # Коэффициент модуляции
    duration = 0.003
    t = np.arange(0, duration, 1 / sample_rate)
    am_signal = A * (1 + modulation_index * np.sin(2 * np.pi * f2 * t)) * np.sin(2 * np.pi * f1 * t)
    am_signal_noise = normal_noise_overlay(am_signal, t)
    show_signal_after_noise_fft(am_signal, am_signal_noise, t, 20)
    target_freqs = np.arange(f1 - f2 * 10, f1 + f2 * 10, f2 / 2)
    freqs, results = goertzel(
        samples=am_signal_noise,
        sample_rate=sample_rate,
        target_freqs=target_freqs,
    )
    plt.stem(freqs, np.array(results)[:, 2], linefmt='b', markerfmt='bo')
    plt.ylabel('$|S(k)|$')
    plt.xlabel('$f(k)$ Гц')
    plt.title(f'$f$ несущего сигнала: {f1}, $f$ информационного сигнала: {f2}')
    plt.grid()
    plt.show()


def main():
    decoder_all_dtmf()
    decode_sequence_dtmf()
    decode_am_signal()


if __name__ == '__main__':
    main()
