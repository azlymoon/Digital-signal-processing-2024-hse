import numpy as np
import matplotlib.pyplot as plt

N, k = 10, 1000
A = 15 * N  # Амплитуда несущего сигнала
B = N  # Амплитуда информационного сигнала
f1 = 3 * k * N  # Частота несущего сигнала
f2 = 100 * N  # Частота информационного сигнала
duration = 0.0025  # Продолжительность сигнала в секундах
sampling_rate = 1e6  # Число временных отсчетов
modulation_index = 1  # Коэффициент модуляции
frequency_deviation = sampling_rate / 500  # Частотная девиация
t = np.arange(0, duration, 1 / sampling_rate)


def fft_spectrum(signal, harmonic_threshold=None):
    # Применение БПФ
    fft_result = np.fft.fft(signal)

    # Получение массива частот
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)

    # Получение амплитуд спектра (усреднение магнитуд по комплексным числам)
    spectrum = np.abs(fft_result) / len(signal)

    # Фильтрация гармоник
    if harmonic_threshold is not None:
        spectrum = np.where(spectrum > harmonic_threshold, spectrum, 0)

    # Отсортировать массив частот для правильного отображения на графике
    idx = np.argsort(frequencies)
    frequencies = frequencies[idx]
    spectrum = spectrum[idx]

    return frequencies, spectrum


def show_spectrum_signal(signal, harmonic_threshold=None, mode=1, title='Спектр сигнала'):
    frequencies, spectrum = fft_spectrum(signal, harmonic_threshold)
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, spectrum)
    if mode == 1:
        plt.xlim(0, sampling_rate / 20)
    elif mode == 2:
        plt.xlim(-sampling_rate / 20, sampling_rate / 20)
    plt.title(title)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()


def normal_noise_overlay(signal):
    print(np.max(signal))
    noise_amplitude = np.max(signal) / 2
    noise = np.random.normal(0, noise_amplitude, len(t))
    return signal + noise


def compare_two_signals(signal1, signal2, title1='Сигнал 1', title2='Сигнал 2'):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal1)
    plt.title(title1)

    plt.subplot(2, 1, 2)
    plt.plot(t, signal2, color='orange')
    plt.title(title2)


def show_modulation(carrier_signal, message_signal, modulation_signal, title_modulation):
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, carrier_signal)
    plt.title('Несущий сигнал')

    plt.subplot(3, 1, 2)
    plt.plot(t, message_signal)
    plt.title('Информационный сигнал')

    plt.subplot(3, 1, 3)
    plt.plot(t, modulation_signal)
    plt.title(title_modulation)

    plt.tight_layout()
    plt.show()


def show_signal_after_noise_fft(signal):
    signal_with_noise = normal_noise_overlay(signal)
    fft_result = np.fft.fft(signal_with_noise)
    ifft_result = np.fft.ifft(fft_result)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Оригинальный сигнал')

    plt.subplot(2, 1, 2)
    plt.plot(t, ifft_result.real, label='Восстановленный сигнал (real)')
    plt.title('Результат обратного FFT')
    plt.legend()

    plt.tight_layout()
    plt.show()


def AM_signal_research(carrier_signal, message_signal):
    am_signal = A * (1 + modulation_index * np.sin(2 * np.pi * f2 * t)) * np.sin(2 * np.pi * f1 * t)
    show_modulation(carrier_signal, message_signal, am_signal, title_modulation='АМ сигнал')

    am_signal_with_noise = normal_noise_overlay(am_signal)

    compare_two_signals(
        signal1=am_signal,
        signal2=am_signal_with_noise,
        title1='AM сигнал без шума',
        title2='AM сигнал с добавление шума',
    )
    show_spectrum_signal(
        signal=am_signal,
        title='Спектр АМ сигнала без шума',
    )
    show_spectrum_signal(
        signal=am_signal_with_noise,
        title='Спектр АМ сигнала с добавлением шума',
        harmonic_threshold=10,
    )

    show_signal_after_noise_fft(am_signal)


def FM_signal_research(carrier_signal, message_signal):
    fm_signal = A * np.sin(2 * np.pi * f1 * t + (B * frequency_deviation) / f2 * np.sin(2 * np.pi * f2 * t))
    show_modulation(carrier_signal, message_signal, fm_signal, title_modulation='АМ сигнал')

    fm_signal_with_noise = normal_noise_overlay(fm_signal)

    compare_two_signals(
        signal1=fm_signal,
        signal2=fm_signal_with_noise,
        title1='ЧM сигнал без шума',
        title2='ЧM сигнал с добавление шума',
    )
    show_spectrum_signal(
        signal=fm_signal,
        title='Спектр ЧM сигнала без шума',
    )
    show_spectrum_signal(
        signal=fm_signal_with_noise,
        title='Спектр ЧM сигнала с добавлением шума',
        harmonic_threshold=5,
    )

    show_signal_after_noise_fft(fm_signal)


def main():
    carrier_signal = A * np.sin(2 * np.pi * f1 * t)
    message_signal = B * np.sin(2 * np.pi * f2 * t)

    AM_signal_research(carrier_signal, message_signal)
    FM_signal_research(carrier_signal, message_signal)


if __name__ == '__main__':
    main()
