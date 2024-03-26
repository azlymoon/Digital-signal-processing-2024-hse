import numpy as np
from numpy.fft import ifft, fft
from scipy.signal import sawtooth
import matplotlib.pyplot as plt


def rectangle(duration, pulse_width, pulse_position, sample_rate=2000, amplitude=1):
    t = np.arange(0, duration, 1 / sample_rate)
    signal = np.zeros_like(t)
    for position in pulse_position:
        pulse_start = int(position * sample_rate)
        pulse_end = int(pulse_start + pulse_width * sample_rate)
        if pulse_end <= len(signal):
            signal[pulse_start:pulse_end] = amplitude

    return t, signal


def triangular(duration, pulse_count, sample_rate=2000, amplitude=1):
    t = np.arange(0, duration, 1 / sample_rate)
    signal = (sawtooth(pulse_count * 2 * np.pi * t, 0.5) + 1) * 0.5 * amplitude
    return t, signal


def exp_sig(duration=10, sample_rate=2000, decay_factor=0.5):
    t = np.arange(0, duration, 1 / sample_rate)
    signal = np.zeros_like(t)
    signal[3 * sample_rate:] = np.exp(-decay_factor * t[3 * sample_rate:])
    return t, signal


def show(signals, times):
    plt.figure(figsize=(10, 6), dpi=80)
    plt.subplots_adjust(hspace=0.5)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(times[i], signals[i])
    plt.show()


def linear_conv(signal1, signal2):
    conv = []
    N, M = len(signal1), len(signal2)
    signal1 = np.concatenate([np.zeros(M - 1, dtype=int), signal1])
    signal2 = np.concatenate([signal2[::-1], np.zeros(N - 1, dtype=int)])
    for i in range(N + M - 1):
        signal2_r = np.roll(signal2, i)
        sm = np.sum(signal1 * signal2_r)
        conv.append(sm)
    return conv


def circle_conv(signal1, signal2):
    return np.real(ifft(fft(signal1) * fft(signal2)))


def run():
    duration = 1
    sample_rate = 2000

    t_rec_1, signal_rec_1 = rectangle(duration, 0.2, [0.4])
    t_rec_2, signal_rec_2 = rectangle(duration, 0.2, [0.4])

    rec_conv = np.convolve(signal_rec_1, signal_rec_2, mode='full')
    t_rec_conv = np.arange(0, 2 * duration, 1 / sample_rate)[:-1]

    show(
        signals=[signal_rec_1, signal_rec_2, rec_conv],
        times=[t_rec_1, t_rec_2, t_rec_conv],
    )
    t_triangular, signal_triangular = triangular(duration, 1)

    triangular_conv = np.convolve(signal_triangular, signal_triangular, mode='full')
    t_triangular_conv = np.arange(0, 2 * duration, 1 / sample_rate)[:-1]

    show(
        signals=[signal_triangular, signal_triangular, triangular_conv],
        times=[t_triangular, t_triangular, t_triangular_conv],
    )
    triangular_rec_conv = np.convolve(signal_triangular, signal_rec_2, mode='full')
    t_triangular_rec_conv = np.arange(0, 2 * duration, 1 / sample_rate)[:-1]

    show(
        signals=[signal_triangular, signal_rec_2, triangular_rec_conv],
        times=[t_triangular, t_rec_2, t_triangular_rec_conv],
    )
    t_exp, signal_exp = exp_sig(duration=10)
    t_rec, signal_rec = rectangle(10, 2, [4])

    exp_rec_conv = np.convolve(signal_rec, signal_exp, mode='full')
    t_exp_rec_conv = np.arange(0, 20, 1 / sample_rate)[:-1]

    show(
        signals=[signal_rec, signal_exp, exp_rec_conv],
        times=[t_rec, t_exp, t_exp_rec_conv],
    )
    conv1 = linear_conv(signal_rec_1, signal_rec_2)
    conv2 = linear_conv(signal_triangular, signal_triangular)
    conv3 = linear_conv(signal_rec, signal_exp)

    show(
        signals=[conv1, conv2, conv3],
        times=[t_rec_conv, t_triangular_conv, t_exp_rec_conv],
    )
    conv1 = circle_conv(signal_rec_1, signal_rec_2)
    conv2 = circle_conv(signal_triangular, signal_triangular)
    conv3 = circle_conv(signal_rec, signal_exp)

    show(
        signals=[conv1, conv2, conv3],
        times=[
            np.arange(0, duration, 1 / sample_rate),
            np.arange(0, duration, 1 / sample_rate),
            np.arange(0, 10, 1 / sample_rate)
        ],
    )

    duration = 0.03
    sample_rate = 10000
    eps = 0.00001

    freq_nums = [
        (941.0, 1336.0),
        (697.0, 1209.0),
    ]
    t = np.arange(0, duration, 1 / sample_rate)
    signal1 = np.sin(2.0 * np.pi * freq_nums[0][0] * t) + np.sin(2.0 * np.pi * freq_nums[0][1] * t)
    signal2 = np.sin(2.0 * np.pi * freq_nums[1][0] * t) + np.sin(2.0 * np.pi * freq_nums[1][1] * t)

    circle_conv_two_signal_math = np.array(
        [np.sum(signal1 * np.roll(signal2[::-1], i + 1)) for i in range(int(duration * sample_rate))])
    circle_conv_two_signal_fft = circle_conv(signal1, signal2)

    plt.subplot(2, 1, 1)
    plt.plot(t, circle_conv_two_signal_math)
    plt.subplot(2, 1, 2)
    plt.plot(t, circle_conv_two_signal_fft)
    plt.show()

    print(
        f'руками: {len(circle_conv_two_signal_math)}',
        f'по теореме: {len(circle_conv_two_signal_fft)}',
        f'совпадения выражения: {np.count_nonzero(np.abs((fft(circle_conv_two_signal_math) - (fft(signal1) * fft(signal2)))) < eps)}', sep='\n'
    )


run()
