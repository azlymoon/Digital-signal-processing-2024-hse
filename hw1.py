import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

A = 1  # Амплитуда сигналов
fs = 10000  # Частота дискретизации
step = 1 / fs  # Шаг дискретизации
duration = 1  # Длительность звука

# Частоты двухканальных сигналов
freq_nums = [
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


def get_signal(t, freq):
    return A * np.sin(2.0 * np.pi * freq[0] * t) + A * np.sin(2.0 * np.pi * freq[1] * t)


def show_plot_signal(duration=0.005, freq=freq_nums[0]):
    t = np.arange(0, duration, step)
    signal = get_signal(t, freq)
    plt.plot(t, signal)
    plt.show()


def play_signal(signal):
    start_time = time.time()
    sd.play(signal, samplerate=fs)
    sd.wait()
    print("Played sound for {:.2f} seconds".format(time.time() - start_time))


def main():
    t = np.arange(0, duration, step)
    command = input()
    while command != 'exit':
        if '0' <= command <= '9':
            num = int(command)
            show_plot_signal(freq=freq_nums[num])
            play_signal(get_signal(t, freq=freq_nums[num]))
        elif command == 'array':
            nums = list(map(int, input().split()))
            for num in nums:
                play_signal(get_signal(t, freq=freq_nums[num]))
        else:
            print('Enter correct command')
        command = input()


if __name__ == '__main__':
    main()
