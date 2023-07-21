import numpy as np



def add_noise_to_signal(signal, target_snr_db = 20):
    signal_watts = signal ** 2
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(signal_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    y_noise = np.random.normal(0, np.sqrt(noise_avg_watts), len(signal_watts))
    return signal + y_noise



def add_noise_to_data(data, target_snr_db = 20):
    for sample in range(data.shape[0]):
        for channel in range(data.shape[2]):
            data[sample, :, channel] = add_noise_to_signal(data[sample, :, channel], target_snr_db)
    return data