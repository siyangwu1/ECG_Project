import numpy as np
import pywt
import wfdb
import matplotlib.pyplot as plt
import filter_set

record = wfdb.rdrecord('arrythmia_db/100', sampto=10000)
record = record.p_signal[:,0]

# Perform DWT
waveletname = 'db5'
coeffs = pywt.wavedec(record, waveletname)

# Plot original signal
plt.figure(figsize=(12, 9))
plt.subplot(2, 1, 1)
plt.plot(record, color="b", alpha=0.5)
plt.title("Original Signal")

# Plot DWT coefficients
plt.subplot(2, 1, 2)
plt.plot(coeffs[0], color="r", alpha=0.5)
plt.title("DWT coefficients")

plt.tight_layout()
plt.show()


'''
record = wfdb.rdrecord('arrythmia_db/100')
annotation = wfdb.rdann('arrythmia_db/100', 'atr', return_label_elements= 'symbol')
print(annotation.summarize_label())
'''



'''
record = wfdb.rdrecord('arrythmia_db/100', sampto=300)
OG_signal = record.p_signal[:, 0]
ecg_mean = np.mean(OG_signal)
filtered_signal = filter_set.butter_highpass(OG_signal, 0.5, 360 , order=2)
filtered_signal += ecg_mean
time = range(0, 300)
plt.figure(1)
plt.plot(time, OG_signal)
plt.title('OG signal')

plt.figure(2)
plt.plot(time, filtered_signal)
plt.title('high_pass_filtered')

plt.show()

'''
'''
# get 10s piece ECG data
record = wfdb.rdrecord('arrythmia_db/100', sampfrom = 0, sampto=3600)
OG_signal = record.p_signal[:, 0]

down_sampled_signal = filter_set.down_sampling(OG_signal, fs_orig= record.fs, fs_new=100)

down_sampled_signal



plt.figure(1)
plt.plot(range(0, 100), down_sampled_signal[:100])
plt.title('arrythmia down_sampled')

plt.figure(2)
plt.plot(range(0, 360), OG_signal[:360])
plt.title('arrythmia OG')

plt.show()

'''
'''
# get 10s piece ECG data
record = wfdb.rdrecord('atrial_fibrillation_db/04015', sampfrom = 20000, sampto=25000)
annotation = wfdb.rdann('atrial_fibrillation_db/04015', 'atr')
OG_signal = record.p_signal[:, 0]

down_sampled_signal = filter_set.down_sampling(OG_signal, fs_orig= record.fs, fs_new=100)

down_sampled_signal = filter_set.butter_highpass(down_sampled_signal, 0.5, 100 , order=2)

#final = pywt.dwt(down_sampled_signal[:1000], 'db1')
#final



plt.figure(1)
plt.plot(range(0, 100), down_sampled_signal[:100])
plt.title('AFIB down_sampled')

plt.figure(2)
plt.plot(range(0, 250), OG_signal[:250])
plt.title('AFIB OG')

plt.show()
'''