import numpy as np
import mne
eegfile = "./chb01_01"
edf = mne.io.read_raw_edf(eegfile+".edf")
print(edf.ch_names)
header = ','.join(edf.ch_names)
np.savetxt(eegfile+'.csv', edf.get_data().T, delimiter=',', header=header)