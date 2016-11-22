import os
import numpy as np
import matplotlib.pyplot as plt

import dataset

path = "../data/"

print("PROCESSING DATA")
mood_dirs = ["sad", "happy", "anxious"]
midi_list = []
totalsum = np.zeros(88, dtype=np.float32)
for mood in mood_dirs:
    music_dir = os.listdir(path + mood + "/music")
    for music in music_dir:
        music_path = path + mood + "/music/" + music
        if os.path.isfile(music_path):
            midi = dataset.load_midi_data(music_path)
            totalsum = totalsum + np.sum(midi, axis=0)

# print totalsum
# totalsum.sort()
keep = [i for i,v in enumerate(totalsum) if v > 500]
print len(keep)
print keep
#x = np.arange(len(totalsum))
#plt.bar(x, totalsum, align='center', alpha=1)
#plt.xticks(x, totalsum)
#plt.ylabel('frequency')
#plt.title('note distribution')

#plt.show()