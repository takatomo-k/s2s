import shutil
import os
from tqdm import tqdm

import pdb;pdb.set_trace()
for p in tqdm(os.listdir("audio/mel/")):
    if "tts" in p:
        os.remove(os.path.join("audio/mel/", p))
    #shutil.move(os.path.join("../ted_speech/audio/mag/", p), "audio/mag/")
