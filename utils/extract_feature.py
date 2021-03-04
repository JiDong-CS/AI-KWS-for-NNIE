import os
import time
import json
import numpy as np
from scipy.io import wavfile
from multiprocessing import Pool, Manager
import tensorflow as tf
import tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

os.environ['CUDA_VISIBLE_DEVICES'] = ""
data_dir = "/data/mobvoi_hotword_dataset/"
label_dir = "/data/mobvoi_hotword_dataset_resources/"
feature_setting = {"sample_rate": 16000, "num_channels": 40, "window_size": 30, "window_step": 20}


def extract_mfcc_feature(wav_filename):
    _, audio_data = wavfile.read(wav_filename)
    audio_tensor = tf.convert_to_tensor(audio_tensor)
    mfcc_feature = frontend_op.audio_microfrontend(
        audio=audio_tensor,
        sample_rate=feature_setting["sample_rate"],
        window_size=feature_setting["window_size"],
        window_step=feature_setting["window_step"],
        num_channel=feature_setting["num_channels"],
        out_scale=1,
        out_type=tf.float32)
    return mfcc_feature.to_numpy()
	

def iter_utterance(utt_ids):
    for index, utt_id in enumerate(utt_ids):
        yield index, utt_id


def handle_utterance(param):
    index, utt_id = param
    
    mfcc_feature = extract_mfcc_feature(os.path.join(data_dir, "%s.wav" % utt_id))
    length = mfcc_feature.shape[0]
    if length <= 100:
        mfcc_feature = np.pad(mfcc_feature, [[0, 100 - mfcc_feature.shape[0]], [0, 0]], "constant", constant_values=0)
    else:
        begin_index = np.random.randint(0, length - 100)
        mfcc_feature = mfcc_feature[begin_index:begin_index + 100, :]
    shared_arr[index] = mfcc_feature.reshape(-1)
    

def main():
    utt_ids = []
    
    for label_filename in ['p_train.json', 'n_train.json', 'p_dev.json', 'n_dev.json', 'p_test.json', 'n_test.json']:
        with open(os.path.join(label_dir, label_filename), 'r', encoding='utf-8') as file:
            for item in json.load(file):
                utt_ids.append(item['utt_id'])
    
    shared_arr = Manager().dict()
    pool = Pool(24)
    start_time = time.time()
    print("Number of utterance: %d" % len(utt_ids))
    print("Start to extract feature")
    pool.map(handle_utterance, iter_utterance(utt_ids))
    print("Extracting feature completely, which takes %f seconds" % (time.time() - start_time))
    
    # save utterance ids in utt_ids.txt at current work folder
    with open("utt_ids.txt", "w", encoding="utf-8") as file:
        for utt_id in utt_ids:
            file.write("%s\n" % utt_id)
    
    # save features as numpy array in features.npy at current work folder
    features = np.zeros([len(utt_ids), len(shared_arr[0])], dtype=np.float32)
    for i in range(len(utt_ids)):
        features[i] = shared_arr[i]
    np.save("features.npy", features)
    