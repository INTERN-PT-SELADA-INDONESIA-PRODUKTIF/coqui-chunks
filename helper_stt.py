import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead", category=UserWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
device = torch.device('cpu')
torch.set_default_tensor_type(torch.FloatTensor)
import torchaudio
torchaudio.set_audio_backend('soundfile')
import wave
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"))
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio import Pipeline
import numpy as np
from pydub import AudioSegment
import json
from datetime import datetime as dt
from stt import Model
import shutil
from helper_db import *



def transcribe(AUDIO_PATH):
    model_path = "cakra-asr.model"
    scorer_path = "cakra-asr.scorer"
    # Muat model
    model = Model(model_path)
    model.enableExternalScorer(scorer_path)
    lm_alpha = 0.75
    lm_beta = 1.85
    model.setScorerAlphaBeta(lm_alpha, lm_beta)
    beam_width = 2500 #500
    model.setBeamWidth(beam_width)
    w = wave.open(AUDIO_PATH, 'r')
    frames = w.getnframes()
    buffer = w.readframes(frames)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    text = model.stt(data16)
    return text



def diarization_audio(PATH_AUDIO):
    offline_vad = Pipeline.from_pretrained("pyannote/voice-activity-detection/config.yaml")
    diarization = offline_vad(PATH_AUDIO)
    list_dict = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dict = {"start": turn.start, "stop":turn.end}
        list_dict.append(dict)
    
    return list_dict



# Function to extract the audio slice for each segment
def extract_audio_slice(segment,audio_file):
    start_time = int(segment['start'] * 1000)  # Convert to milliseconds
    stop_time = int(segment['stop'] * 1000)    # Convert to milliseconds
    return audio_file[start_time:stop_time]



def audio_chunk(PATH_AUDIO,chunk_audio_path, base_path):
    segments = diarization_audio(PATH_AUDIO)
    audio_file = AudioSegment.from_file(PATH_AUDIO)
    audio_slices = [extract_audio_slice(segment, audio_file) for segment in segments]
    path_list = []
    # Save each audio slice to separate files
    for i, audio_slice in enumerate(audio_slices):
        output_file = f"{chunk_audio_path}/{base_path}_audio_slice_{i}.wav"
        audio_slice.export(output_file, format="wav")
        path_list.append(output_file)

    return path_list



def get_wav_duration(file_path):
    with wave.open(file_path, "rb") as wav_file:
        # Mendapatkan jumlah frame audio dalam file WAV
        num_frames = wav_file.getnframes()

        # Mendapatkan sample rate (jumlah frame per detik)
        sample_rate = wav_file.getframerate()

        # Menghitung durasi dalam detik
        duration = num_frames / sample_rate

    return duration



def segment_embedding(audio, PATH):
  
  '''
  Processing audion into embeddings data
  :param segment : whisper segments result
  :param duration : frame audio/rate
  :param audio : audio
  :param PATH : audio directory
  :return : embedding
  '''
  start = 0
    # Whisper overshoots the end timestamp in the last segment
  end = get_wav_duration(PATH)
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(PATH, clip)
  
  return embedding_model(waveform[None])



def chunk_embedding(chunk_path):
    len_chunk = len(chunk_path)
    audio = Audio()
    embeddings = np.zeros(shape=(len_chunk, 192))

    for i in range(len_chunk):
        embeddings[i] = segment_embedding(audio, chunk_path[i])
    embeddings = np.nan_to_num(embeddings)
    return embeddings



def model_cluster_upload(embeddings):

    '''
    Create clustering audio embeddings data to define labels for each detected sound
    The clustering using KMeans clustering and the number of cluster be obtained by KElbowvisualizer result
    :param embeddings : audio embeddings result
    :return labels : labeling result
    '''

    model_Kmeans = KMeans(random_state=10)
    visualizer = KElbowVisualizer(model_Kmeans, k=(1,6), visualize=False)
    visualizer.fit(embeddings)
    plt.close()
    elbow_value = visualizer.elbow_value_
    model_Kmeans_update = KMeans(n_clusters = elbow_value, random_state=10)
    model_Kmeans_update.fit(embeddings)
    labels = model_Kmeans_update.labels_

    return labels



def concate_process(labels, path_list, base_path):
    result_dict = []
    current_label = None
    current_concate = []

    for label, path in zip(labels, path_list):
        if current_label is None:
            current_label = label
            current_concate.append(path)
        elif current_label == label:
            current_concate.append(path)
        else:
            result_dict.append({"labels": current_label, "concate": current_concate})
            current_label = label
            current_concate = [path]

    # Tambahkan dictionary terakhir
    if current_label is not None:
        result_dict.append({"labels": current_label, "concate": current_concate})

    concate_list = []
    labels_concate = []
    # Loop untuk menggabungkan audio
    for i, result in enumerate(result_dict):
        # Inisialisasi audio gabungan
        concatenated_audio = AudioSegment.empty()
        list_audio_paths = result_dict[i]["concate"]
        for path in list_audio_paths:
            audio = AudioSegment.from_file(path)
            # Jeda antara audio, dalam milidetik (1 detik = 1000 milidetik)
            silence = AudioSegment.silent(duration=1000)
            concatenated_audio += audio + silence
        labels_concate.append(result_dict[i]["labels"])
        # Simpan audio hasil gabungan
        CONCATE_PATH = "concatenated_audios"
        output_path = os.path.join(CONCATE_PATH, base_path)
        audio_concate_path = f"{output_path}/concate_{i}.wav"
        concatenated_audio.export(audio_concate_path, format="wav")
        concate_list.append(audio_concate_path)
    
    return concate_list, labels_concate


def map_values(value, mapping):
    '''
    vectorization
    '''
    
    return mapping[value]



# def segmentation_transcript(concate_list, labels_concate):
#     label_unique = []
#     for i in labels_concate:
#         if i not in label_unique:
#             label_unique.append(i)
#         else:
#             pass
#     name_speaker = [f'SPEAKER_{str(i + 1)}' for i in range(len(label_unique))]

#     dictionary = {}
#     for key, value in zip(label_unique, name_speaker):
#         dictionary[key] = value

#     mapped_label = np.vectorize(map_values)(labels_concate, dictionary)


#     text_list = []
#     for audio_slice in concate_list:
#         text = transcribe(audio_slice)+' '
#         text_list.append(text)
    
#     list_dict = [{"text": text, "path": concate, "speaker": label}for text, concate, label in zip(text_list, concate_list, mapped_label)]

#     return list_dict

def segmentation_transcript(concate_list, labels_concate):
    
    label_unique = []
    for i in labels_concate:
        if i not in label_unique:
            label_unique.append(i)
        else:
            pass
    name_speaker = [f'SPEAKER_{str(i + 1)}' for i in range(len(label_unique))]

    dictionary = {}
    for key, value in zip(label_unique, name_speaker):
        dictionary[key] = value

    mapped_label = np.vectorize(map_values)(labels_concate, dictionary)


    text_list = []
    for i in  range(len(concate_list)):
        path_audio = concate_list[i]
        transcript = transcribe(concate_list[i])+' '
        speaker = mapped_label[i]
        chunk_db(path_audio, transcript, speaker)
        text_list.append(transcript)

    list_dict = [{"text": transcript, "path": concate, "speaker": label}for transcript, concate, label in zip(text_list, concate_list, mapped_label)]

    return list_dict




def upload_file(PATH_AUDIO):
    CONCATE_TEXT_PATH = 'concatenated_text'
    CONCATE_AUDIO_PATH = "concatenated_audios"
    AUDIO_CHUNK_PATH = 'chunk_audio'

    base_path = dt.now().strftime('%Y%m%d%H%M%S')

    concate_text_path = os.path.join(CONCATE_TEXT_PATH, base_path)
    os.mkdir(concate_text_path)

    concate_audio_path = os.path.join(CONCATE_AUDIO_PATH, base_path)
    os.mkdir(concate_audio_path)

    if  os.path.exists(AUDIO_CHUNK_PATH):
        shutil.rmtree(AUDIO_CHUNK_PATH)
        os.mkdir(AUDIO_CHUNK_PATH)

        
    else:
        os.mkdir(AUDIO_CHUNK_PATH)

    # os.mkdir(AUDIO_CHUNK_PATH)
    chunk_audio_path = os.path.join(AUDIO_CHUNK_PATH, base_path)
    os.mkdir(chunk_audio_path)

    header_db(PATH_AUDIO)
    
    path_list = audio_chunk(PATH_AUDIO, chunk_audio_path, base_path)
    embeddings = chunk_embedding(path_list)
    labels = model_cluster_upload(embeddings)
    concate_list, labels_concate = concate_process(labels, path_list, base_path)
    list_dict = segmentation_transcript(concate_list, labels_concate)

    with open(f"{concate_text_path}/output_all.json", "w") as file:
        json.dump(list_dict, file)

    return print("Script berhasil diunggah")

