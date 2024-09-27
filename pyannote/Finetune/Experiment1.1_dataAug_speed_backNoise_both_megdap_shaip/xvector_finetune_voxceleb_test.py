import os
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import io
from scipy.io import wavfile
import torch
import csv

# Load the fine-tuned model
model_paths = [
    "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/Models/Models_lr=0.0001_weight_decay=0_amsgrad=False/xvector_epoch5_EER0.000300.pt",
    "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/Models/Models_lr=0.001_weight_decay=0_amsgrad=True/xvector_epoch6_EER0.000250.pt",
    "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/Models/Models_lr=0.0001_weight_decay=0_amsgrad=True/xvector_epoch3_EER0.000234.pt",
    "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/Models/Models_lr=0.001_weight_decay=0.0005_amsgrad=False/xvector_epoch1_EER0.000332.pt",
    "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/Models/Models_lr=0.0001_weight_decay=0.0005_amsgrad=False/xvector_epoch2_EER0.000156.pt",
    "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/Models/Models_lr=0.001_weight_decay=0.0005_amsgrad=True/xvector_epoch1_EER0.000333.pt",
    "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/Models/Models_lr=0.0001_weight_decay=0.0005_amsgrad=True/xvector_epoch3_EER0.000226.pt",
]

# CSV file to store results
csv_file = "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1.1_dataAug_speed_backNoise_both_megdap_shaip/model_eer_results_voxceleb.csv"

# Load the dataset
dataset = load_dataset("Codec-SUPERB/Voxceleb1_test_original")
#print(dataset)
# Access the test split
test_dataset = dataset["test"]
#print(test_dataset)
test_dataset = test_dataset[:]

def extract_embedding(audio_path):
    # Extract the waveform and sample rate from audio_path
    waveform = audio_path['array']
    sample_rate = audio_path['sampling_rate']

    # Convert the waveform to the appropriate data type for writing the audio file
    # Make sure the values are in the range [-1, 1] for float32 data type
    waveform = waveform.astype(np.float32)

    # Create an in-memory file object to write the audio data
    audio_file = io.BytesIO()

    # Write the waveform to the in-memory file as a .wav file
    # Replace 'wavfile.write' with the appropriate function based on the library you use
    wavfile.write(audio_file, sample_rate, waveform)

    # Reset the file pointer to the beginning of the file
    audio_file.seek(0)

    # Now, pass the in-memory file object to the inference function
    # Replace 'inference' with the function you're using for inference
    embedding = inference(audio_file)

    return embedding.reshape(1, -1)


# Calculate distances for all pairs of embeddings
def calculate_distances(embeddings):
    # Concatenate embeddings along the first axis
    embeddings_concatenated = np.concatenate(embeddings, axis=0)
    distances = cdist(embeddings_concatenated, embeddings_concatenated, metric="cosine")
    return distances

def calculate_far_frr(distances, speaker_ids, threshold):
    genuine_scores = []
    impostor_scores = []
    for i in range(len(distances)):
        for j in range(len(distances)):
            #print("speakerids= ",speaker_ids[i], speaker_ids[j])
            if i != j:
                if speaker_ids[i] == speaker_ids[j]:
                    genuine_scores.append(distances[i][j])
                else:
                    impostor_scores.append(distances[i][j])

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    #print(len(genuine_scores), len(impostor_scores))
    far = np.sum(impostor_scores <= threshold) / len(impostor_scores)
    frr = np.sum(genuine_scores > threshold) / len(genuine_scores)
    
    return far, frr

# Calculate EER
def calculate_eer(distances, speaker_ids):
    thresholds_rounded = np.around(distances, decimals=1)  
    thresholds_unique = np.unique(thresholds_rounded)

    eer = None
    min_diff = 1.0  # Initialize minimum difference between FAR and FRR
    with tqdm(total=len(thresholds_unique), desc="Calculating EER") as pbar:
        for threshold in thresholds_unique:
            far, frr = calculate_far_frr(distances, speaker_ids, threshold)
            diff = abs(far - frr)
            #print('far=', far, 'frr=', frr)
            if diff < min_diff:
                #print('far=', far, 'frr=', frr)
                min_diff = diff
                eer = (far + frr) / 2
            pbar.update(1)  # Update progress bar
    return eer

results = []

for model_path in model_paths:

    print(f"Testing model {model_path}")
    
    # Load the model
    fine_tuned_model = torch.load(model_path)
    if isinstance(fine_tuned_model, torch.nn.DataParallel):
        fine_tuned_model = fine_tuned_model.module

    inference = Inference(fine_tuned_model, window="whole")

    # Initialize lists to store speaker IDs and embeddings
    speaker_ids = []
    embeddings = []

    # Iterate through the test dataset
    for audio, id in tqdm(zip(test_dataset['audio'], test_dataset['id']), desc="Calculating embeddings"):
        # print("Audio:", audio)
        # print("ID:", id)
        audio_data = audio
        speaker_id = id.split('+')[1]
        #print(speaker_id)
        speaker_ids.append(speaker_id)
        embedding = extract_embedding(audio_data)
        embeddings.append(embedding)

    # Calculate distances
    distances = calculate_distances(embeddings)

    # Calculate EER
    eer = calculate_eer(distances, speaker_ids)
    print(f"Model: {model_path}, Equal Error Rate (EER): {eer}\n")

    results.append([model_path.split('/')[-2] + "_" + os.path.basename(model_path),"Voxceleb_test",  eer])

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Data", "EER"])
    writer.writerows(results)
