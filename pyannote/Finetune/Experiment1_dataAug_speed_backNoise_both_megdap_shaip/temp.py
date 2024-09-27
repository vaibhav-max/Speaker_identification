import os
from scipy.spatial.distance import cdist
from pyannote.audio import Inference
import numpy as np
from tqdm import tqdm
import torch

# Paths and parameters
fine_tuned_model_path = "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1_dataAug_speed_backNoise_both_megdap_shaip/Models/pyannote.audio.models.embedding.XVectorSincNet_epoch13_EER0.0003.pt"
dataset_folders = [
    "/data/Root_content/Vaani/Speaker_ID/Dataset/megdap/test",
    "/data/Root_content/Vaani/Speaker_ID/Dataset/megdap/val",
    "/data/Root_content/Vaani/Speaker_ID/Dataset/shaip/test",
    "/data/Root_content/Vaani/Speaker_ID/Dataset/shaip/val"
]

num_files_to_process = 100 # Set None to process all files

# Load the fine-tuned model
def load_model(model_path):
    model = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model

# Initialize Inference object
def initialize_inference(model):
    return Inference(model, window="whole")

# Extract embeddings from audio files
def extract_embedding(audio_path, inference):
    embedding = inference(audio_path)
    return embedding.reshape(1, -1)

# Calculate distances for all pairs of embeddings
def calculate_distances(embeddings):
    embeddings_concatenated = np.concatenate(embeddings, axis=0)
    distances = cdist(embeddings_concatenated, embeddings_concatenated, metric="cosine")
    return distances

# Calculate FAR and FRR
def calculate_far_frr(distances, speaker_ids, threshold):
    genuine_scores = []
    impostor_scores = []
    for i in range(len(distances)):
        for j in range(len(distances)):
            if i != j:
                if speaker_ids[i] == speaker_ids[j]:
                    genuine_scores.append(distances[i][j])
                else:
                    impostor_scores.append(distances[i][j])

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    far = np.sum(impostor_scores <= threshold) / len(impostor_scores)
    frr = np.sum(genuine_scores > threshold) / len(genuine_scores)
    
    return far, frr

# Calculate EER
def calculate_eer(distances, speaker_ids):
    thresholds_rounded = np.around(distances, decimals=1)  
    thresholds_unique = np.unique(thresholds_rounded)

    eer = None
    min_diff = 1.0  
    with tqdm(total=len(thresholds_unique), desc="Calculating EER") as pbar:
        for threshold in thresholds_unique:
            far, frr = calculate_far_frr(distances, speaker_ids, threshold)
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
            pbar.update(1)
    return eer

# Main function for testing multiple folders
def test_multiple_folders(model, folders, num_files_to_process):
    folder_eers = {}
    inference = initialize_inference(model)
    
    for folder in folders:
        speaker_ids = []
        embeddings = []
        num_files_processed = 0

        folder_name = folder.split('/')[-2] + '_' + os.path.basename(folder)
        #print(folder_name)

        for filename in tqdm(os.listdir(folder), desc=f"Processing folder {folder_name}"):
            if num_files_processed == num_files_to_process:
                break
            if filename.endswith(".wav"):
                audio_path = os.path.join(folder, filename)
                speaker_id = filename.split('_')[2]  
                speaker_ids.append(speaker_id)
                embedding = extract_embedding(audio_path, inference)
                embeddings.append(embedding)
                num_files_processed += 1
        
        distances = calculate_distances(embeddings)
        eer = calculate_eer(distances, speaker_ids)
        folder_eers[folder_name] = eer
        print(f"EER for folder {folder_name}: {eer}")

    return folder_eers

if __name__ == "__main__":
    model = load_model(fine_tuned_model_path)
    folder_eers = test_multiple_folders(model, dataset_folders, num_files_to_process)
    
    print("\nFinal EERs for each folder:")
    for folder, eer in folder_eers.items():
        print(f"{os.path.basename(folder)}: {eer}")
