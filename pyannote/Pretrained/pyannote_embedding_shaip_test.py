import os
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
import numpy as np
from tqdm import tqdm
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Instantiate pretrained model
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")
inference = Inference(model, window="whole")
inference.to(torch.device("cuda"))

# Path to your dataset folder
dataset_folder = "/data/Root_content/Vaani/Speaker_ID/Dataset/megdap/val/"

# Limit the number of files to process for testing
num_files_to_process = None

# Define function to extract embeddings
def extract_embedding(audio_path):
    embedding = inference(audio_path)
    #print(embedding.shape)
    return embedding.reshape(1, -1)

# Calculate distances for all pairs of embeddings
'''If you have n embeddings, the shape of the distances array will be (n, n). Each row and column in this matrix 
correspond to a different embedding, and the value at position [i, j] represents the cosine distance between 
the i-th and j-th embeddings.'''
def calculate_distances(embeddings):
    # Concatenate embeddings along the first axis
    embeddings_concatenated = np.concatenate(embeddings, axis=0)
    distances = cdist(embeddings_concatenated, embeddings_concatenated, metric="cosine")
    return distances


# Initialize lists to store speaker IDs and embeddings
speaker_ids = []
embeddings = []

# Iterate through a limited number of .wav files in the dataset folder
for filename in tqdm(os.listdir(dataset_folder)[:num_files_to_process], desc="Calculating embeddings"):
    if filename.endswith(".wav"):
        audio_path = os.path.join(dataset_folder, filename)
        speaker_id = filename.split('_')[2]  # Extract speaker ID
        # print(speaker_id)
        speaker_ids.append(speaker_id)
        embedding = extract_embedding(audio_path)
        #print(embedding.shape)
        embeddings.append(embedding)

# Calculate distances
distances = calculate_distances(embeddings)

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
    min_diff = 1.0  # Initialize minimum difference between FAR and FRR
    with tqdm(total=len(thresholds_unique), desc="Calculating EER") as pbar:
        for threshold in thresholds_unique:
            far, frr = calculate_far_frr(distances, speaker_ids, threshold)
            print("far, frr" , far, frr, threshold)
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
            pbar.update(1)  # Update progress bar
    return eer

eer = calculate_eer(distances, speaker_ids)
print("Equal Error Rate (EER):", eer)

