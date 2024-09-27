import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio
from pyannote.audio import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
import random

# Enable device-side assertions
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Parameters and Paths
config_file_path = "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1_dataAug_speed_backNoise_both_megdap_shaip/config.yaml"

train_data_dir_1 = "/data/Root_content/Vaani/Speaker_ID/Dataset/shaip/train"
train_data_dir_2 = "/data/Root_content/Vaani/Speaker_ID/Dataset/megdap/train"
train_data_dir = [train_data_dir_1, train_data_dir_2]

val_data_dir_1 = "/data/Root_content/Vaani/Speaker_ID/Dataset/shaip/val"
val_data_dir_2 = "/data/Root_content/Vaani/Speaker_ID/Dataset/megdap/val"
val_data_dir = [val_data_dir_1, val_data_dir_2]

noisy_data_dir = "/data/Root_content/Vaani/Speaker_ID/Dataset/Noise/Combined"  # Add your noisy audio folder path here

output_model_dir = "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1_dataAug_speed_backNoise_both_megdap_shaip/Models"
output_plot_dir = "/data/Root_content/Vaani/Speaker_ID/pyannote/Fine_tune/Experiment1_dataAug_speed_backNoise_both_megdap_shaip/Plots"
max_train_samples = None # if not required, set to None
max_val_samples = None
num_classes = 8750
speeds = [0.9, 1.1]

# Load YAML configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Plot training results
def plot_training_results(train_loss_list, train_accuracy_list, eer_list, num_epochs, output_dir):
    epochs = range(1, num_epochs + 1)

    # Plot training loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_list, label='Train Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))
    plt.close()

    # Plot training accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_accuracy_list, label='Train Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_accuracy.png"))
    plt.close()

    # Plot EER
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, eer_list, label='EER', color='red')
    plt.title('Equal Error Rate (EER)')
    plt.xlabel('Epochs')
    plt.ylabel('EER')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eer.png"))
    plt.close()

def adjust_power_to_target(signal, target_power):
    current_power = torch.mean(signal ** 2)
    scaling_factor = torch.sqrt(target_power / current_power)
    adjusted_signal = signal * scaling_factor
    return adjusted_signal

# Custom Dataset class with data augmentation
class CustomDataset(Dataset):
    def __init__(self, root_dirs, noisy_dir, max_samples=None, transform=None):
        self.root_dirs = root_dirs
        self.noisy_dir = noisy_dir
        self.file_list = []
        for root_dir in root_dirs:
            files = os.listdir(root_dir)[:max_samples] if max_samples else os.listdir(root_dir)
            self.file_list.extend([(root_dir, file) for file in files])
        self.transform = transform
        self.max_length = self._calculate_max_length()
        self.speaker_to_label = self._create_speaker_to_label()
        self.noisy_files = os.listdir(noisy_dir)

    def _calculate_max_length(self):
        max_length = 0
        for root_dir, file_name in self.file_list:
            file_path = os.path.join(root_dir, file_name)
            waveform, sample_rate = torchaudio.load(file_path)
            waveform_length = waveform.shape[1]
            max_length = max(max_length, waveform_length)
        return max_length

    def _create_speaker_to_label(self):
        speaker_to_label = {}
        label_counter = 0
        for root_dir, file_name in self.file_list:
            speaker_id = file_name.split('_')[2]
            if speaker_id not in speaker_to_label:
                speaker_to_label[speaker_id] = label_counter
                label_counter += 1
        print('label counter', label_counter)
        return speaker_to_label

    def __len__(self):
        #print("length",len(self.file_list) * 3  )
        return len(self.file_list) * 3  # Each original audio, speed perturbed audio, and noisy audio once

    def __getitem__(self, idx):
        original_idx = idx // 3
        perturb_type = idx % 3
       
        root_dir, file_name = self.file_list[original_idx]
        file_path = os.path.join(root_dir, file_name)

        # Load original audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample the audio to 16 kHz if the sampling rate is different
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Ensure the audio is mono (if not already)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if perturb_type == 1:  # Apply speed perturbation
            speed_idx = random.randint(0, 1)
            speed_factor = speeds[speed_idx]
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed_factor)], ['rate', str(sample_rate)]])
            waveform = wav

        # Apply augmentation by mixing with a noisy audio file if required
        elif perturb_type == 2: 
            snr_range = [0, 15]
            
            noisy_file = random.choice(self.noisy_files)
            noisy_path = os.path.join(self.noisy_dir, noisy_file)
            noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_path)

            # Resample the noisy audio to 16 kHz if the sampling rate is different
            if noisy_sample_rate != 16000:
                resampler = Resample(orig_freq=noisy_sample_rate, new_freq=16000)
                noisy_waveform = resampler(noisy_waveform)

            # Ensure the noisy audio is mono (if not already)
            if noisy_waveform.shape[0] > 1:
                noisy_waveform = noisy_waveform.mean(dim=0, keepdim=True)

            target_snr = random.uniform(snr_range[0], snr_range[1])
            target_power = torch.mean(waveform ** 2) / (10 ** (target_snr / 10))
            noisy_waveform = adjust_power_to_target(noisy_waveform, target_power)
            
            # # Trim or pad noisy waveform to match the length of the original waveform
            noisy_waveform = self._adjust_waveform_length(noisy_waveform, target_length=waveform.shape[1])

            # Mix original and noisy audio
            waveform = waveform + noisy_waveform

        # Apply preprocessing transformations if specified
        if self.transform:
            waveform = self.transform(waveform)

        # Pad or trim waveform to the maximum length
        waveform = self._adjust_waveform_length(waveform, target_length=self.max_length)

        # Get speaker ID from filename and convert to label
        speaker_id = file_name.split('_')[2]
        label = self.speaker_to_label[speaker_id]

        return waveform, label

    def _adjust_waveform_length(self, waveform, target_length):
        waveform_length = waveform.shape[1]
        if waveform_length < target_length:
            # Pad waveform if shorter than target length
            padding = target_length - waveform_length
            waveform = nn.functional.pad(waveform, (0, padding))
        elif waveform_length > target_length:
            # Trim waveform if longer than target length
            waveform = waveform[:, :target_length]
        return waveform

class CustomDatasetVal(Dataset):
    def __init__(self, root_dirs, max_samples=None, transform=None):
        self.root_dirs = root_dirs
        self.file_list = []
        for root_dir in root_dirs:
            files = os.listdir(root_dir)[:max_samples] if max_samples else os.listdir(root_dir)
            self.file_list.extend([(root_dir, file) for file in files])
        self.transform = transform
        self.max_length = self._calculate_max_length()
        self.speaker_to_label = self._create_speaker_to_label()

    def _calculate_max_length(self):
        max_length = 0
        for root_dir, file_name in self.file_list:
            file_path = os.path.join(root_dir, file_name)
            waveform, sample_rate = torchaudio.load(file_path)
            waveform_length = waveform.shape[1]
            max_length = max(max_length, waveform_length)
        return max_length

    def _create_speaker_to_label(self):
        speaker_to_label = {}
        label_counter = 0
        for root_dir, file_name in self.file_list:
            speaker_id = file_name.split('_')[2]
            if speaker_id not in speaker_to_label:
                speaker_to_label[speaker_id] = label_counter
                label_counter += 1
        print(label_counter)
        return speaker_to_label

    def __len__(self):
        #print("length_val" ,len(self.file_list) )
        return len(self.file_list)

    def __getitem__(self, idx):
        root_dir, file_name = self.file_list[idx]
        file_path = os.path.join(root_dir, file_name)

        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample the audio to 16 kHz if the sampling rate is different
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Ensure the audio is mono (if not already)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply preprocessing transformations if specified
        if self.transform:
            waveform = self.transform(waveform)

        # Pad or trim waveform to the maximum length
        waveform = self._adjust_waveform_length(waveform)

        # Get speaker ID from filename and convert to label
        speaker_id = file_name.split('_')[2]
        label = self.speaker_to_label[speaker_id]
        #print(waveform.shape)
        return waveform, label

    def _adjust_waveform_length(self, waveform):
        waveform_length = waveform.shape[1]
        if waveform_length < self.max_length:
            # Pad waveform if shorter than max length
            padding = self.max_length - waveform_length
            waveform = nn.functional.pad(waveform, (0, padding))
        elif waveform_length > self.max_length:
            # Trim waveform if longer than max length
            waveform = waveform[:, :self.max_length]
        return waveform
    
def calculate_distances(embeddings):
    # Calculate cosine distances between embedding vectors
    distances = cosine_distances(embeddings)
    return distances

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
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
            pbar.update(1)  # Update progress bar
    return eer

def fine_tune_model(config, model, device, max_train_samples=None, max_val_samples=None):
    try:
        # Move model to GPU
        model.to(device)

        # Prepare training dataset and data loader
        print("Train Dataloader")
        train_dataset = CustomDataset(train_data_dir, noisy_data_dir, max_samples=max_train_samples)
        train_dataloader = DataLoader(train_dataset, shuffle=True, **config['dataloader_args'])

        # Prepare validation dataset and data loader
        print("Validation Dataloader")
        val_dataset = CustomDatasetVal(val_data_dir, max_samples=max_val_samples)
        val_dataloader = DataLoader(val_dataset, **config['dataloader_args'])

        # Define loss function, optimizer, scheduler
        criterion = getattr(nn, config['loss'])(**config['loss_args'])
        optimizer = getattr(optim, config['optimizer'])(model.parameters(), **config['optimizer_args'])

        train_loss_list = []
        train_accuracy_list = []
        eer_list = []  # Initialize list to store EER for each epoch
        
        # Fine-tuning training loop
        model.train()
        for epoch in range(config['num_epochs']):
            # Training
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            model.train()
            for data, target in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} Training"):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
            train_loss /= len(train_dataset)
            train_accuracy = correct_train / total_train
            train_accuracy_list.append(train_accuracy)  # Append training accuracy to the list

            # Validation
            model.eval()
            y_true = []
            y_score = []
            with torch.no_grad():
                for data, target in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} Validation"):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    y_true.extend(target.cpu().numpy())
                    y_score.extend(output.cpu().numpy())

            distances = calculate_distances(y_score)

            # Calculate EER
            eer = calculate_eer(distances, y_true)
            eer_list.append(eer)  # Append EER to the list

            # Print training loss and EER
            print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, EER: {eer:.4f}")

            # Save the model after every epoch
            architecture_name = config['model']
            save_path = os.path.join(output_model_dir, f"{architecture_name}_epoch{epoch+1}_EER{eer:.4f}.pt")
            torch.save(model, save_path)

            train_loss_list.append(train_loss)
            print("Train_accuracy_list", train_accuracy_list)
            print("Train_loss_list", train_loss_list) 
            print("EER_list", eer_list)

        plot_training_results(train_loss_list, train_accuracy_list, eer_list, config['num_epochs'], output_plot_dir)

    except RuntimeError as e:
        # Handle CUDA errors
        print(f"Error occurred: {e}")
        # Free up GPU memory
        torch.cuda.empty_cache()
        # Restart the training process
        fine_tune_model(config, model, device, max_train_samples, max_val_samples)

if __name__ == "__main__":

    # Load YAML configuration
    config = load_config(config_file_path)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model
    #model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="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")

    # Modify the resnet.seg_1 layer
    # in_features = model.resnet.seg_1.in_features  # Get the number of input features to the layer
    # model.resnet.seg_1 = nn.Linear(in_features=in_features, out_features=num_classes)

    # Add the extra layer
    extra_layer = nn.Linear(in_features=512, out_features=num_classes)
    model.embedding = nn.Sequential(model.embedding, extra_layer)
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Fine-tune the model
    fine_tune_model(config, model, device, max_train_samples, max_val_samples)
