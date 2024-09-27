import os
from torch.utils.data import Dataset, DataLoader
from torchaudio import load as audio_load
from pyannote.audio.tasks import SpeakerEmbedding
from pyannote.audio.models.embedding import RawEmbedding
from pytorch_lightning import Trainer
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        file_paths = []
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(self.root_dir, split)
            for file in os.listdir(split_dir):
                if file.endswith(".wav"):
                    file_paths.append((os.path.join(split_dir, file), file))
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, file_name = self.file_paths[idx]
        waveform, sample_rate = audio_load(file_path)
        speaker_id = file_name.split('_')[2]
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, speaker_id

# Assuming your_dataset is your dataset object
root_dir = "/home/vaibh/Vaani/Speaker_ID/Dataset/shaip"
your_dataset = CustomDataset(root_dir)

# Define the task
speaker_id_task = SpeakerEmbedding(your_dataset)

# Instantiate the model
model = RawEmbedding(task=speaker_id_task)

# Fine-tune the model
model.freeze_up_to('encoder')  # Freeze early layers if needed
trainer = Trainer()  # Initialize the Trainer
trainer.fit(model)  # Train the model

# Optionally, customize optimizer and scheduler
def configure_optimizers(self):
    optimizer = SGD(self.parameters(), lr=0.001)  # Customize your optimizer
    lr_scheduler = ExponentialLR(optimizer, 0.9)  # Customize your scheduler
    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

model.configure_optimizers = configure_optimizers.__get__(model)  # Override the method
trainer.fit(model)  # Train the model with customized optimizer and scheduler
