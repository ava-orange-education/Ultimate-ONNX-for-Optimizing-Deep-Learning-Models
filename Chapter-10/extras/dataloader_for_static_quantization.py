"""Data loading functions used during quantization."""

import os
import numpy as np
from typing import Callable
import glob
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import Subset
import torchaudio
import onnxruntime as ort

# Set random seed for reproducibility
np.random.seed(42)


class LibriSpeechDataset(Dataset):
    """Custom Dataset class for loading and preprocessing Libri Speech audio.

    Args:
        root_dir (str): Path to directory containing images
        sample (int): Number of images to randomly sample from the directory.
                     If None, uses all images. Default: 100.
    """

    def __init__(self, root_dir: str, processor: Callable, sample: Optional[int] = 100):
        self.root_dir = root_dir
        self.root_dir = os.path.join(self.root_dir, "dev-clean")
        self.processor = processor

        # Validate root directory exists
        if not os.path.isdir(self.root_dir):
            raise RuntimeError(f"Image root directory not found at {self.root_dir}")

        # Get all audio images in dataset
        self.text_files = glob.glob(os.path.join(self.root_dir, "*/*/*.txt"))

        self.data_map = self.parse_dataset(self.text_files)

        # Randomly sample images if sample size is specified
        if sample is not None:
            if sample > len(self.data_map):
                raise ValueError(
                    f"Sample size {sample} exceeds available images ({len(self.data_map)})"
                )
            self.audio_paths = np.random.choice(list(self.data_map.keys()), sample)

    def parse_dataset(self, text_files):
        transcript_dict = {}

        for text_file in text_files:
            with open(text_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # skip empty lines
                    file_name, transcript = line.split(" ", 1)
                    folder_1, folder_2, _ = file_name.split("-")
                    file_path = os.path.join(self.root_dir, folder_1, folder_2, file_name + ".flac")
                    transcript_dict[file_path] = transcript

        return transcript_dict
        
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.audio_paths)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Loads and preprocesses an image at the given index.

        Args:
            index (int): Index of the image to load

        Returns:
            dict: Dictionary containing the preprocessed image with key 'images'
        """
        audio_path = self.audio_paths[index]
        transcription = self.data_map[audio_path]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        results = self.processor(waveform, sample_rate)
        results["transcription"] = transcription
        return results


class WhisperDataReader(CalibrationDataReader):
    """Adapter class to make PyTorch DataLoader work with ONNX Runtime calibration.

    Args:
        data_loader: PyTorch DataLoader yielding batches of preprocessed data
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)
        self.length = len(data_loader)

    def get_next(self) -> Optional[Dict[str, Any]]:
        """Gets the next batch of data for calibration.

        Returns:
            dict: A dictionary mapping input names to numpy arrays, or None if no more data
        """
        try:
            processed_data = next(self.iter)
            processed_data.pop("transcription")
            return processed_data
        except StopIteration:
            return None

    def __len__(self) -> int:
        """Returns the number of batches in the data loader."""
        return self.length

    def set_range(self, start_index: int, end_index: int) -> None:
        """Sets the data range for calibration.

        Args:
            start_index: Starting index of the subset
            end_index: Ending index of the subset (exclusive)
        """
        dl = Subset(self.data_loader, indices=range(start_index, end_index))
        self.iter = iter(dl)
        self.length = len(dl)  # Update length to reflect subset size

    def rewind(self) -> None:
        """Resets the iterator to the beginning of the dataset."""
        self.iter = iter(self.data_loader)
        self.length = len(self.data_loader)


def encoder_processor(whisper_model_name="openai/whisper-tiny.en"):
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    def apply(waveform, sample_rate):
        results = processor(waveform[0], sampling_rate=sample_rate, return_tensors="np")
        return {'input_features': results["input_features"]}
    return apply

def decoder_init_processor(encoder_onnx_path, whisper_model_name="openai/whisper-tiny.en"):
    encoder_sess = ort.InferenceSession(encoder_onnx_path)
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    SOT_TOKEN = 50257           # Start of transcript token
    decoder_input_ids = np.array([[SOT_TOKEN]], dtype=np.int64)

    def apply(waveform, sample_rate):
        results = processor(waveform[0], sampling_rate=sample_rate, return_tensors="np")
        encoder_hidden_states = encoder_sess.run(["encoder_output"], {"input_features": results["input_features"]})[0]  # (1, seq_len, hidden_size)
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }

    return apply

def decoder_step_processor(encoder_onnx_path, decoder_init_path, decoder_step_path, whisper_model_name="openai/whisper-tiny.en"):
    encoder_sess = ort.InferenceSession(encoder_onnx_path)
    decoder_init_sess = ort.InferenceSession(decoder_init_path)
    decoder_step_sess = ort.InferenceSession(decoder_step_path)
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    SOT_TOKEN = 50257           # Start of transcript token
    decoder_input_ids_start = np.array([[SOT_TOKEN]], dtype=np.int64)
    num_layers = 4

    def apply(waveform, sample_rate):
        results = processor(waveform[0], sampling_rate=sample_rate, return_tensors="np")
        encoder_hidden_states = encoder_sess.run(["encoder_output"], {"input_features": results["input_features"]})[0]  # (1, seq_len, hidden_size)
                
        # Run decoder_init model
        inputs = {
            "decoder_input_ids": decoder_input_ids_start,
            "encoder_hidden_states": encoder_hidden_states,
        }
        init_outputs = decoder_init_sess.run(None, inputs)
        
        logits = init_outputs[0]
        generated_ids = [np.argmax(logits[0, -1])]
        
        past_self_key = init_outputs[1::4]
        past_self_value = init_outputs[2::4]
        past_cross_key = init_outputs[3::4]
        past_cross_value = init_outputs[4::4]

        # Randomly iterate for 1 to 30 times. The last iteration's inputs will be used as calibration data for step model.
        NUM_ITERS = np.random.randint(1, 30)
        for step in range(NUM_ITERS):
            decoder_input_ids = np.array([[generated_ids[-1]]], dtype=np.int64)

            step_inputs = {
                "decoder_input_ids": decoder_input_ids,
                "cache_position": np.array([step + 1], dtype=np.int64),
            }

            # Flatten past_kv to feed as input
            for i in range(num_layers):
                step_inputs[f"past_self_key_{i}"] = past_self_key[i]
                step_inputs[f"past_self_value_{i}"] = past_self_value[i]
                step_inputs[f"past_cross_key_{i}"] = past_cross_key[i]
                step_inputs[f"past_cross_value_{i}"] = past_cross_value[i]

            step_outputs = decoder_step_sess.run(None, step_inputs)

            logits = step_outputs[0]
            next_token = int(np.argmax(logits[0, -1]))

            generated_ids.append(next_token)

            # Update KV cache for next step
            past_self_key = step_outputs[1::4]
            past_self_value = step_outputs[2::4]
            past_cross_key = step_outputs[3::4]
            past_cross_value = step_outputs[4::4]
        
        return step_inputs

    return apply
