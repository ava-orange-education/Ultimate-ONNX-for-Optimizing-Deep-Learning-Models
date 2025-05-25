
import sys
sys.path.append("../utils")
import numpy as np
from inspect_int8_outputs import promote_shaped_outputs_to_graph, compare_models, save_comparison_results
from transformers import WhisperProcessor
from datasets import load_dataset

# Usage Example
if __name__ == "__main__":
    # Prepare sample input

    # load librispeech ASR dataset and read audio files
    ds = load_dataset("librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
    sample = next(iter(ds))

    # Load processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    # Load and process audio
    waveform = sample['audio']['array']

    # Convert to log-mel spectrogram
    inputs = processor(waveform, sampling_rate=16000, return_tensors="np")
    print(inputs["input_features"].shape)
    input_sample = {
        "input_features": inputs["input_features"]
    }
    print(input_sample['input_features'].shape)
    fp32_path = "./whisper_encoder.onnx"
    fp32_path_output = "./whisper_encoder_updated.onnx"
    int8_path = "./whisper_encoder_opt_int8_static_quant.onnx"
    int8_path_output = "./whisper_encoder_opt_int8_static_quant_updated.onnx"
    promote_shaped_outputs_to_graph(fp32_path, fp32_path_output)
    promote_shaped_outputs_to_graph(int8_path, int8_path_output)

    # Compare models
    comparison_results = compare_models(
        fp32_model_path=fp32_path_output,
        int8_model_path=int8_path_output,
        input_data=input_sample,
    )

    # Print to console
    print("\n=== Comparison Summary ===")
    print(f"Compared {comparison_results['_global']['num_outputs_compared']} outputs")
    print(
        f"{comparison_results['_global']['outputs_with_mismatches']} outputs exceeded tolerance"
    )
    print(f"Worst output: {comparison_results['_global']['worst_output']}")
    print(
        f"Average mismatch: {comparison_results['_global']['avg_mismatch_percentage']:.2f}%"
    )

    # Save to file
    save_comparison_results(
        fp32_path_output, comparison_results, "model_comparison_report.txt"
    )
    print("\nSaved detailed report to 'model_comparison_report.txt'")
