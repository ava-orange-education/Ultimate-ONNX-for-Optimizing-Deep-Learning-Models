import onnx
from onnx import shape_inference
from onnx import helper, shape_inference
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from onnx import ModelProto
from typing import Dict


def infer_shapes_with_onnx(model: ModelProto) -> ModelProto:
    """
    Perform shape inference using ONNX's built-in shape inference.

    Args:
        model (ModelProto): ONNX Model

    Returns:
        onnx.ModelProto: Model with inferred shapes.
    """
    # Run shape inference
    inferred_model = shape_inference.infer_shapes(model)

    # Optional: Check for errors
    onnx.checker.check_model(inferred_model)

    return inferred_model


def promote_shaped_outputs_to_graph(model_path: str, output_model_path: str) -> None:
    """
    Checks all node outputs in the ONNX model and promotes them to graph outputs
    if they have shape information.

    Args:
        model_path (str): Path to input ONNX model
        output_model_path (str): Path to save modified ONNX model
    """
    # Load the model
    model = onnx.load(model_path)

    # Run shape inference to ensure shapes are populated
    model = infer_shapes_with_onnx(model)

    # Get all existing output names to avoid duplicates
    existing_outputs = {output.name for output in model.graph.output}

    # Dictionary to track value_info (contains shape information)
    value_info = {vi.name: vi for vi in model.graph.value_info}

    # Collect all node outputs with shape information
    new_outputs = []
    for node in model.graph.node:
        for output_name in node.output:
            if output_name in value_info and output_name not in existing_outputs:
                # Create new output with the same type and shape
                vi = value_info[output_name]
                new_output = helper.make_tensor_value_info(
                    name=vi.name,
                    elem_type=vi.type.tensor_type.elem_type,
                    shape=[dim.dim_value for dim in vi.type.tensor_type.shape.dim],
                )
                new_outputs.append(new_output)

    # Add the new outputs to the graph
    model.graph.output.extend(new_outputs)

    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Added {len(new_outputs)} new outputs with shape information")
    print(f"Saved modified model to {output_model_path}")


def compare_models(
    fp32_model_path: str,
    int8_model_path: str,
    input_data: Dict,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Dict:
    """
    Compare outputs between FP32 and INT8 ONNX models with detailed analysis.

    Args:
        fp32_model_path (str): Path to FP32 ONNX model
        int8_model_path (str): Path to INT8 quantized ONNX model
        input_data (Dict): Dictionary of input data (e.g., {'input_name': np.array})
        rtol (float): Relative tolerance for comparison
        atol (float): Absolute tolerance for comparison

    Returns:
        Dict: Comparison results with statistics and mismatch details
    """
    # Initialize sessions
    fp32_sess = ort.InferenceSession(
        fp32_model_path, providers=["CPUExecutionProvider"]
    )
    int8_sess = ort.InferenceSession(
        int8_model_path, providers=["CPUExecutionProvider"]
    )

    # Run inference
    fp32_outputs = fp32_sess.run(None, input_data)
    int8_outputs = int8_sess.run(None, input_data)

    # Get output names (handle models with different output orders)
    fp32_names = [output.name for output in fp32_sess.get_outputs()]
    int8_names = [output.name for output in int8_sess.get_outputs()]
    common_names = list(set(fp32_names) & set(int8_names))

    results = defaultdict(dict)

    for name in common_names:
        # Get corresponding outputs
        fp32_idx = fp32_names.index(name)
        int8_idx = int8_names.index(name)
        fp32_arr = fp32_outputs[fp32_idx]
        int8_arr = int8_outputs[int8_idx]

        # Basic validation
        if fp32_arr.shape != int8_arr.shape:
            raise ValueError(
                f"Shape mismatch for {name}: FP32 {fp32_arr.shape} vs INT8 {int8_arr.shape}"
            )

        # Calculate metrics
        abs_diff = np.abs(fp32_arr - int8_arr)
        rel_diff = abs_diff / (np.abs(fp32_arr) + 1e-9)  # Avoid division by zero

        results[name] = {
            "max_abs_diff": np.max(abs_diff),
            "mean_abs_diff": np.mean(abs_diff),
            "max_rel_diff": np.max(rel_diff),
            "mean_rel_diff": np.mean(rel_diff),
            "fp32_range": (np.min(fp32_arr), np.max(fp32_arr)),
            "int8_range": (np.min(int8_arr), np.max(int8_arr)),
            "is_close": np.allclose(fp32_arr, int8_arr, rtol=rtol, atol=atol),
            "mismatch_percentage": np.mean((abs_diff > atol) & (rel_diff > rtol)) * 100,
            "abs_diff_stddev": np.std(abs_diff),
            "fp32_output_sample": fp32_arr.flatten()[:5],  # First 5 elements
            "int8_output_sample": int8_arr.flatten()[:5],
        }

    # Add global statistics
    results["_global"] = {
        "num_outputs_compared": len(common_names),
        "outputs_with_mismatches": sum(
            1 for name in common_names if not results[name]["is_close"]
        ),
        "avg_mismatch_percentage": np.mean(
            [results[name]["mismatch_percentage"] for name in common_names]
        ),
        "worst_output": max(common_names, key=lambda x: results[x]["max_rel_diff"]),
    }

    return dict(results)


def save_comparison_results(
    fp32_model_path: ModelProto,
    comparison_results: Dict,
    output_file: str = "model_comparison.txt",
) -> None:
    """
    Save model comparison results to a text file.

    Args:
        fp32_model_path (ModelProto): FP32 model reference to parse the node
            names in order.
        comparison_results (Dict): Results from compare_models()
        output_file (str): Path to output text file
    """
    fp32_model = onnx.load(fp32_model_path)

    with open(output_file, "w") as f:
        # Write summary
        f.write("=== Model Comparison Results ===\n")
        f.write(f"FP32 vs INT8 Model Output Analysis\n\n")

        f.write("=== Summary ===\n")
        f.write(
            f"Compared {comparison_results['_global']['num_outputs_compared']} outputs\n"
        )
        f.write(
            f"{comparison_results['_global']['outputs_with_mismatches']} outputs exceeded tolerance\n"
        )
        f.write(f"Worst output: {comparison_results['_global']['worst_output']}\n")
        f.write(
            f"Average mismatch percentage: {comparison_results['_global']['avg_mismatch_percentage']:.2f}%\n\n"
        )

        # Write detailed results
        f.write("=== Detailed Output Comparison ===\n")

        for node in reversed(fp32_model.graph.node):
            for output_name in node.output:
                if output_name not in comparison_results:
                    continue

                metrics = comparison_results[output_name]
                if metrics["abs_diff_stddev"] > 1:
                    print(
                        f"Tensor: {output_name}, Abs Diff Stddev: {metrics['abs_diff_stddev']:.2f}"
                    )

                f.write(f"\nOutput: {output_name}\n")
                f.write(
                    f"  FP32 range: [{metrics['fp32_range'][0]:.6f}, {metrics['fp32_range'][1]:.6f}]\n"
                )
                f.write(
                    f"  INT8 range: [{metrics['int8_range'][0]:.6f}, {metrics['int8_range'][1]:.6f}]\n"
                )
                f.write(f"  Max absolute difference: {metrics['max_abs_diff']:.6f}\n")
                f.write(f"  Mean absolute difference: {metrics['mean_abs_diff']:.6f}\n")
                f.write(
                    f"  Max relative difference: {metrics['max_rel_diff']:.6f} ({metrics['max_rel_diff'] * 100:.2f}%)\n"
                )
                f.write(
                    f"  Mismatch percentage: {metrics['mismatch_percentage']:.2f}%\n"
                )
                f.write(
                    f"  Standard deviation of absolute difference: {metrics['abs_diff_stddev']:.2f}\n"
                )
                f.write(
                    f"  Within tolerance: {'Yes' if metrics['is_close'] else 'No'}\n"
                )
                f.write(
                    f"  FP32 sample values: {np.array2string(metrics['fp32_output_sample'], precision=4)}\n"
                )
                f.write(
                    f"  INT8 sample values: {np.array2string(metrics['int8_output_sample'], precision=4)}\n"
                )

        f.write("\n=== END OF REPORT ===")


# Usage Example
if __name__ == "__main__":
    # Prepare sample input
    # This should be generated from the calibration dataset class.
    input_sample = {
        "images": np.fromfile(
            "../Chapter-8/yolov12/sample_input.raw", dtype=np.float32
        ).reshape(1, 3, 640, 640)
    }

    fp32_path = "../Chapter-8/yolov12/yolov12n.onnx"
    fp32_path_output = "../Chapter-8/yolov12/yolov12n_updated.onnx"
    int8_path = "../Chapter-8/yolov12/yolov12n_int8_static_quant.onnx"
    int8_path_output = "../Chapter-8/yolov12/yolov12n_int8_static_quant_updated.onnx"
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
