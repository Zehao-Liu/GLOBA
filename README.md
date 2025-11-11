# GLOBA: Rethinking Parameter Conflicts in Model Merging

This repository is the official implementation for the paper **GLOBA: Rethinking Parameter Conflicts in Model Merging**.

Our work introduces **GLOBA (GLObal Basis Analysis Framework)**, a novel training-free model merging technique. GLOBA operates from a geometric perspective, mathematically proving that parameter conflicts arise exclusively from non-orthogonal components of task vectors. By projecting models into a unified global coordinate system, GLOBA can precisely extract conflict-free orthogonal components and classify overlapping parameters into five distinct types. This allows for adaptive merging strategies that selectively mitigate harmful conflicts while retaining beneficial synergies.

If you have any questions, please feel free to open an issue on this repository.

## Abstract

Model merging serves as a training-free technique that combines multiple task-specific models into a unified multi-task model, but parameter conflicts often lead to performance drops. Previous methods flatten weight matrices into one-dimensional vectors, losing the inherent structural information of their row and column spaces. We mathematically prove and experimentally validate that parameter conflicts arise from non-orthogonal components of task vectors, while orthogonal components are conflict-free. Furthermore, we find that non-orthogonal components can contain both harmful conflicts and beneficial synergies. To precisely locate parameter conflicts and extract orthogonal components, we propose GLOBA (GLObal Basis Analysis Framework), which projects task vectors onto a global basis to align them within a unified coordinate system and construct a task interaction matrix. Following energy-based pruning, we divide parameters into five types based on the orthogonal relationships between the row spaces and column spaces of task vectors. Experiments on three fine-tuned models (mathematics, coding, and instruction-following) using LLaMA-2-7B and LLaMA-2-13B demonstrate significant performance gains through selective retention of beneficial parameters and removal of conflicting ones.

## The GLOBA Framework

Our method offers a principled approach to understand and resolve parameter conflicts by analyzing the geometric structure of task vectors (`delta = Model_Finetuned - Model_Base`).

The GLOBA pipeline consists of several key steps:

1.  **Low-Rank Approximation**: We apply SVD to the task vectors of the models to be merged, retaining the most significant components based on an energy threshold. This step denoises the task vectors and captures their essential information in a low-rank format.
2.  **Global Basis Construction**: We construct a shared, global basis from the singular vectors of all models. This creates a unified subspace where different models' parameters can be directly compared and analyzed.
3.  **Projection and Disentanglement**: We project the low-rank task vectors onto this global basis, resulting in small, core "task interaction matrices" (`C1`, `C2`).
4.  **Modular Analysis**: The key innovation lies here. We analyze the structural relationship between `C1` and `C2` to partition `C2` into six mutually exclusive and complete components based on how they interact with `C1`:
    * **Type D (Direct Overlap)**: Parameters modified by both models at the same position. Sub-divided into `D+` (same-sign) and `D-` (opposite-sign).
    * **Type E (Structural Overlap)**: Parameters in `C2` that fill "structural holes" in `C1`, i.e., positions where `C1` is zero but the corresponding row and column are otherwise occupied by `C1`.
    * **Type C (Pure Column Orthogonality)**: `C2` parameters that share a row with `C1`'s parameters but not a column.
    * **Type B (Pure Row Orthogonality)**: `C2` parameters that share a column with `C1`'s parameters but not a row.
    * **Type A (Complete Orthogonality)**: `C2` parameters that are in rows and columns completely untouched by `C1`.
5.  **Flexible Merging**: Each of these six components, plus the base `C1` matrix, can be independently scaled with a coefficient before being summed up to create the final merged model. This allows for highly customized merging strategies.

## How to Use GLOBA (`svd_merging`)

The GLOBA method is implemented as the `svd_merging` function. Its behavior is entirely controlled by a hyperparameter block at the top of the function. To change the merging strategy, you only need to edit this section in `merging_methods.py`.

```python
# Location: merging_methods.py -> MergingMethod class -> svd_merging function

# =================================================================================
# --- HYPERPARAMETER CONTROL AREA (Modify here to control merge strategy) ---
# =================================================================================
# SVD energy threshold for the initial delta matrices (task vectors) in Step 1
svd_energy_threshold_model1 = 0.90
svd_energy_threshold_model2 = 0.99

# Energy-based pruning threshold for the projected C matrices in Step 4
c_pruning_energy_c1 = 0.95
c_pruning_energy_c2 = 0.80

# Scaling coefficients for C matrix components. Set to 0 to disable a component.
scale_c1_identity = 1.0       # The base C1 matrix itself

# --- The following are the six orthogonal components from C2 ---
scale_c2_type_D_minus = 1.0   # Type D- (Opposite-sign Direct Overlap)
scale_c2_type_D_plus = 0.0    # Type D+ (Same-sign Direct Overlap)
scale_c2_type_E = 1.0         # Type E (Structural Overlap)
scale_c2_type_B = 0.0         # Type B (Pure Row Orthogonality)
scale_c2_type_C = 0.0         # Type C (Pure Column Orthogonality)
scale_c2_type_A = 0.0         # Type A (Complete Orthogonality)

# Energy threshold for creating the global basis vectors
global_basis_energy_threshold = 0.999

# Averaging weights for layers skipped during SVD (e.g., bias, norm layers)
skip_layer_avg_weight_model1 = 0.5
skip_layer_avg_weight_model2 = 0.5
# =================================================================================
```

By changing the `scale_c2_*` values, you can implement the selective merging strategies discussed in the paper.

## Running Experiments

### Environment Setup

* [PyTorch 2.4.0](https://pytorch.org/)
* [transformers 4.33.1](https://huggingface.co/docs/transformers/index)
* [datasets 2.13.1](https://huggingface.co/docs/datasets/index)
* [vllm 0.1.4](https://github.com/vllm-project/vllm)
* [numpy](https://github.com/numpy/numpy)
* [tqdm](https://github.com/tqdm/tqdm)

### Models and Datasets

Our experiments use fine-tuned models based on LLaMA-2 backbones.
* **Instruction-Following**: `Llama-2-7b-instruct`, `WizardLM-13B-V1.0` (on AlpacaEval).
* **Mathematical Reasoning**: `WizardMath-7B-V1.0`, `WizardMath-13B-V1.0` (on GSM8K).
* **Code Generation**: `llama2_7b_code`, `llama-2-13b-code-alpaca` (on MBPP).

The GSM8K, MATH, and MBPP datasets should be placed in the `math_code_data/` folder. Other datasets and all models can be automatically downloaded. You can modify the `cache_dir` in `utils/load_config.py` to specify your own storage path.

### Execution Scripts

The primary script for merging is `merge_llms_instruct_math_code.py`.

* **Example of merging with GLOBA (`svd_merging`):**
    To merge an instruction model and a math model, run the following command. The actual merging strategy is determined by the hyperparameters you set inside the `svd_merging` function.
    ```bash
    python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name svd_merging --tensor_parallel_size 1
    ```

* **Example of merging with Baselines (e.g., Task Arithmetic):**
    ```bash
    python merge_llms_instruct_math_code.py --merge_instruct --merge_math --merging_method_name task_arithmetic --scaling_coefficient 1.0 --tensor_parallel_size 1
    ```

❗**Note 1**: The number of required GPUs is `num_models_to_merge * tensor_parallel_size`. For merging two 13B models with `tensor_parallel_size=1`, you need `2 * 1 = 2` GPUs.

❗**Note 2**: If you encounter a `vllm` error like "AssertionError: data parallel group is already initialized", please use the `direct_inference_merged_llms_instruct_math_code.py` script as a fallback for evaluation.

### Evaluation Process

For some datasets, evaluation is a two-step process. First, run the main script to generate output files, then use the official evaluation scripts.

* **For AlpacaEval:**
    We use the **AlpacaEval 2.0** framework. Our evaluation follows a specific custom setup:

    1.  **Download Reference Outputs:** Download the `model_outputs.json` file from the [`alpaca_eval/results/text_davinci_001/`](https://github.com/tatsu-lab/alpaca_eval/tree/main/results/text_davinci_001) directory in the official repository. This file will serve as the ground truth reference for the evaluator.

    2.  **Configure the Judge Model:** In your local `alpaca_eval` environment, modify the configuration for the `chatgpt_fn` annotator to change the judge model from its default to **`gpt-4o-mini`**.

    3.  **Prepare Your Output File:** Before running the evaluation, ensure that you have set your desired model name within your generated model outputs file (the JSON file you are evaluating).

    4.  **Run Evaluation:** Execute the evaluation command, providing your model's outputs and the downloaded reference outputs.

    ```bash
    alpaca_eval --model_outputs ./path/to/your/generated_outputs.json \
                --reference_outputs ./path/to/the/downloaded_model_outputs.json \
                --annotators_config chatgpt_fn
    ```

* **For MBPP:**
    Please first install the environment from the [bigcode-evaluation-harness repository](https://github.com/bigcode-project/bigcode-evaluation-harness). Then, run:
    ```bash
    accelerate launch ./bigcode-evaluation-harness/main.py --tasks mbpp --allow_code_execution --load_generations_path ./path/to/your/generated_output.jsonl
    ```
```