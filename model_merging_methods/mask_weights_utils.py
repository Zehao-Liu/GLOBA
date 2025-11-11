import argparse
import jsonlines
import sys
import shutil
import logging
import os
import time
from tqdm import tqdm
import glob
import json
import torch
import torch.nn as nn
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from human_eval.data import write_jsonl, read_problems, stream_jsonl

from model_merging_methods.task_vector import TaskVector
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize, get_param_names_to_merge
from utils.evaluate_llms_utils import batch_data, extract_answer_number, remove_boxed, last_boxed_only_string, process_results, \
    generate_instruction_following_task_prompt, get_math_task_prompt, generate_code_task_prompt, read_mbpp
from utils.load_config import cache_dir

import numpy as np
import re


def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    # if use_rescale and mask_rate != 1.0:
    #     masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    if use_rescale:
        masked_input_tensor = masked_input_tensor * 3
    return masked_input_tensor


def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return: dictionary containing masked weights
    """
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        # Exclude specific parameters
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    elif weight_format == "delta_weight":
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict
    else:
        raise ValueError(f"Invalid weight_format value: {weight_format}")

    # Print included and excluded parameters
    for param_name in finetuned_model.named_parameters():
        if param_name[0] in model_param_dict:
            print(f"Including parameter: {param_name[0]}")  # Included parameter
        else:
            print(f"Excluding parameter: {param_name[0]}")  # Excluded parameter

    # Mask the weights
    with torch.no_grad():
        masked_param_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items(), desc="Masking model weights"):
            # Apply mask to the parameters
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy)

        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_param_dict)
            masked_param_dict = new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)

    return masked_param_dict





