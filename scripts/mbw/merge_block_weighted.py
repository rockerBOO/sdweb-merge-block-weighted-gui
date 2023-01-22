# from https://note.com/kohya_ss/n/n9a485a066d5b
# kohya_ss
#   original code: https://github.com/eyriewow/merge-models

# use them as base of this code
# 2022/12/15
# bbc-mc

import os

# import argparse
import re
from typing import Optional, Dict, Callable
import torch
from tqdm import tqdm
from torch import Tensor

from modules import sd_models


NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

KEY_POSITION_IDS = "cond_stage_model.transformer.text_model.embeddings.position_ids"


def dprint(str, flg):
    if flg:
        print(str)


def load_model(model, device="cpu"):
    model_info = sd_models.get_closet_checkpoint_match(model)

    if model_info is None:
        raise RuntimeError("Invalid model")

    model_file = model_info.filename
    state = sd_models.read_state_dict(model_file, map_location=device)

    if state is None:
        raise RuntimeError(f"Could not read file into {device}")

    return state


def merge(
    input_weights: str,
    model_0: str,
    model_1: str,
    device="cpu",
    base_alpha=0.5,
    output_file: Optional[str] = "",
    allow_overwrite=False,
    verbose=False,
    prune: Optional[str] = None,
    save_as_safetensors=False,
    save_as_half=False,
    skip_position_ids=0,
):
    if input_weights is None:
        weights = []
    else:
        weights = [float(w) for w in input_weights.split(",")]

    if len(weights) != NUM_TOTAL_BLOCKS:
        _err_msg = f"weights value must be {NUM_TOTAL_BLOCKS}."
        print(_err_msg)
        return False, _err_msg

    device = device if device in ["cpu", "cuda"] else "cpu"

    print("loading", model_0)
    theta_0: Dict[str, Tensor] = load_model(model_0, device)

    print("loading", model_1)
    theta_1: Dict[str, Tensor] = load_model(model_1, device)

    alpha = base_alpha

    if not output_file or output_file == "":
        output_file = f'bw-{model_0}-{model_1}-{str(alpha)[2:] + "0"}.ckpt'

    # check if output file already exists
    if os.path.isfile(output_file) and not allow_overwrite:
        _err_msg = f"Exiting... [{output_file}]"
        print(_err_msg)
        return False, _err_msg

    print("  merging ...")
    dprint("-- start Stage 1/2 --", verbose)

    convert_to = conv_fp16 if save_as_half else conv_full

    a = stage_one(
        theta_0,
        theta_1,
        weights,
        alpha=alpha,
        verbose=verbose,
        skip_position_ids=skip_position_ids,
        convert_to=convert_to,
    )

    dprint("-- start Stage 2/2 --", verbose)
    final_weights = stage_two(
        a,
        theta_1,
        verbose=verbose,
        skip_position_ids=skip_position_ids,
        convert_to=convert_to,
    )

    print("Saving...")

    _, extension = os.path.splitext(output_file)
    if extension.lower() == ".safetensors" or save_as_safetensors:
        if save_as_safetensors and extension.lower() != ".safetensors":
            output_file = output_file + ".safetensors"
        import safetensors.torch

        safetensors.torch.save_file(
            final_weights, output_file, metadata={"format": "pt"}
        )
    else:
        torch.save({"state_dict": final_weights}, output_file)

    print("Done!")

    return (
        True,
        f"{output_file}<br>base_alpha applied [{count_target_of_basealpha}] times.",
    )


re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12
re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12


def stage_one(
    theta_0: Dict[str, Tensor],
    theta_1: Dict[str, Tensor],
    weights: list[float],
    alpha: float = 0.0,
    skip_position_ids=0,
    convert_to: Optional[Callable] = None,
    verbose=False,
):
    count_target_of_basealpha = 0

    for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
        # not a unet model or key not in origin model
        if "model" not in key or key not in theta_1:
            dprint(f"  key - {key}", verbose)
            continue

        if KEY_POSITION_IDS in key:
            print(key)
            if skip_position_ids == 1:
                print(
                    f"  modelA: skip 'position_ids' : dtype:{theta_0[KEY_POSITION_IDS].dtype}"
                )
                dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                continue
            elif skip_position_ids == 2:
                theta_0[key] = torch.tensor([list(range(77))], dtype=torch.int64)
                print(
                    f"  modelA: reset 'position_ids': dtype:{theta_0[KEY_POSITION_IDS].dtype}"
                )
                dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                continue
            else:
                print(
                    f"  modelA: 'position_ids' key found. do nothing : {skip_position_ids}: dtype:{theta_0[KEY_POSITION_IDS].dtype}"
                )

        dprint(f"  key : {key}", verbose)
        current_alpha = alpha

        # check weighted and U-Net or not
        if weights is not None and "model.diffusion_model." in key:
            # check block index
            weight_index = -1

            if "time_embed" in key:
                weight_index = 0  # before input blocks
            elif ".out." in key:
                weight_index = NUM_TOTAL_BLOCKS - 1  # after output blocks
            else:
                m = re_inp.search(key)
                if m:
                    inp_idx = int(m.groups()[0])
                    weight_index = inp_idx
                else:
                    m = re_mid.search(key)
                    if m:
                        weight_index = NUM_INPUT_BLOCKS
                    else:
                        m = re_out.search(key)
                        if m:
                            out_idx = int(m.groups()[0])
                            weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx

            if weight_index >= NUM_TOTAL_BLOCKS:
                raise RuntimeError(f"error. illegal block index: {key}")

            if weight_index >= 0:
                current_alpha = weights[weight_index]
                dprint(f"weighted '{key}': {current_alpha}", verbose)
        else:
            count_target_of_basealpha = count_target_of_basealpha + 1
            dprint(f"base_alpha applied: [{key}]", verbose)

        theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]

        if convert_to is not None:
            theta_0[key] = convert_to(theta_0[key])

    return theta_0


def stage_two(
    theta_0: Dict[str, Tensor],
    theta_1: Dict[str, Tensor],
    skip_position_ids=0,
    convert_to: Optional[Callable] = None,
    verbose=False,
):
    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if "model" not in key or key in theta_0:
            dprint(f"  key - {key}", verbose)
            continue

        if KEY_POSITION_IDS in key:
            if skip_position_ids == 1:
                print(
                    f"  modelB: skip 'position_ids' : {theta_0[KEY_POSITION_IDS].dtype}"
                )
                dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                continue
            elif skip_position_ids == 2:
                theta_0[key] = torch.tensor([list(range(77))], dtype=torch.int64)
                print(
                    f"  modelB: reset 'position_ids': {theta_0[KEY_POSITION_IDS].dtype}"
                )
                dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                continue
            else:
                print(
                    f"  modelB: 'position_ids' key found. do nothing : {skip_position_ids}"
                )

        dprint(f"  key : {key}", verbose)
        theta_0.update({key: theta_1[key]})

        if convert_to is not None:
            theta_0[key] = convert_to(theta_0[key])

    return theta_0


def conv_fp16(t: Tensor):
    return t.half()


def conv_bf16(t: Tensor):
    return t.bfloat16()


def conv_full(t):
    return t


# _g_precision_func = {
#     "full": conv_full,
#     "fp32": conv_full,
#     "fp16": conv_fp16,
#     "bf16": conv_bf16,
# }
#
#
# def check_weight_type(k: str) -> str:
#     if k.startswith("model.diffusion_model"):
#         return "unet"
#     elif k.startswith("first_stage_model"):
#         return "vae"
#     elif k.startswith("cond_stage_model"):
#         return "clip"
#     return "other"


# unet_conv,
# text_encoder_conv,
# vae_conv,
# others_conv
# specific_part_conv = ["copy", "convert", "delete"]
# extra_opt = {
#     "unet": unet_conv,
#     "clip": text_encoder_conv,
#     "vae": vae_conv,
#     "other": others_conv,
# }


# def converter(t: Tensor, conv_t: str):
#     if not isinstance(t, Tensor):
#         return
#     # w_t = check_weight_type(wk)
#     # conv_t = extra_opt[w_t]
#     conv_t = "convert"
#     if conv_t == "convert":
#         return conv_func(t)
#     elif conv_t == "copy":
#         t
#     elif conv_t == "delete":
#         return
#
#
# def prune(conv_type, precision):
#     conv_func = _g_precision_func[precision]
#     ok = {}
#
#     def _hf(wk: str, t: Tensor):
#         if not isinstance(t, Tensor):
#             return
#         # w_t = check_weight_type(wk)
#         # conv_t = extra_opt[w_t]
#         conv_t = "convert"
#         if conv_t == "convert":
#             ok[wk] = conv_func(t)
#         elif conv_t == "copy":
#             ok[wk] = t
#         elif conv_t == "delete":
#             return
#
#     if conv_type == "ema-only":
#         for k in tqdm(state_dict):
#             ema_k = "___"
#             try:
#                 ema_k = "model_ema." + k[6:].replace(".", "")
#             except:
#                 pass
#             if ema_k in state_dict:
#                 _hf(k, state_dict[ema_k])
#                 # print("ema: " + ema_k + " > " + k)
#             elif not k.startswith("model_ema.") or k in [
#                 "model_ema.num_updates",
#                 "model_ema.decay",
#             ]:
#                 _hf(k, state_dict[k])
#             #     print(k)
#             # else:
#             #     print("skipped: " + k)
#     elif conv_type == "no-ema":
#         for k, v in tqdm(state_dict.items()):
#             if "model_ema" not in k:
#                 _hf(k, v)
#     else:
#         for k, v in tqdm(state_dict.items()):
#             _hf(k, v)
