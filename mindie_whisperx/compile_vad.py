# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import torch
import torch.nn as nn
from pyannote.audio import Model

import mindietorch

SAMPLE_RATE = 16000
WIND_SIZE = 5
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 32
CHANNEL = 1


def trace_vad(model_dir, traced_model_dir):
    vad_model = Model.from_pretrained(model_dir, use_auth_token=None)
    chunks = torch.randn(MAX_BATCH_SIZE, CHANNEL, WIND_SIZE * SAMPLE_RATE)
    torch.jit.save(vad_model.to_torchscript(method="trace", example_inputs=chunks), traced_model_dir)


def compile_vad(traced_model_dir, compiled_model_dir, soc_version):
    traced_model = torch.jit.load(traced_model_dir)
    traced_model.eval()

    min_shape = (MIN_BATCH_SIZE, CHANNEL, WIND_SIZE * SAMPLE_RATE)
    max_shape = (MAX_BATCH_SIZE, CHANNEL, WIND_SIZE * SAMPLE_RATE)
    mie_inputs = []
    mie_inputs.append(mindietorch.Input(min_shape=min_shape, max_shape=max_shape))

    compiled_module = mindietorch.compile(
        traced_model,
        inputs=mie_inputs,
        precision_policy=mindietorch.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        require_full_compilation=False,
        allow_tensor_replace_int=True,
        torch_executed_ops=[],
        soc_version=soc_version,
        optimization_level=0)

    torch.jit.save(compiled_module, compiled_model_dir)
    print(f"save {compiled_model_dir} success.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vad_model_path', type=str, required=True, help="please provide vad model path.")
    parser.add_argument('-soc_version', type=str, required=True, help="please provide soc_version.")
    parser.add_argument('-save_path', type=str, default="compiled_models", help="compiled models save dir.")
    parser.add_argument('-device_id', type=int, default=0)

    args = parser.parse_args()
    device = f"npu:{args.device_id}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"Directory {args.save_path} created.")
    else:
        print(f"Directory {args.save_path} already exists.")
    mindietorch.set_device(args.device_id)

    vad_model_dir = os.path.join(args.vad_model_path, "whisperx-vad-segmentation.bin")
    vad_traced_model_dir = os.path.join(args.save_path, "vad_traced_model.pt")
    vad_compiled_model_dir = os.path.join(args.save_path, "mindie_vad.ts")

    trace_vad(vad_model_dir, vad_traced_model_dir)

    compile_vad(vad_traced_model_dir, vad_compiled_model_dir, args.soc_version)