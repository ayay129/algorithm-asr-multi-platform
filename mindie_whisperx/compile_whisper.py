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
import time
import math
import torch
import mindietorch
from mindietorch._enums import dtype
from whisper_large_v3.modeling_whisper import MindieWhisperForConditionalGeneration as MindieWhisper
from whisper_large_v3.utils import CompileInfo


def compile_encoder(model : MindieWhisper, args, compile_info : CompileInfo):

    class Encoder(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_features):
            return self.model(input_features=input_features, return_dict=False)

    encoder = model.get_encoder()
    input_features = torch.randn([args.bs, compile_info.mel_feature_size, compile_info.max_frames])
    encoder_traced = torch.jit.trace(Encoder(encoder), (input_features))
    input_info = [mindietorch.Input(shape=(args.bs, compile_info.mel_feature_size, compile_info.max_frames))]
    compiled = mindietorch.compile(encoder_traced,
                                   inputs=input_info,
                                   precision_policy=mindietorch.PrecisionPolicy.FP16,
                                   soc_version=args.soc_version,
                                   )
    save_file = os.path.join(args.save_path, f"{compile_info.prefix_name[0]}{args.bs}.ts")
    torch.jit.save(compiled, save_file)
    print(f"Compile encoder success, saved in {save_file}")


def compile_prefill_decoder(model : MindieWhisper, args, compile_info : CompileInfo):
    print("Start compiling prefill_decoder.")

    encoder_outputs = torch.randn([args.bs, compile_info.encoder_seq_len, compile_info.hidden_size])
    decoder_input_ids = torch.randint(1, 4, (args.bs, 1))
    prefill_decoder_traced = torch.jit.trace(model.eval(), (decoder_input_ids, encoder_outputs))

    input_info = [mindietorch.Input(shape=(args.bs, 1), dtype=dtype.INT64),
                  mindietorch.Input(shape=(args.bs, compile_info.encoder_seq_len, compile_info.hidden_size))]

    prefill_decoder_compiled = mindietorch.compile(prefill_decoder_traced,
                                                    inputs=input_info,
                                                   precision_policy=mindietorch.PrecisionPolicy.FP16,
                                                   soc_version=args.soc_version)

    save_file = os.path.join(args.save_path, f"{compile_info.prefix_name[1]}{args.bs}.ts")
    torch.jit.save(prefill_decoder_compiled, save_file)
    print(f"Compile prefill_decoder success, saved in {save_file}.")


def compile_incre_decoder(args, compile_info : CompileInfo):
    class Decoder(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            return self.model.forward(*args)[0]

    mindie_whisper_instance = \
        MindieWhisper.from_pretrained(args.model_path, is_use_ifa=True, hardware=args.hardware).to("cpu")
    decoder = Decoder(mindie_whisper_instance)
    print("Start compiling decoder.")

    encoder_outputs = torch.randn([args.bs, compile_info.encoder_seq_len, compile_info.hidden_size])
    decoder_input_ids = torch.randint(1, 4, (args.bs, 1))
    actual_seq_len = torch.ones((args.bs))
    all_past_key_value = [
                        torch.randn(
                            [args.bs, compile_info.max_decode_step, compile_info.head_num, compile_info.head_size]
                            ),
                        torch.randn(
                            [args.bs, compile_info.max_decode_step, compile_info.head_num, compile_info.head_size]
                        ),
                        torch.randn(
                            [args.bs, compile_info.encoder_seq_len, compile_info.head_num, compile_info.head_size]
                        ),
                        torch.randn(
                            [args.bs, compile_info.encoder_seq_len, compile_info.head_num, compile_info.head_size]
                          )] * compile_info.layer_nums

    traced_args = [decoder_input_ids, encoder_outputs, actual_seq_len]
    traced_args.extend(all_past_key_value)
    traced_decoder = torch.jit.trace(decoder, traced_args)
    # BSND
    key_value_infos = [
          mindietorch.Input(shape=(args.bs, compile_info.max_decode_step, compile_info.head_num, compile_info.head_size),
                            dtype=dtype.FLOAT16),
          mindietorch.Input(shape=(args.bs, compile_info.max_decode_step, compile_info.head_num, compile_info.head_size),
                            dtype=dtype.FLOAT16),
          mindietorch.Input(shape=(args.bs, compile_info.encoder_seq_len, compile_info.head_num, compile_info.head_size),
                            dtype=dtype.FLOAT16),
          mindietorch.Input(shape=(args.bs, compile_info.encoder_seq_len, compile_info.head_num, compile_info.head_size),
                            dtype=dtype.FLOAT16
          )] * compile_info.layer_nums
    input_info = [mindietorch.Input(shape=(args.bs, 1), dtype=dtype.INT64), # input ids
                  mindietorch.Input(shape=(args.bs, compile_info.encoder_seq_len, compile_info.hidden_size)),
                  mindietorch.Input(shape=(args.bs,), dtype=dtype.INT64)] # actual sq len

    input_info.extend(key_value_infos)
    float_size = 4
    voc_size = 51866
    buffer_size = math.ceil((args.bs * 1 * voc_size * float_size) / 1024 / 1024)
    print(f"Set {buffer_size}/MB for output.")
    compiled_decoder = mindietorch.compile(traced_decoder,
                                           inputs=input_info,
                                           precision_policy=mindietorch.PrecisionPolicy.FP16,
                                           soc_version=args.soc_version,
                                           default_buffer_size_vec=[buffer_size])
    save_file = os.path.join(args.save_path, f"{compile_info.prefix_name[2]}{args.bs}.ts")
    torch.jit.save(compiled_decoder, save_file)

    print(f"Compile whisper_decoder success, saved in {save_file}.")


def compile_scatter_update(args, compile_info):
    class MindieScatter(torch.nn.Module):
        def forward(self, past_key_value, indices, update_states):
            out = torch.ops.aie.scatter_update(past_key_value, indices, update_states, 1)
            return out

    bs = args.bs
    self_past_key_value = torch.randn([bs, compile_info.max_decode_step, compile_info.head_num, compile_info.head_size])
    encoder_past_key_value = torch.randn([bs, compile_info.encoder_seq_len, compile_info.head_num, compile_info.head_size])
    indices = torch.tensor([0] * bs)
    update_states = torch.randn([bs, 1, 20, 64])
    traced = torch.jit.trace(MindieScatter(), (self_past_key_value, indices, update_states))

    self_attn_info = mindietorch.Input(shape=self_past_key_value.shape, dtype=mindietorch.dtype.FLOAT16)
    encoder_attn_info = mindietorch.Input(shape=encoder_past_key_value.shape, dtype=mindietorch.dtype.FLOAT16)
    indices_info = mindietorch.Input(shape=indices.shape, dtype=mindietorch.dtype.INT64)
    update_states_info = mindietorch.Input(shape=update_states.shape, dtype=mindietorch.dtype.FLOAT16)

    compile_self = mindietorch.compile(traced, inputs=[self_attn_info, indices_info, update_states_info],
                                       soc_version=args.soc_version)
    torch.jit.save(compile_self, f"{args.save_path}/{compile_info.prefix_name[3]}{args.bs}.ts")

    compile_self = mindietorch.compile(traced, inputs=[encoder_attn_info, indices_info, encoder_attn_info],
                                       soc_version=args.soc_version)
    torch.jit.save(compile_self, f"{args.save_path}/{compile_info.prefix_name[4]}{args.bs}.ts")
    print("compile scatter success.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, required=True, help="please provide model path.")
    parser.add_argument('-bs', type=int, default=8, help="please provide batch_size, default:16.")
    parser.add_argument('-soc_version', type=str, required=True, help="please provide soc_version.")
    parser.add_argument('-save_path', type=str, default="compiled_models", help="compiled models save dir.")
    parser.add_argument('-hardware', type=str, choices=["300IPro", "800IA2"], default="800IA2")
    parser.add_argument('-device_id', type=int, default=0)
    compile_args = parser.parse_args()


    mindie_whisper = \
        MindieWhisper.from_pretrained(compile_args.model_path, hardware=compile_args.hardware).to("cpu")

    print("Start compiling Mindie-Whisper, it will take some time, please wait.")
    if not compile_args.save_path:
        raise ValueError("Please provide the directory where the compiled model saved.")
    if not os.path.exists(compile_args.save_path):
        os.makedirs(compile_args.save_path)
        print(f"Directory {compile_args.save_path} created.")
    else:
        print(f"Directory {compile_args.save_path} already exists.")
    mindietorch.set_device(compile_args.device_id)
    compile_scatter_update(compile_args, CompileInfo)
    compile_encoder(mindie_whisper, compile_args, CompileInfo)
    compile_prefill_decoder(mindie_whisper, compile_args, CompileInfo)
    compile_incre_decoder(compile_args, CompileInfo)