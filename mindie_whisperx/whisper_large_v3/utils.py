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

from dataclasses import dataclass, field
from typing import List


@dataclass
class CompileInfo:
    prefix_name = ["mindie_whisper_encoder_bs",
                    "mindie_decoder_prefill_bs",
                    "mindie_whisper_decoder_bs",
                    "mindie_self_scatter_bs",
                    "mindie_encoder_scatter_bs"]
    mel_feature_size = 128
    max_frames = 3000
    max_decode_step = 448
    head_num = 20
    head_size = 64
    encoder_seq_len = 1500
    hidden_size = 1280
    layer_nums = 32
    machine_type = ["300IPro", "800IA2"]
    attention_type = {"300IPro": "FA_HIGH_PERF", "800IA2": "PFA"}
    # 2 : decoder_input_ids, encoder_outputs
    param_num_min = 2
    # 131 : ecoder_input_ids, encoder_outputs, actual_seq_len and 128 kvcache tensors.
    param_num_max = 131