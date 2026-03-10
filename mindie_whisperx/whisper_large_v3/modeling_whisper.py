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
import copy
from typing import Optional, Tuple, Union, List, Dict

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.whisper.modeling_whisper import WhisperDecoder, WhisperEncoder, WhisperEncoderLayer, \
    WhisperModel, WhisperDecoderLayer, WhisperAttention
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.utils import GenerationMixin
import mindietorch
from mindietorch._enums import dtype
from .utils import CompileInfo


MAX_REPEAT_NUM = 25


class MindieFlashAttention(WhisperAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config=None
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            is_decoder,
            bias,
            is_causal,
            config
        )

        if config.hardware not in CompileInfo.machine_type:
            raise ValueError(f"Initialize MindieFlashAttention failed, "
                               f"hardware is required, but got {config.hardware}"
                               f"only suppsort {CompileInfo.machine_type}.")

        self.attention_type = CompileInfo.attention_type[config.hardware]

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        # cross attn
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[1] == key_value_states.shape[1]
        ):  # past_key_value layout is batch_size, sequence_length, num_heads, head_dim

            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states.to(torch.float16), value_states.to(torch.float16))

        # B N S D
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).contiguous().transpose(1, 2)

        attn_output = torch.ops.aie.flash_attention(
            query=query_states,
            key=key_states.transpose(1, 2),
            value=value_states.transpose(1, 2),
            num_head=self.num_heads,
            scale=self.scaling,
            layout="BNSD",
            type=self.attention_type
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.view(bsz, tgt_len, self.num_heads, self.head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value


class MindieIncreFlashAttention(WhisperAttention):

    def __init__(
        self, embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config=None
    ):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias, is_causal, config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        actual_seq_len: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if past_key_value is None:
            raise ValueError("Current operation is incre_flash_attention, past_key_value is required.")

        bsz, tgt_len, _ = hidden_states.size()
        if tgt_len != 1:
            raise ValueError(f"Current operation is incre_flash_attention, query's seq length should be equal to 1."
                             f"but got {tgt_len}.")

        query_states = self.q_proj(hidden_states).view(bsz, tgt_len, self.num_heads, self.head_dim).contiguous()
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz).to(torch.float16)
        values_states = self._shape(self.v_proj(hidden_states), -1, bsz).to(torch.float16)
        past_key_cache, past_value_cache = past_key_value[0], past_key_value[1]

        if actual_seq_len is not None:
            indices = actual_seq_len - 1
            past_key_cache = torch.ops.aie.scatter_update(past_key_cache, indices, key_states, axis=1)
            past_value_cache = torch.ops.aie.scatter_update(past_value_cache, indices, values_states, axis=1)
            past_key_value = (past_key_cache, past_value_cache)
            attn_output = torch.ops.aie.incre_flash_attention(
                query=query_states,
                key=past_key_cache,
                value=past_value_cache,
                actual_seq_lengths=actual_seq_len,
                num_head=self.num_heads,
                scale=self.scaling,
                layout="BSND")
        else:
            attn_output = torch.ops.aie.incre_flash_attention(
                query=query_states,
                key=past_key_cache,
                value=past_value_cache,
                num_head=self.num_heads,
                scale=self.scaling,
                layout="BSND")
        attn_output = attn_output.view(bsz, tgt_len, self.num_heads, self.head_dim)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


class MindieWhisperDecoderLayer(WhisperDecoderLayer):

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.d_model
        if config.is_use_ifa:
            self.self_attn = MindieIncreFlashAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                config=config,
                is_decoder=True
            )
            self.encoder_attn = MindieIncreFlashAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                is_decoder=True,
                config=config)
        else:
            self.self_attn = MindieFlashAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                config=config,
                is_decoder=True
            )
            self.encoder_attn = MindieFlashAttention(
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                is_decoder=True,
                config=config
            )

    def forward(
            self,
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            layer_head_mask,
            cross_attn_layer_head_mask,
            past_key_value,
            cross_attn_past_key_value,
            output_attentions,
            use_cache,
            actual_seq_len,
            encoder_attention_mask=None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attn_past_key_value = past_key_value if past_key_value is not None else None
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            actual_seq_len=actual_seq_len
        )
        hidden_states = residual + hidden_states

        cross_attn_present_key_value = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            hidden_states, _, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                actual_seq_len=None
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states.reshape(-1, 1, self.embed_dim)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MindieWhisperDecoder(WhisperDecoder):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([MindieWhisperDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.config = config

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            past_key_values,
            actual_seq_len,
            use_cache=True,
            attention_mask=None):
        if input_ids is None:
            raise ValueError("You have to specify either decoder_input_ids")
        inputs_embeds = self.embed_tokens(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        past_key_values_length = past_key_values[0].shape[2] if past_key_values is not None else 0
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        inputs_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        hidden_states = inputs_embeds + positions

        past_key_value_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (past_key_values[4 * idx], past_key_values[4 * idx + 1]) \
                if past_key_values is not None else None
            cross_past_key_value = (past_key_values[4 * idx + 2], past_key_values[4 * idx + 3]) \
                if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                actual_seq_len=actual_seq_len,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=past_key_value,
                cross_attn_past_key_value=cross_past_key_value,
                output_attentions=None,
                use_cache=use_cache)

            hidden_states = layer_outputs[0]
            past_key_value_cache.extend(layer_outputs[1])

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, past_key_value_cache


class MindieWhisperEncoderLayer(WhisperEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.self_attn = MindieFlashAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config)


class MindieWhisperEncoder(WhisperEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([MindieWhisperEncoderLayer(config)
            for _ in range(config.encoder_layers)])


class MindieWhisperModel(WhisperModel):

    def __init__(self, config):
        super().__init__(config)
        self.decoder = MindieWhisperDecoder(config)
        self.encoder = MindieWhisperEncoder(config)

    def forward(
            self,
            encoder_outputs,
            decoder_input_ids,
            past_key_values,
            actual_seq_len,
            use_cache: Optional[bool] = False,
            return_dict: Optional[bool] = True,
            input_features: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
    ) -> List[torch.Tensor]:
        if input_features is None and encoder_outputs is None:
            raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `forward`.")

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            actual_seq_len=actual_seq_len
        )
        return decoder_outputs


class MindieWhisperForConditionalGeneration(WhisperForConditionalGeneration, GenerationMixin):

    def __init__(self, config, is_use_ifa=False, hardware="800IA2"):
        super().__init__(config)
        config.is_use_ifa = is_use_ifa
        config.hardware = hardware
        self.model = MindieWhisperModel(config)
        self.has_load = False
        self.mindie_encoder = None
        self.mindie_decoder_prefill = None
        self.mindie_decoder = None
        self.save_path = None
        self.self_attn_scatter = None
        self.encoder_attn_scatter = None
        self.file_prefix_names = CompileInfo.prefix_name
        self.past_key_value = []

    def forward(self, *args):
        if len(args) not in (CompileInfo.param_num_min, CompileInfo.param_num_max):
            raise ValueError(f"The args length of forward can only be {CompileInfo.param_num_min} "
                f"or {CompileInfo.param_num_max}, but got {len(args)}.")
        decoder_input_ids = args[0]
        encoder_outputs = args[1]
        if len(args) == CompileInfo.param_num_max:
            # 2 : param index
            actual_seq_len = args[2]
            # 3 : param index
            past_key_values = args[3:]
        else:
            past_key_values = None
            actual_seq_len = None

        outputs = self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            actual_seq_len=actual_seq_len,
            use_cache=True,
            return_dict=False,
            input_features=None,
        )
        lm_logits = self.proj_out(outputs[0])
        return [lm_logits] + outputs[1]

    def load_mindie_models(self, save_path, batch_size):
        if not (save_path and batch_size):
            raise ValueError(f"Please provide batch_size and the directory where the compiled models saved,\
                             but found save_path is {save_path}, batch_size is{batch_size}.")
        self._check_save_path(save_path, batch_size)

        for _ in range(32):
            self.past_key_value.append(
                torch.ones([batch_size, CompileInfo.max_decode_step, CompileInfo.head_num, CompileInfo.head_size],
                    dtype=torch.float16).to("npu")
            )
            self.past_key_value.append(
                torch.ones([batch_size, CompileInfo.max_decode_step, CompileInfo.head_num, CompileInfo.head_size],
                    dtype=torch.float16).to("npu")
            )
            self.past_key_value.append(
                torch.ones([batch_size, CompileInfo.encoder_seq_len, CompileInfo.head_num, CompileInfo.head_size],
                    dtype=torch.float16).to("npu")
            )
            self.past_key_value.append(
                torch.ones([batch_size, CompileInfo.encoder_seq_len, CompileInfo.head_num, CompileInfo.head_size],
                    dtype=torch.float16).to("npu")
            )
        print("init past key value cache success.")

        if not self.has_load:
            self.mindie_encoder = torch.jit.load(f"{save_path}/{self.file_prefix_names[0]}{batch_size}.ts")
            print(f"load {save_path}/{self.file_prefix_names[0]}{batch_size}.ts success.")

            self.mindie_decoder_prefill = torch.jit.load(f"{save_path}/{self.file_prefix_names[1]}{batch_size}.ts")
            print(f"load {save_path}/{self.file_prefix_names[1]}{batch_size}.ts success.")

            self.mindie_decoder = torch.jit.load(f"{save_path}/{self.file_prefix_names[2]}{batch_size}.ts")
            print(f"load {save_path}/{self.file_prefix_names[2]}{batch_size}.ts success.")

            self.self_attn_scatter = torch.jit.load(f"{save_path}/{self.file_prefix_names[3]}{batch_size}.ts")
            print(f"load {save_path}/{self.file_prefix_names[3]}{batch_size}.ts success.")

            self.encoder_attn_scatter = torch.jit.load(f"{save_path}/{self.file_prefix_names[4]}{batch_size}.ts")
            print(f"load {save_path}/{self.file_prefix_names[4]}{batch_size}.ts success.")
            self.has_load = True
        else:
            print("Mindie whisper has already load.")

    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            **model_kwargs):
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            print(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to("cpu") if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device="cpu")

        this_peer_finished = False  # used by synced_gpus only
        kv_actual_step = 1
        indices = torch.tensor([0] * input_ids.shape[0]).to("npu")
        is_first_step = True
        unfinished_sequences_repeat_num = 0
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, is_first_step, **model_kwargs)
            args = [model_inputs["decoder_input_ids"].contiguous().to("npu"), model_inputs["encoder_outputs"]]
            if is_first_step:
                outputs = self.mindie_decoder_prefill(*args)
                for idx in range(32):
                    self.self_attn_scatter(self.past_key_value[4 * idx], indices, outputs[1 + 4 * idx])
                    self.self_attn_scatter(self.past_key_value[4 * idx + 1], indices, outputs[1 + 4 * idx + 1])
                    self.encoder_attn_scatter(self.past_key_value[4 * idx + 2], indices, outputs[1 + 4 * idx + 2])
                    self.encoder_attn_scatter(self.past_key_value[4 * idx + 3], indices, outputs[1 + 4 * idx + 3])
                is_first_step = False
            else:
                kv_actual_step += 1
                args.append(torch.tensor([kv_actual_step] * input_ids.shape[0]).to("npu"))
                args.extend(self.past_key_value)
                outputs = self.mindie_decoder(*args)
            if isinstance(outputs, list):
                next_token_logits = outputs[0].to("cpu")[:, -1, :]
            else:
                next_token_logits = outputs.to("cpu")[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids.to("cpu"), next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids.to("cpu"), next_tokens[:, None]], dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True
                
                if torch.sum(unfinished_sequences == 1).item() == 1:
                    unfinished_sequences_repeat_num = unfinished_sequences_repeat_num + 1

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores) or unfinished_sequences_repeat_num == MAX_REPEAT_NUM:
                this_peer_finished = True

            if this_peer_finished:
                break

        return input_ids

    def generate(
            self,
            input_features: Optional[torch.Tensor] = None,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            return_dict_in_generate: Optional[bool] = None,
            return_timestamps=None,
            **kwargs,
    ):
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        num_segment_frames = 3000
        if input_features.shape[-1] != num_segment_frames:
            raise ValueError(f"Whisper model only support {num_segment_frames} "
                             f"speech frames, but got {input_features.shape[-1]}")
        encoder_outputs = self.mindie_encoder(input_features.to("npu"))[0]
        kwargs["encoder_outputs"] = encoder_outputs
        outputs = GenerationMixin.generate(
            self,
            input_features,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_dict_in_generate=return_dict_in_generate,
            **kwargs
        )
        return outputs

    def _maybe_initialize_input_ids_for_generation(
            self,
            inputs: Optional[torch.Tensor] = None,
            bos_token_id: Optional[int] = None,
            model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.size()[:-1]
            input_ids = torch.ones(shape, dtype=torch.long, device="cpu") * -100
            return input_ids.to("npu")

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_decoder_input_ids_for_generation(
            self,
            batch_size: int,
            model_input_name: str,
            model_kwargs: Dict[str, torch.Tensor],
            decoder_start_token_id: int = None,
            bos_token_id: int = None,
            device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device="cpu") * decoder_start_token_id

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask.to(self.device)

        return decoder_input_ids.to("npu"), model_kwargs

    def _check_save_path(self, save_path, batch_size):
        file_list = os.listdir(save_path)
        expected_files = [file + f"{batch_size}.ts" for file in self.file_prefix_names]
        for file in expected_files:
            if file not in file_list:
                raise ValueError(f"Expected file name is {file}, but can't be found in path: {save_path}")

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            if_first_step,
            encoder_outputs=None,
            **kwargs
    ):
        if not if_first_step:
            decoder_input_ids_shape = decoder_input_ids.shape
            remove_prefix_length = decoder_input_ids_shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        return {
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids
        }