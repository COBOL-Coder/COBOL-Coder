# Copyright 2025 the LlamaFactory team.
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

import heapq
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, TypedDict, Union

import torch
import torch.nn.functional as F
from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from ..extras import logging

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset"]]


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    seed: int,
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once(
                "The samples between different datasets will not be mixed in streaming mode."
            )

        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once(
                "We recommend using `mix_strategy=concat` in non-streaming mode."
            )

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy=(
                "first_exhausted"
                if data_args.mix_strategy.endswith("under")
                else "all_exhausted"
            ),
        )
    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.val_size))
        train_set = dataset.skip(int(data_args.val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = (
            int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        )
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})


def preprocess_sp_dataset(seq_ids, world_size, sequence_parallel_mode):
    if sequence_parallel_mode == "zigzag-ring":
        step = len(seq_ids) // (2 * world_size)
        value_chunks = [seq_ids[s : s + step] for s in range(0, len(seq_ids), step)]
        local_values = list()
        for rank in range(world_size):
            local_values.append(
                value_chunks[rank] + value_chunks[2 * world_size - rank - 1]
            )
        return local_values
    else:
        raise NotImplementedError(
            "Other sequence parallel modes are to be implemented."
        )


# HARD CODE FOR QWEN2.5-VL 7B TODO: change to use constants
SPATIAL_MERGE_SIZE = 2
TOKENS_PER_SECOND = 2
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_ID = 151652
VISION_END_ID = 151653
PAD_ID = 151643


def preprocess_sp_qwen25_vl(
    inputs: dict[str, torch.Tensor], seqs: int, cutoff_len: int
) -> list[dict[str, torch.Tensor]]:
    input_ids = inputs["input_ids"][..., :-1]
    position_ids = inputs["position_ids"][..., :-1]
    labels = inputs["labels"][..., 1:]
    attention_mask = inputs["attention_mask"][..., :-1]

    _, vision_ends = torch.where(input_ids == VISION_END_ID)
    _, vision_starts = torch.where(input_ids == VISION_START_ID)
    if len(vision_ends) == 0:
        chunks = inputs
    else:
        image_grid_thw = inputs["image_grid_thw"]
        pixel_values = inputs["pixel_values"]
        chunks = []
        start = 0
        pixel_start = 0
        for i, (vision_start, vision_end) in enumerate(zip(vision_starts, vision_ends)):
            text_chunk = {"type": "text"}
            text_chunk["input_ids"] = input_ids[:, start:vision_start]
            text_chunk["position_ids"] = position_ids[:, :, start:vision_start]
            text_chunk["labels"] = labels[:, start:vision_start]
            text_chunk["attention_mask"] = attention_mask[:, start:vision_start]
            text_chunk["length"] = text_chunk["input_ids"].shape[1]

            vision_chunk = {"type": "vision"}
            vision_chunk["input_ids"] = input_ids[:, vision_start : vision_end + 1]
            vision_chunk["position_ids"] = position_ids[
                :, :, vision_start : vision_end + 1
            ]
            vision_chunk["labels"] = labels[:, vision_start : vision_end + 1]
            vision_chunk["attention_mask"] = attention_mask[
                :, vision_start : vision_end + 1
            ]
            vision_chunk["image_grid_thw"] = image_grid_thw[i, :][None]
            num_pixels = (
                image_grid_thw[i, 0] * image_grid_thw[i, 1] * image_grid_thw[i, 2]
            )
            vision_chunk["pixel_values"] = pixel_values[
                pixel_start : pixel_start + num_pixels, :
            ]
            pixel_start += num_pixels
            vision_chunk["length"] = vision_chunk["input_ids"].shape[1]

            chunks.append(text_chunk)
            chunks.append(vision_chunk)

            start = vision_end + 1

        if start < input_ids.shape[1]:
            text_chunk = {"type": "text"}
            text_chunk["input_ids"] = input_ids[:, start:]
            text_chunk["position_ids"] = position_ids[:, :, start:]
            text_chunk["labels"] = labels[:, start:]
            text_chunk["attention_mask"] = attention_mask[:start:]
            text_chunk["length"] = text_chunk["input_ids"].shape[1]
            chunks.append(text_chunk)

    # do naive grouping for now
    if len(chunks) < seqs:
        buckets_to_fill = seqs - len(chunks) + 1
        longest_text_idx = 0
        longest_length = 0
        for i, chunk in enumerate(chunks):
            if chunk["type"] == "text":
                if chunk["length"] > longest_length:
                    longest_length = chunk["length"]
                    longest_text_idx = i
        new_chunks = [{} for _ in range(buckets_to_fill)]
        for k in chunks[longest_text_idx].keys():
            if k in ["input_ids", "position_ids", "labels", "attention_mask"]:
                splits = chunks[longest_text_idx][k].chunk(buckets_to_fill, dim=-1)
                for i, split in enumerate(splits):
                    new_chunks[i][k] = split
                    new_chunks[i]["length"] = split.shape[-1]
        chunks.pop(longest_text_idx)
        chunks.extend(new_chunks)

    buckets = [[] for _ in range(seqs)]
    chunks = sorted(chunks, key=lambda x: x["length"])
    for i, chunk in enumerate(chunks):
        buckets[i % len(buckets)].append(chunk)

    for i, bucket in enumerate(buckets):
        merged = {}
        for elt in bucket:
            for key in elt:
                if key in [
                    "input_ids",
                    "posision_ids",
                    "labels",
                    "attention_mask",
                    "image_grid_thw",
                    "pixel_values",
                ]:
                    if key in merged:
                        if key in ["image_grid_thw", "pixel_values"]:
                            dim = 0
                        else:
                            dim = -1
                        merged[key] = torch.cat([merged[key], elt[key]], dim=dim)
                    else:
                        merged[key] = elt[key]
        buckets[i] = merged

    max_length_seq = max(
        [elt["input_ids"].shape[-1] for elt in buckets if "input_ids" in elt]
    )
    # pad to maximum seq size, this is so sequences across batches are the same size
    # this cancause problems with gradient accumulation if they are different sizes
    max_length = 2 * cutoff_len // seqs
    assert (
        max_length > max_length_seq
    ), f"Fixed length shorter than maximum seq length: {max_length}, {max_length_seq}"
    for bucket in buckets:
        for key in bucket:
            if key == "attention_mask":
                pad_length = max_length - bucket[key].shape[-1]
                pad_value = 0
                bucket[key] = F.pad(
                    bucket[key], (0, pad_length), mode="constant", value=pad_value
                )
            elif key == "labels":
                pad_length = max_length - bucket[key].shape[-1]
                pad_value = -100
                bucket[key] = F.pad(
                    bucket[key], (0, pad_length), mode="constant", value=pad_value
                )
            elif key == "input_ids":
                pad_length = max_length - bucket[key].shape[-1]
                pad_value = PAD_ID
                bucket[key] = F.pad(
                    bucket[key], (0, pad_length), mode="constant", value=pad_value
                )
            elif key == "position_ids":
                pad_length = max_length - bucket[key].shape[-1]
                pad_value = 1
                bucket[key] = F.pad(
                    bucket[key], (0, pad_length), mode="constant", value=pad_value
                )

    return buckets


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    # spatial_merge_size = self.config.vision_config.spatial_merge_size
    # image_token_id = self.config.image_token_id
    # video_token_id = self.config.video_token_id
    # vision_start_token_id = self.config.vision_start_token_id
    spatial_merge_size = SPATIAL_MERGE_SIZE
    image_token_id = IMAGE_TOKEN_ID
    video_token_id = VIDEO_TOKEN_ID
    vision_start_token_id = VISION_START_ID
    tokens_per_second = TOKENS_PER_SECOND

    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def preprocess_sp_multimodal(
    example, world_size, sequence_parallel_mode, mm_plugin, processor
) -> Dict[str, List]:
    """
    Splits input_ids and all token-level keys into approximately balanced chunks for sequence parallelism
    while keeping vision segments intact. Additionally, if a text block
    is too long and causes imbalance, part of it will be split off and reassigned to a lower-loaded chunk.

    flow:
      1. Identify token-level keys (lists with the same length as input_ids) and extra keys.
      2. Walk the input_ids to create blocks. Each block is either a vision block (atomic) or a text block (splittable),
         and for each block, we slice every token-level key accordingly.
      3. Merge adjacent text blocks.
      4. Do an initial zigzag assignment (taking from the beginning and end of the block list)
         using a min-heap over current chunk totals.
      5. If imbalance is large, find a splittable text block in the heavier chunk, split off a portion (for every token key),
         and assign it to the lighter chunk.
      6. Reconstruct the final chunks. Each chunk will contain all token-level keys (reassembled in order)
         plus the collected "images" (and "videos") from vision segments. Extra keys are attached unchanged.

    """
    if sequence_parallel_mode != "zigzag-ring":
        raise NotImplementedError("Only zigzag-ring mode is supported.")

    # Convert to list for easy manipulation if tensor
    example = {
        k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in example.items()
    }
    input_ids = example["input_ids"]

    images = example.get("images", [])

    # change
    mm_inputs = mm_plugin._get_mm_inputs(images, [], [], processor)
    position_ids, _ = get_rope_index(
        input_ids=torch.tensor(input_ids)[None],
        image_grid_thw=mm_inputs["image_grid_thw"],
        attention_mask=torch.tensor(example["attention_mask"])[None],
    )
    position_ids = position_ids.tolist()
    example["position_ids_1"] = position_ids[0][0]
    example["position_ids_2"] = position_ids[1][0]
    example["position_ids_3"] = position_ids[2][0]

    videos = example.get('videos', [])  # future use

    # Identify token-level keys: lists with same length as input_ids (except images/videos)
    token_keys = [
        k
        for k, v in example.items()
        if isinstance(v, list)
        and len(v) == len(input_ids)
        and k not in ["images", "videos"]
    ]
    extra_keys = {
        k: v for k, v in example.items() if k not in token_keys + ["images", "videos"]
    }

    # === Step 1: Create blocks from token-level keys based on input_ids boundaries ===
    # Each block is a dict with:
    #   'tokens': a dict mapping each token key to its sliced list
    #   'type': either "vision" (from VISION_START_ID to VISION_END_ID, inclusive) or "text"
    #   'orig_idx': a marker for original order (later used to reassemble tokens)
    #   For vision blocks, also store "images": the associated image, since LLamaFactory data module did not process images, it leaves image paths intact before going to data collator
    blocks = []
    i = 0
    block_index = 0
    vision_counter = 0
    while i < len(input_ids):
        if input_ids[i] == VISION_START_ID:
            start = i
            while i < len(input_ids) and input_ids[i] != VISION_END_ID:
                i += 1
            if i < len(input_ids):
                i += 1
            block_tokens = {k: example[k][start:i] for k in token_keys}
            block_length = len(block_tokens["input_ids"])
            block_dict = {
                "tokens": block_tokens,
                "type": "vision",
                "orig_idx": block_index,
                "length": block_length,
                "images": (
                    [images[vision_counter]] if vision_counter < len(images) else []
                ),
            }
            blocks.append(block_dict)
            block_index += 1
            vision_counter += 1
        else:
            start = i
            while i < len(input_ids) and input_ids[i] != VISION_START_ID:
                i += 1
            block_tokens = {k: example[k][start:i] for k in token_keys}
            block_length = len(block_tokens["input_ids"])
            blocks.append(
                {
                    "tokens": block_tokens,
                    "type": "text",
                    "orig_idx": block_index,
                    "length": block_length,
                }
            )
            block_index += 1

    # === Step 2: Merge adjacent text blocks ===
    merged_blocks = []
    for block in blocks:
        if (
            merged_blocks
            and block["type"] == "text"
            and merged_blocks[-1]["type"] == "text"
        ):
            # Extend each token key in the previous block
            for k in token_keys:
                merged_blocks[-1]["tokens"][k].extend(block["tokens"][k])
            merged_blocks[-1]["length"] += block["length"]
        else:
            merged_blocks.append(block)
    blocks = merged_blocks

    # === Step 3: Initial zigzag assignment ===
    # Assign blocks from both ends to balance overall token causal mask (we might want to use attn var len for future videos).
    local_blocks = [[] for _ in range(world_size)]
    totals = [0] * world_size
    heap = [(0, rank) for rank in range(world_size)]
    heapq.heapify(heap)
    left, right = 0, len(blocks) - 1
    while left <= right:
        if left <= right:
            cur_total, rank = heapq.heappop(heap)
            local_blocks[rank].append(blocks[left])
            totals[rank] += blocks[left]["length"]
            heapq.heappush(heap, (totals[rank], rank))
            left += 1
        if left <= right:
            cur_total, rank = heapq.heappop(heap)
            local_blocks[rank].append(blocks[right])
            totals[rank] += blocks[right]["length"]
            heapq.heappush(heap, (totals[rank], rank))
            right -= 1

    # === Step 4: Post-process to rebalance by splitting large text blocks ===
    # If one chunk is much heavier, try to split a text block and move part of it.
    total_tokens = len(input_ids)
    target = total_tokens // world_size
    THRESHOLD = 0
    diff = max(totals) - min(totals)
    while diff > THRESHOLD:
        max_rank = totals.index(max(totals))
        min_rank = totals.index(min(totals))

        candidate_idx = None
        for idx, block in enumerate(local_blocks[max_rank]):
            if block["type"] == "text" and block["length"] > 1:
                candidate_idx = idx
                break
        if candidate_idx is None:
            break
        block = local_blocks[max_rank][candidate_idx]
        move_count = min(block["length"] - 1, diff // 2)
        if move_count <= 0:
            break
        moved_tokens = {}
        for k in token_keys:
            moved_tokens[k] = block["tokens"][k][-move_count:]
            block["tokens"][k] = block["tokens"][k][:-move_count]
        block["length"] = len(block["tokens"]["input_ids"])
        new_block = {
            "tokens": moved_tokens,
            "type": "text",
            "orig_idx": block["orig_idx"] + 0.1,
            "length": move_count,
        }
        local_blocks[min_rank].append(new_block)
        totals[max_rank] -= move_count
        totals[min_rank] += move_count
        diff = max(totals) - min(totals)

    # === Step 5: Reconstruct final chunks ===
    result = []
    for blocks_list in local_blocks:
        sorted_blocks = sorted(blocks_list, key=lambda b: b["orig_idx"])
        chunk_dict = {}
        for k in token_keys:
            chunk_dict[k] = []
        chunk_images = []
        for b in sorted_blocks:
            for k in token_keys:
                chunk_dict[k].extend(b["tokens"][k])
            if b["type"] == "vision" and "images" in b:
                chunk_images.extend(b["images"])
        chunk_dict["images"] = chunk_images
        chunk_dict.update(extra_keys)
        result.append(chunk_dict)

    # === Step 6: Pad token-level keys to equal length across chunks, current zigzag-ring attn needs this, but future attn var len may not need this ===
    # Define pad values per key: for attention_mask use 0, for labels use -100, otherwise 0.
    pad_map = {}
    for k in token_keys:
        if k == "attention_mask":
            pad_map[k] = 0
        elif k == "labels":
            pad_map[k] = -100
        elif k == "input_ids":
            pad_map[k] = PAD_ID
        elif k.startswith("position_id"):
            pad_map[k] = 1

    # Collate adds fake inputs, for example for messages that don't contain images it
    # will add a fake image. This supposedly helps avoid process hanging issues with
    # zero3/fsdp
    # 8192 = 8 * 2**10
    fake_input_buffer = 192
    max_lengths = {k: max(len(chunk[k]) + fake_input_buffer for chunk in result) for k in token_keys}
    # max_lengths = {k: 8192 for k in token_keys}

    for chunk in result:
        for k in token_keys:
            pad_length = max_lengths[k] - len(chunk[k])
            if pad_length > 0:
                chunk[k].extend([pad_map[k]] * pad_length)

    # change
    for elt in result:
        elt["position_ids"] = [
            [elt["position_ids_1"]],
            [elt["position_ids_2"]],
            [elt["position_ids_3"]],
        ]
        del elt["position_ids_1"]
        del elt["position_ids_2"]
        del elt["position_ids_3"]

    # === Step 7: Convert token-level keys to tensors and reassemble final dataset ===
    for i in range(len(result)):
        result[i].update(
            {
                k: torch.tensor(v)
                for k, v in result[i].items()
                if isinstance(v, list) and k not in ["images", "videos", "audios"]
            }
        )

    # change
    del example["position_ids_1"]
    del example["position_ids_2"]
    del example["position_ids_3"]

    dataset = {}
    for key in example.keys():
        dataset[key] = []
    for chunk in result:
        for key in example.keys():
            if key in chunk:
                dataset[key].append(chunk[key])
            else:
                dataset[key].append(None)

    # for i in range(len(dataset["input_ids"])):
    #     print(f"In data preprocessor: {len(dataset['input_ids'][i])}")
    return dataset
