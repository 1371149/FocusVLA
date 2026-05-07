import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heads import HEADS_TO_FINETUNE, infer_libero_scene, set_heads_to_finetune, summarize_heads
from local_components import (
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    PROPRIO_DIM,
    ActionTokenizer,
    ProprioProjector,
    PurePromptBuilder,
    TeacherActionCollator,
    TeacherExportBatchTransform as BaseTeacherExportBatchTransform,
    get_current_action_mask,
    get_next_actions_mask,
)
from modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel

transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

from prismatic.vla.datasets import RLDSDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ExportConfig:
    vla_path: str = "<MODEL_ROOT>/openvla-7b-oft-finetuned-libero-spatial"
    data_root_dir: str = "<OPENVLA_FINETUNE_ROOT>/dataset"
    dataset_name: str = "libero_spatial_no_noops"
    output_path: str = "<OPENVLA_OFT_ROOT>/my_finetune/exports/libero_spatial_teacher_preview_16.pt"
    output_dir: Optional[str] = None
    max_samples: int = 10
    export_batch_size: int = 2
    records_per_shard: int = 256
    start_index: int = 0
    end_index: Optional[int] = None
    num_shards: int = 1
    shard_rank: int = 0
    shuffle_buffer_size: int = 1
    image_aug: bool = True
    use_proprio: bool = True
    num_images_in_input: int = 2
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"
    proprio_projector_path: Optional[str] = None
    include_raw_images: bool = True
    average_selected_heads_for_storage: bool = False


@dataclass
class TeacherRecord:
    dataset_name: str
    language_instruction: str
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray
    actions: np.ndarray
    pixel_values: Any
    primary_image_uint8: Optional[np.ndarray]
    wrist_image_uint8: Optional[np.ndarray]
    proprio: Optional[np.ndarray]
    timestep: int
    episode_file_path: str
    teacher_text_mean_logits: Any
    teacher_action_mean_logits: Any
    text_query_mask: np.ndarray
    action_query_mask: np.ndarray
    query_token_ids: np.ndarray


class TeacherExportBatchTransform(BaseTeacherExportBatchTransform):
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = rlds_batch["dataset_name"]
        current_action = rlds_batch["action"][0]
        image = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        language_instruction = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        prompt_builder = self.prompt_builder_fn("openvla")
        future_actions = rlds_batch["action"][1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {language_instruction}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        pixel_values = self.image_transform(image)

        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        sample = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": dataset_name,
            "actions": actions,
        }
        if self.use_wrist_image:
            wrist_pixels = []
            for key in rlds_batch["observation"].keys():
                if "wrist" in key:
                    wrist_image = Image.fromarray(rlds_batch["observation"][key][0])
                    wrist_pixels.append(self.image_transform(wrist_image))
            if wrist_pixels:
                sample["pixel_values_wrist"] = torch.cat(wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            sample["proprio"] = rlds_batch["observation"]["proprio"]

        sample["language_instruction"] = rlds_batch["task"]["language_instruction"].decode().lower()

        if self.include_raw_images:
            sample["primary_image_uint8"] = np.copy(rlds_batch["observation"]["image_primary"][0])

            wrist_keys = [k for k in rlds_batch["observation"].keys() if "wrist" in k]
            if wrist_keys:
                sample["wrist_image_uint8"] = np.copy(rlds_batch["observation"][wrist_keys[0]][0])

        timestep = -1
        if "timestep" in rlds_batch["observation"]:
            raw_timestep = rlds_batch["observation"]["timestep"]
            timestep = int(raw_timestep[0] if np.ndim(raw_timestep) > 0 else raw_timestep)
        sample["timestep"] = timestep

        episode_file_path = ""
        episode_metadata = rlds_batch.get("episode_metadata")
        if isinstance(episode_metadata, dict) and "file_path" in episode_metadata:
            raw_file_path = episode_metadata["file_path"]
            if isinstance(raw_file_path, bytes):
                episode_file_path = raw_file_path.decode()
            else:
                episode_file_path = str(raw_file_path)
        sample["episode_file_path"] = episode_file_path
        return sample


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export teacher attention labels with image/action data into one archive file.")
    parser.add_argument("--vla-path", default=ExportConfig.vla_path)
    parser.add_argument("--data-root-dir", default=ExportConfig.data_root_dir)
    parser.add_argument("--dataset-name", default=ExportConfig.dataset_name)
    parser.add_argument("--output-path", default=ExportConfig.output_path)
    parser.add_argument("--output-dir", default=ExportConfig.output_dir)
    parser.add_argument("--max-samples", type=int, default=ExportConfig.max_samples)
    parser.add_argument("--export-batch-size", type=int, default=ExportConfig.export_batch_size)
    parser.add_argument("--records-per-shard", type=int, default=ExportConfig.records_per_shard)
    parser.add_argument("--start-index", type=int, default=ExportConfig.start_index)
    parser.add_argument("--end-index", type=int, default=ExportConfig.end_index)
    parser.add_argument("--num-shards", type=int, default=ExportConfig.num_shards)
    parser.add_argument("--shard-rank", type=int, default=ExportConfig.shard_rank)
    parser.add_argument("--shuffle-buffer-size", type=int, default=ExportConfig.shuffle_buffer_size)
    parser.add_argument("--image-aug", action="store_true", default=ExportConfig.image_aug)
    parser.add_argument("--use-proprio", action="store_true", default=ExportConfig.use_proprio)
    parser.add_argument("--no-proprio", dest="use_proprio", action="store_false")
    parser.add_argument("--num-images-in-input", type=int, default=ExportConfig.num_images_in_input)
    parser.add_argument("--device", default=ExportConfig.device)
    parser.add_argument("--proprio-projector-path", default=ExportConfig.proprio_projector_path)
    parser.add_argument("--include-raw-images", action="store_true", default=ExportConfig.include_raw_images)
    parser.add_argument("--no-raw-images", dest="include_raw_images", action="store_false")
    parser.add_argument(
        "--average-selected-heads-for-storage",
        action="store_true",
        default=ExportConfig.average_selected_heads_for_storage,
    )
    args = parser.parse_args()
    return ExportConfig(
        vla_path=args.vla_path,
        data_root_dir=args.data_root_dir,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        export_batch_size=args.export_batch_size,
        records_per_shard=args.records_per_shard,
        start_index=args.start_index,
        end_index=args.end_index,
        num_shards=args.num_shards,
        shard_rank=args.shard_rank,
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=args.image_aug,
        use_proprio=args.use_proprio,
        num_images_in_input=args.num_images_in_input,
        device=args.device,
        proprio_projector_path=args.proprio_projector_path,
        include_raw_images=args.include_raw_images,
        average_selected_heads_for_storage=args.average_selected_heads_for_storage,
    )


def model_is_on_hf_hub(model_path: str) -> bool:
    from huggingface_hub import HfApi

    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def resolve_model_path(vla_path: str) -> str:
    if model_is_on_hf_hub(vla_path):
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=vla_path)
    return vla_path


def load_model_and_processor(cfg: ExportConfig):
    resolved_vla_path = resolve_model_path(cfg.vla_path.rstrip("/"))
    processor = AutoProcessor.from_pretrained(resolved_vla_path, trust_remote_code=cfg.trust_remote_code)

    torch_dtype = torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float16
    model = AutoModelForVision2Seq.from_pretrained(
        resolved_vla_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = model.to(cfg.device)
    model.eval()
    model.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    model.config.use_cache = False
    return model, processor


def auto_find_proprio_projector(vla_path: str) -> Optional[str]:
    model_dir = Path(vla_path)
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    candidates = sorted(model_dir.glob("proprio_projector--*_checkpoint.pt"))
    if not candidates:
        latest_candidate = model_dir / "proprio_projector--latest_checkpoint.pt"
        if latest_candidate.exists():
            return str(latest_candidate)
        return None

    def _step_key(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.split("--")[-1].split("_")[0])
        except ValueError:
            return -1

    return str(max(candidates, key=_step_key))


def remove_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_proprio_projector_if_needed(cfg: ExportConfig, llm_dim: int):
    if not cfg.use_proprio:
        return None

    projector_path = cfg.proprio_projector_path or auto_find_proprio_projector(cfg.vla_path)
    if not projector_path:
        print("Warning: `use_proprio=True` but no proprio projector checkpoint was found. Export will run without proprio conditioning.")
        return None

    projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
    print(f"Loading proprio projector from: {projector_path}")
    state_dict = torch.load(projector_path, map_location="cpu", weights_only=True)
    state_dict = remove_ddp_prefix(state_dict)
    projector.load_state_dict(state_dict)
    projector = projector.to(cfg.device)
    projector.eval()
    return projector


def build_dataset(cfg: ExportConfig, processor, model):
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = TeacherExportBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        include_raw_images=cfg.include_raw_images,
    )
    return RLDSDataset(
        Path(cfg.data_root_dir),
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )


def build_export_pixel_values(sample: Dict[str, Any]) -> torch.Tensor:
    pixel_values = sample["pixel_values"]
    if "pixel_values_wrist" in sample:
        pixel_values = torch.cat((pixel_values, sample["pixel_values_wrist"]), dim=0)

    ##################################
    # Match training-time clean inputs: keep exported clean images in bf16 so the
    # stored supervision input matches the training forward precision.
    return pixel_values.to(torch.bfloat16)
    ##################################


def build_single_item_batch(sample: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    pixel_values = build_export_pixel_values(sample)

    batch = {
        "input_ids": sample["input_ids"].unsqueeze(0).to(device),
        "attention_mask": torch.ones_like(sample["input_ids"]).unsqueeze(0).to(device),
        "labels": sample["labels"].unsqueeze(0).to(device),
        "pixel_values": pixel_values.unsqueeze(0).to(device),
    }

    if "proprio" in sample:
        proprio_np = np.array(sample["proprio"], dtype=np.float32, copy=True)
        proprio = torch.from_numpy(proprio_np).reshape(1, -1).to(device)
        batch["proprio"] = proprio
    return batch


def align_query_logits_and_masks(query_logits: torch.Tensor, labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
    target_token_ids = labels[:, 1:]
    if attention_mask is None:
        valid_query_mask = torch.ones_like(target_token_ids, dtype=torch.bool)
    else:
        valid_query_mask = attention_mask[:, 1:].bool()

    action_mask = (get_current_action_mask(target_token_ids) | get_next_actions_mask(target_token_ids)) & valid_query_mask
    text_mask = target_token_ids.eq(IGNORE_INDEX) & valid_query_mask

    if query_logits.shape[2] == target_token_ids.shape[1] + 1:
        query_logits = query_logits[:, :, :-1, :]
    elif query_logits.shape[2] != target_token_ids.shape[1]:
        raise ValueError(
            f"Unexpected query length mismatch: logits query len = {query_logits.shape[2]}, label len = {target_token_ids.shape[1]}"
        )

    if query_logits.shape[2] != action_mask.shape[1]:
        raise ValueError(
            f"Mask/logit mismatch after alignment: query len = {query_logits.shape[2]}, mask len = {action_mask.shape[1]}"
        )

    return query_logits, text_mask, action_mask, target_token_ids


def masked_query_mean(query_logits: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    weights = query_mask[:, None, :, None].to(query_logits.dtype)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (query_logits * weights).sum(dim=2) / denom


##################################
def to_serialized_bf16_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def to_hdf5_compatible_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
        array = tensor.numpy()
    else:
        array = np.array(value, copy=True)

    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array
##################################


##################################
def compute_teacher_means(
    model,
    proprio_projector,
    batch: Dict[str, torch.Tensor],
    use_proprio: bool,
    num_images_in_input: int,
    average_selected_heads_for_storage: bool = False,
) -> Dict[str, Any]:
    num_image_tokens = model.vision_backbone.get_num_patches() * num_images_in_input
    vision_start_idx = 0
    vision_end_idx = vision_start_idx + num_image_tokens
    query_start_idx = vision_end_idx + (1 if use_proprio else 0)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(torch.bfloat16),
                labels=batch["labels"],
                output_hidden_states=False,
                output_attentions=True,
                proprio=batch.get("proprio") if (use_proprio and proprio_projector is not None) else None,
                proprio_projector=proprio_projector if (use_proprio and proprio_projector is not None) else None,
            )

    attentions = output.attentions
    per_layer_text = []
    per_layer_action = []
    text_query_mask_np = None
    action_query_mask_np = None
    query_token_ids_np = None

    for layer_idx, layer_attn in enumerate(attentions):
        selected_heads = HEADS_TO_FINETUNE[layer_idx]
        selected_logits = layer_attn[:, selected_heads, query_start_idx:, vision_start_idx:vision_end_idx].float()
        selected_logits, text_mask, action_mask, query_token_ids = align_query_logits_and_masks(
            selected_logits, batch["labels"], batch.get("attention_mask")
        )

        if text_query_mask_np is None:
            text_query_mask_np = text_mask.to(torch.uint8).cpu().numpy()
            action_query_mask_np = action_mask.to(torch.uint8).cpu().numpy()
            query_token_ids_np = query_token_ids.to(torch.int32).cpu().numpy()

        text_mean = masked_query_mean(selected_logits, text_mask)
        action_mean = masked_query_mean(selected_logits, action_mask)
        if average_selected_heads_for_storage:
            text_mean = text_mean.mean(dim=1, keepdim=True)
            action_mean = action_mean.mean(dim=1, keepdim=True)
        per_layer_text.append(to_serialized_bf16_cpu(text_mean))
        per_layer_action.append(to_serialized_bf16_cpu(action_mean))

    return {
        "teacher_text_mean_logits": torch.stack(per_layer_text, dim=1),
        "teacher_action_mean_logits": torch.stack(per_layer_action, dim=1),
        "text_query_mask": text_query_mask_np,
        "action_query_mask": action_query_mask_np,
        "query_token_ids": query_token_ids_np,
        "vision_token_count": np.int32(num_image_tokens),
        "tokens_per_image": np.int32(num_image_tokens // max(1, num_images_in_input)),
    }
##################################


##################################
def sample_to_record(
    sample: Dict[str, Any],
    teacher_data: Dict[str, Any],
    sample_idx: int = 0,
    pixel_values_override: Optional[torch.Tensor] = None,
) -> TeacherRecord:
    dataset_name = sample["dataset_name"]
    if isinstance(dataset_name, bytes):
        dataset_name = dataset_name.decode()

    primary_image = sample.get("primary_image_uint8")
    wrist_image = sample.get("wrist_image_uint8")
    proprio = sample.get("proprio")

    pixel_values = pixel_values_override if pixel_values_override is not None else build_export_pixel_values(sample)

    return TeacherRecord(
        dataset_name=str(dataset_name),
        language_instruction=sample["language_instruction"],
        input_ids=sample["input_ids"].cpu().numpy().astype(np.int32),
        labels=sample["labels"].cpu().numpy().astype(np.int32),
        attention_mask=np.ones_like(sample["input_ids"].cpu().numpy(), dtype=np.uint8),
        actions=np.asarray(sample["actions"], dtype=np.float32),
        ##################################
        pixel_values=to_serialized_bf16_cpu(pixel_values),
        ##################################
        primary_image_uint8=None if primary_image is None else np.asarray(primary_image, dtype=np.uint8),
        wrist_image_uint8=None if wrist_image is None else np.asarray(wrist_image, dtype=np.uint8),
        proprio=None if proprio is None else np.asarray(proprio, dtype=np.float32).reshape(-1),
        timestep=int(sample.get("timestep", -1)),
        episode_file_path=str(sample.get("episode_file_path", "")),
        ##################################
        teacher_text_mean_logits=to_serialized_bf16_cpu(teacher_data["teacher_text_mean_logits"][sample_idx]),
        teacher_action_mean_logits=to_serialized_bf16_cpu(teacher_data["teacher_action_mean_logits"][sample_idx]),
        ##################################
        text_query_mask=teacher_data["text_query_mask"][sample_idx][: max(0, sample["input_ids"].shape[0] - 1)].astype(np.uint8),
        action_query_mask=teacher_data["action_query_mask"][sample_idx][: max(0, sample["input_ids"].shape[0] - 1)].astype(np.uint8),
        query_token_ids=teacher_data["query_token_ids"][sample_idx][: max(0, sample["input_ids"].shape[0] - 1)].astype(np.int32),
    )
##################################


def _ensure_cpu_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().clone()
    else:
        tensor = torch.from_numpy(np.array(value, copy=True))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def build_packed_pt_payload(records: List[TeacherRecord], cfg: ExportConfig, model) -> dict:
    num_records = len(records)
    seq_lengths = torch.tensor([record.input_ids.shape[0] for record in records], dtype=torch.int32)
    query_lengths = torch.tensor([record.text_query_mask.shape[0] for record in records], dtype=torch.int32)
    max_seq_len = int(seq_lengths.max().item())
    max_query_len = int(query_lengths.max().item())

    input_ids = torch.zeros((num_records, max_seq_len), dtype=torch.int32)
    labels = torch.full((num_records, max_seq_len), fill_value=IGNORE_INDEX, dtype=torch.int32)
    text_query_mask = torch.zeros((num_records, max_query_len), dtype=torch.bool)
    action_query_mask = torch.zeros((num_records, max_query_len), dtype=torch.bool)

    for idx, record in enumerate(records):
        seq_len = record.input_ids.shape[0]
        query_len = record.text_query_mask.shape[0]
        input_ids[idx, :seq_len] = _ensure_cpu_tensor(record.input_ids, dtype=torch.int32)
        labels[idx, :seq_len] = _ensure_cpu_tensor(record.labels, dtype=torch.int32)
        text_query_mask[idx, :query_len] = _ensure_cpu_tensor(record.text_query_mask, dtype=torch.bool)
        action_query_mask[idx, :query_len] = _ensure_cpu_tensor(record.action_query_mask, dtype=torch.bool)

    actions = torch.stack([_ensure_cpu_tensor(record.actions, dtype=torch.float32) for record in records], dim=0)
    pixel_values = torch.stack([_ensure_cpu_tensor(record.pixel_values, dtype=torch.bfloat16) for record in records], dim=0)
    teacher_text_mean_logits = torch.stack(
        [_ensure_cpu_tensor(record.teacher_text_mean_logits, dtype=torch.bfloat16) for record in records], dim=0
    )
    teacher_action_mean_logits = torch.stack(
        [_ensure_cpu_tensor(record.teacher_action_mean_logits, dtype=torch.bfloat16) for record in records], dim=0
    )

    data = {
        "input_ids": input_ids,
        "labels": labels,
        "sequence_length": seq_lengths,
        "pixel_values": pixel_values,
        "actions": actions,
        "teacher_text_mean_logits": teacher_text_mean_logits,
        "teacher_action_mean_logits": teacher_action_mean_logits,
        "text_query_mask": text_query_mask,
        "action_query_mask": action_query_mask,
        "query_length": query_lengths,
    }

    if any(record.proprio is not None for record in records):
        proprio_dim = next(np.asarray(record.proprio, dtype=np.float32).reshape(-1).shape[0] for record in records if record.proprio is not None)
        proprio = torch.zeros((num_records, proprio_dim), dtype=torch.float32)
        proprio_present = torch.zeros((num_records,), dtype=torch.bool)
        for idx, record in enumerate(records):
            if record.proprio is None:
                continue
            proprio[idx] = _ensure_cpu_tensor(record.proprio, dtype=torch.float32).reshape(-1)
            proprio_present[idx] = True
        data["proprio"] = proprio
        data["proprio_present"] = proprio_present

    return {
        "config": asdict(cfg),
        "meta": {
            "num_records": num_records,
            "num_layers": records[0].teacher_text_mean_logits.shape[0],
            "num_selected_heads": records[0].teacher_text_mean_logits.shape[1],
            "vision_token_count": int(model.vision_backbone.get_num_patches() * cfg.num_images_in_input),
            "tokens_per_image": int(model.vision_backbone.get_num_patches()),
            "num_images_in_input": cfg.num_images_in_input,
            "action_token_begin_idx": ACTION_TOKEN_BEGIN_IDX,
            "pt_storage_format": "packed_pt_v2",
        },
        "selected_heads": [HEADS_TO_FINETUNE[layer_idx] for layer_idx in sorted(HEADS_TO_FINETUNE.keys())],
        "data": data,
    }


def write_hdf5(records: List[TeacherRecord], output_path: Path, cfg: ExportConfig, model) -> None:
    import h5py

    if not records:
        raise ValueError("No records were exported.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_records = len(records)
    max_seq_len = max(record.input_ids.shape[0] for record in records)
    max_query_len = max(record.text_query_mask.shape[0] for record in records)
    pixel_shape = records[0].pixel_values.shape
    action_shape = records[0].actions.shape
    teacher_shape = records[0].teacher_text_mean_logits.shape

    has_raw_primary = any(record.primary_image_uint8 is not None for record in records)
    has_raw_wrist = any(record.wrist_image_uint8 is not None for record in records)
    has_proprio = any(record.proprio is not None for record in records)

    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["num_records"] = num_records
        h5_file.attrs["num_layers"] = teacher_shape[0]
        h5_file.attrs["num_selected_heads"] = teacher_shape[1]
        h5_file.attrs["vision_token_count"] = int(model.vision_backbone.get_num_patches() * cfg.num_images_in_input)
        h5_file.attrs["tokens_per_image"] = int(model.vision_backbone.get_num_patches())
        h5_file.attrs["num_images_in_input"] = cfg.num_images_in_input
        h5_file.attrs["action_token_begin_idx"] = ACTION_TOKEN_BEGIN_IDX

        ##################################
        # HDF5 does not store bf16 natively; keep PT exports as bf16 and fall back to
        # fp16 only for HDF5 compatibility.
        h5_file.create_dataset("pixel_values", shape=(num_records, *pixel_shape), dtype=np.float16, compression="gzip")
        h5_file.create_dataset("actions", shape=(num_records, *action_shape), dtype=np.float32, compression="gzip")
        h5_file.create_dataset("teacher_text_mean_logits", shape=(num_records, *teacher_shape), dtype=np.float16, compression="gzip")
        h5_file.create_dataset("teacher_action_mean_logits", shape=(num_records, *teacher_shape), dtype=np.float16, compression="gzip")
        ##################################

        h5_file.create_dataset("input_ids", shape=(num_records, max_seq_len), dtype=np.int32, compression="gzip")
        h5_file.create_dataset("labels", shape=(num_records, max_seq_len), dtype=np.int32, compression="gzip")
        h5_file.create_dataset("attention_mask", shape=(num_records, max_seq_len), dtype=np.uint8, compression="gzip")
        h5_file.create_dataset("sequence_length", shape=(num_records,), dtype=np.int32)

        h5_file.create_dataset("text_query_mask", shape=(num_records, max_query_len), dtype=np.uint8, compression="gzip")
        h5_file.create_dataset("action_query_mask", shape=(num_records, max_query_len), dtype=np.uint8, compression="gzip")
        h5_file.create_dataset("query_token_ids", shape=(num_records, max_query_len), dtype=np.int32, compression="gzip")
        h5_file.create_dataset("query_length", shape=(num_records,), dtype=np.int32)

        if has_proprio:
            proprio_dim = next(record.proprio.shape[0] for record in records if record.proprio is not None)
            h5_file.create_dataset("proprio", shape=(num_records, proprio_dim), dtype=np.float32, compression="gzip")
        if has_raw_primary:
            primary_shape = next(record.primary_image_uint8.shape for record in records if record.primary_image_uint8 is not None)
            h5_file.create_dataset("primary_image_uint8", shape=(num_records, *primary_shape), dtype=np.uint8, compression="gzip")
        if has_raw_wrist:
            wrist_shape = next(record.wrist_image_uint8.shape for record in records if record.wrist_image_uint8 is not None)
            h5_file.create_dataset("wrist_image_uint8", shape=(num_records, *wrist_shape), dtype=np.uint8, compression="gzip")

        h5_file.create_dataset("dataset_name", data=np.asarray([record.dataset_name for record in records], dtype=object), dtype=string_dtype)
        h5_file.create_dataset("language_instruction", data=np.asarray([record.language_instruction for record in records], dtype=object), dtype=string_dtype)
        h5_file.create_dataset("episode_file_path", data=np.asarray([record.episode_file_path for record in records], dtype=object), dtype=string_dtype)
        h5_file.create_dataset("timestep", data=np.asarray([record.timestep for record in records], dtype=np.int32))
        h5_file.create_dataset(
            "selected_heads",
            data=np.asarray([HEADS_TO_FINETUNE[layer_idx] for layer_idx in sorted(HEADS_TO_FINETUNE.keys())], dtype=np.int32),
        )

        for idx, record in enumerate(records):
            seq_len = record.input_ids.shape[0]
            query_len = record.text_query_mask.shape[0]

            h5_file["pixel_values"][idx] = to_hdf5_compatible_numpy(record.pixel_values, dtype=np.float16)
            h5_file["actions"][idx] = record.actions
            h5_file["teacher_text_mean_logits"][idx] = to_hdf5_compatible_numpy(record.teacher_text_mean_logits, dtype=np.float16)
            h5_file["teacher_action_mean_logits"][idx] = to_hdf5_compatible_numpy(record.teacher_action_mean_logits, dtype=np.float16)

            h5_file["input_ids"][idx, :seq_len] = record.input_ids
            h5_file["labels"][idx, :seq_len] = record.labels
            h5_file["labels"][idx, seq_len:] = IGNORE_INDEX
            h5_file["attention_mask"][idx, :seq_len] = record.attention_mask
            h5_file["sequence_length"][idx] = seq_len

            h5_file["text_query_mask"][idx, :query_len] = record.text_query_mask
            h5_file["action_query_mask"][idx, :query_len] = record.action_query_mask
            h5_file["query_token_ids"][idx, :query_len] = record.query_token_ids
            h5_file["query_length"][idx] = query_len

            if has_proprio:
                if record.proprio is not None:
                    h5_file["proprio"][idx] = record.proprio
                else:
                    h5_file["proprio"][idx] = 0
            if has_raw_primary and record.primary_image_uint8 is not None:
                h5_file["primary_image_uint8"][idx] = record.primary_image_uint8
            if has_raw_wrist and record.wrist_image_uint8 is not None:
                h5_file["wrist_image_uint8"][idx] = record.wrist_image_uint8

    manifest_path = output_path.with_suffix(".json")
    manifest = asdict(cfg)
    manifest["selected_heads"] = [HEADS_TO_FINETUNE[layer_idx] for layer_idx in sorted(HEADS_TO_FINETUNE.keys())]
    manifest["num_records"] = num_records
    manifest["teacher_text_mean_logits_shape"] = list(teacher_shape)
    manifest["pixel_values_shape"] = list(pixel_shape)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_pt(records: List[TeacherRecord], output_path: Path, cfg: ExportConfig, model) -> None:
    if not records:
        raise ValueError("No records were exported.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_packed_pt_payload(records, cfg, model)
    torch.save(payload, output_path)

    manifest_path = output_path.with_suffix(".json")
    manifest = asdict(cfg)
    manifest["num_records"] = len(records)
    manifest["teacher_text_mean_logits_shape"] = list(records[0].teacher_text_mean_logits.shape)
    manifest["pixel_values_shape"] = list(records[0].pixel_values.shape)
    ##################################
    manifest["pt_storage_dtype"] = "bfloat16"
    manifest["pt_storage_format"] = "packed_pt_v2"
    ##################################
    manifest["selected_heads"] = payload["selected_heads"]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


##################################
def write_pt_payload(records: List[TeacherRecord], output_path: Path, cfg: ExportConfig, model) -> dict:
    if not records:
        raise ValueError("No records were exported.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_packed_pt_payload(records, cfg, model)
    torch.save(payload, output_path)
    return payload


def resolve_export_window(dataset_length: int, cfg: ExportConfig) -> tuple[int, int]:
    start_index = max(0, min(cfg.start_index, dataset_length))

    if cfg.end_index is not None:
        stop_index = max(start_index, min(cfg.end_index, dataset_length))
    else:
        stop_index = min(start_index + cfg.max_samples, dataset_length)

    return start_index, stop_index


def apply_dataset_shard(dataset: RLDSDataset, cfg: ExportConfig) -> RLDSDataset:
    if cfg.num_shards <= 1:
        return dataset

    if cfg.shard_rank < 0 or cfg.shard_rank >= cfg.num_shards:
        raise ValueError(f"`shard_rank` must be in [0, num_shards). Got rank={cfg.shard_rank}, num_shards={cfg.num_shards}")

    if not hasattr(dataset, "dataset") or not hasattr(dataset.dataset, "shard"):
        raise ValueError("The underlying RLDS dataset does not expose tf.data.Dataset.shard().")

    original_length = len(dataset)
    dataset.dataset = dataset.dataset.shard(num_shards=cfg.num_shards, index=cfg.shard_rank)
    dataset.dataset_length = max(0, (original_length - cfg.shard_rank + cfg.num_shards - 1) // cfg.num_shards)
    print(
        f"Applying dataset shard: rank={cfg.shard_rank}/{cfg.num_shards}, "
        f"shard_length={dataset.dataset_length}, original_length={original_length}"
    )
    return dataset


def skip_iterator_samples(iterator, count: int) -> None:
    for _ in range(count):
        next(iterator)


def flush_pt_shard(
    records: List[TeacherRecord],
    output_dir: Path,
    shard_idx: int,
    cfg: ExportConfig,
    model,
    shard_infos: List[Dict[str, Any]],
    global_record_offset: int,
) -> int:
    shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
    write_pt_payload(records, shard_path, cfg, model)
    shard_infos.append(
        {
            "file": shard_path.name,
            "num_records": len(records),
            "record_offset": global_record_offset,
        }
    )
    return global_record_offset + len(records)


def write_sharded_manifest(
    output_dir: Path,
    cfg: ExportConfig,
    shard_infos: List[Dict[str, Any]],
    total_records: int,
    first_record: TeacherRecord,
) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest = asdict(cfg)
    manifest.update(
        {
            "format": "teacher_sharded_pt_v1",
            "num_records": total_records,
            "shard_count": len(shard_infos),
            "teacher_text_mean_logits_shape": list(first_record.teacher_text_mean_logits.shape),
            "pixel_values_shape": list(first_record.pixel_values.shape),
            "pt_storage_dtype": "bfloat16",
            "selected_heads": [HEADS_TO_FINETUNE[layer_idx] for layer_idx in sorted(HEADS_TO_FINETUNE.keys())],
            "shards": shard_infos,
        }
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path
##################################


def write_export(records: List[TeacherRecord], output_path: Path, cfg: ExportConfig, model) -> None:
    suffix = output_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        write_pt(records, output_path, cfg, model)
        return
    if suffix in {".h5", ".hdf5"}:
        write_hdf5(records, output_path, cfg, model)
        return
    raise ValueError(f"Unsupported output suffix `{suffix}`. Use `.pt`, `.pth`, `.h5`, or `.hdf5`.")


def main() -> None:
    cfg = parse_args()

    head_scene = infer_libero_scene(
        cfg.dataset_name,
        cfg.output_path,
        cfg.output_dir,
        cfg.vla_path,
    )
    set_heads_to_finetune(head_scene)
    print(f"Using attention heads for `{head_scene}`: {summarize_heads(HEADS_TO_FINETUNE)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this export script.")

    if cfg.export_batch_size <= 0:
        raise ValueError("`export_batch_size` must be > 0.")
    if cfg.records_per_shard <= 0:
        raise ValueError("`records_per_shard` must be > 0.")
    if cfg.num_shards <= 0:
        raise ValueError("`num_shards` must be > 0.")

    model, processor = load_model_and_processor(cfg)
    proprio_projector = load_proprio_projector_if_needed(cfg, model.llm_dim)
    dataset = build_dataset(cfg, processor, model)
    dataset = apply_dataset_shard(dataset, cfg)
    collator = TeacherActionCollator(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    start_index, stop_index = resolve_export_window(len(dataset), cfg)
    total_to_export = stop_index - start_index
    if total_to_export <= 0:
        raise ValueError(
            f"No samples to export with start_index={cfg.start_index}, "
            f"end_index={cfg.end_index}, max_samples={cfg.max_samples}."
        )

    iterator = iter(dataset)
    if start_index > 0:
        print(f"Skipping first {start_index} samples before export...")
        skip_iterator_samples(iterator, start_index)

    records: List[TeacherRecord] = []
    shard_records: List[TeacherRecord] = []
    shard_infos: List[Dict[str, Any]] = []
    first_record: Optional[TeacherRecord] = None
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else None
    global_record_offset = 0
    shard_idx = 0
    exported = 0
    progress = tqdm(total=total_to_export, desc="Exporting teacher labels")
    while exported < total_to_export:
        batch_samples = []
        target_batch_size = min(cfg.export_batch_size, total_to_export - exported)
        for _ in range(target_batch_size):
            batch_samples.append(next(iterator))

        batch = collator(batch_samples)
        batch = {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        teacher_data = compute_teacher_means(
            model=model,
            proprio_projector=proprio_projector,
            batch=batch,
            use_proprio=cfg.use_proprio,
            num_images_in_input=cfg.num_images_in_input,
            average_selected_heads_for_storage=cfg.average_selected_heads_for_storage,
        )
        for batch_idx, sample in enumerate(batch_samples):
            record = sample_to_record(
                sample,
                teacher_data,
                sample_idx=batch_idx,
                pixel_values_override=batch["pixel_values"][batch_idx].detach().to(device="cpu", dtype=torch.bfloat16),
            )
            if first_record is None:
                first_record = record

            if output_dir is None:
                records.append(record)
            else:
                shard_records.append(record)
                if len(shard_records) >= cfg.records_per_shard:
                    global_record_offset = flush_pt_shard(
                        shard_records,
                        output_dir,
                        shard_idx,
                        cfg,
                        model,
                        shard_infos,
                        global_record_offset,
                    )
                    shard_records = []
                    shard_idx += 1
            print(
                f"[sample {start_index + exported + batch_idx}] dataset={record.dataset_name}, "
                f"seq_len={record.input_ids.shape[0]}, "
                f"query_len={record.text_query_mask.shape[0]}, "
                f"teacher_shape={record.teacher_text_mean_logits.shape}"
            )

        exported += len(batch_samples)
        progress.update(len(batch_samples))

    progress.close()

    if output_dir is None:
        write_export(records, Path(cfg.output_path), cfg, model)
        print(f"Saved {len(records)} records to {cfg.output_path}")
        print(f"Saved export manifest to {Path(cfg.output_path).with_suffix('.json')}")
    else:
        if shard_records:
            global_record_offset = flush_pt_shard(
                shard_records,
                output_dir,
                shard_idx,
                cfg,
                model,
                shard_infos,
                global_record_offset,
            )
        manifest_path = write_sharded_manifest(output_dir, cfg, shard_infos, global_record_offset, first_record)
        print(f"Saved {global_record_offset} records across {len(shard_infos)} shards to {output_dir}")
        print(f"Saved shard manifest to {manifest_path}")


if __name__ == "__main__":
    main()
