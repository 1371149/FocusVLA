import os
import csv
import json
import bisect
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import transformers
from modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
import contextlib

# Force replacement of the classes imported from the system transformers package.
transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import math

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from collections import Counter
##################################
from local_components import TeacherActionCollator
from heads import HEADS_TO_FINETUNE, infer_libero_scene, set_heads_to_finetune, summarize_heads
##################################


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

heads_to_finetune = HEADS_TO_FINETUNE


import torch
import torch.nn.functional as F
from contextlib import contextmanager


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "<MODEL_ROOT>/openvla-7b-oft-finetuned-libero-object"             
    data_root_dir: Path = Path("<OPENVLA_FINETUNE_ROOT>/dataset")      
    dataset_name: str = "libero_object_no_noops"    
    ##################################
    clean_teacher_dataset_path: Path = Path("<OPENVLA_OFT_ROOT>/my_finetune/exports/libero_object_8head/merged")
    ##################################
    poison_data_root_dir: Path = Path("<OPENVLA_FINETUNE_ROOT>/blue_mug_dataset")
    poison_dataset_name: str = "libero_object_no_noops"
    poison_batch_size: int = 2
    run_root_dir: Path = Path("runs_object_mug_8head")
    shuffle_buffer_size: int = 100_000

    use_l1_regression: bool = True 
    use_diffusion: bool = False 
    num_diffusion_steps_train: int = 50 
    use_film: bool = False 
    num_images_in_input: int = 2 
    use_proprio: bool = True 

    batch_size: int = 2 
    ##################################
    preserve_clean_teacher_export_batches: bool = True
    clean_teacher_export_batch_size: Optional[int] = None
    ##################################
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0 
    num_steps_before_decay: int = 100000
    grad_accumulation_steps: int = 4 
    max_steps: int = 1_005
    use_val_set: bool = False 
    val_freq: int = 10 
    val_time_limit: int = 180 
    save_freq: int = 500 
    save_latest_checkpoint_only: bool = False 

    resume: bool = True 
    resume_step: Optional[int] = 150000 
    image_aug: bool = True 
    clean_aug_image_aug: bool = True
    clean_aug_batch_size: Optional[int] = None
    diffusion_sample_freq: int = 50 
    lambda_uniform: float = 2.0
    lambda_noise_attention: float = 0.1

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False 
    run_id_note: Optional[str] = None 
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10 

##################################
class TeacherStoredDataset(Dataset):
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Teacher dataset not found: {self.dataset_path}")

        self.dataset_statistics = None
        self._payload = None
        self._records = None
        self._h5_file = None
        self._manifest = None
        self._shard_infos = None
        self._shard_cumulative_sizes = None
        self._active_shard_idx = None
        self._active_shard_records = None
        self._suffix = self.dataset_path.suffix.lower()
        ##################################
        self.export_batch_size = 1
        ##################################

        if self.dataset_path.is_dir():
            self._init_from_manifest(self.dataset_path / "manifest.json")
        elif self._suffix == ".json":
            self._init_from_manifest(self.dataset_path)
        elif self._suffix in {".pt", ".pth"}:
            self._payload = torch.load(self.dataset_path, map_location="cpu")
            self._records = self._payload.get("records")
            self._length = self._payload_num_records(self._payload)
            ##################################
            sidecar_manifest_path = self.dataset_path.with_suffix(".json")
            if sidecar_manifest_path.exists():
                with open(sidecar_manifest_path, "r", encoding="utf-8") as f:
                    sidecar_manifest = json.load(f)
                self.export_batch_size = int(sidecar_manifest.get("export_batch_size", 1))
            ##################################
        elif self._suffix in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(self.dataset_path, "r") as h5_file:
                self._length = int(h5_file.attrs["num_records"])
            ##################################
            sidecar_manifest_path = self.dataset_path.with_suffix(".json")
            if sidecar_manifest_path.exists():
                with open(sidecar_manifest_path, "r", encoding="utf-8") as f:
                    sidecar_manifest = json.load(f)
                self.export_batch_size = int(sidecar_manifest.get("export_batch_size", 1))
            ##################################
        else:
            raise ValueError(f"Unsupported teacher dataset format: {self.dataset_path}")

    def _init_from_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Teacher shard manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        shard_infos = manifest.get("shards")
        if not shard_infos:
            raise ValueError(f"Shard manifest is missing `shards`: {manifest_path}")

        self._manifest = manifest
        self._shard_infos = shard_infos
        ##################################
        self.export_batch_size = int(manifest.get("export_batch_size", 1))
        ##################################
        cumulative_sizes = []
        running_total = 0
        for shard_info in shard_infos:
            running_total += int(shard_info["num_records"])
            cumulative_sizes.append(running_total)
        self._shard_cumulative_sizes = cumulative_sizes
        self._length = int(manifest.get("num_records", running_total))

    def __len__(self) -> int:
        return self._length

    ##################################
    def get_export_groups(self, group_size: int) -> List[List[int]]:
        group_size = int(group_size)
        if group_size <= 0:
            raise ValueError(f"`group_size` must be > 0, got {group_size}.")

        dataset_ranges = [(0, len(self))]
        if self._shard_infos is not None:
            base_dir = self.dataset_path if self.dataset_path.is_dir() else self.dataset_path.parent
            dataset_ranges = []
            running_total = 0
            current_range_start = 0
            current_part_key = None

            for shard_info in self._shard_infos:
                shard_path = (base_dir / shard_info["file"]).resolve()
                part_key = str(shard_path.parent)
                if current_part_key is None:
                    current_part_key = part_key
                    current_range_start = running_total
                elif part_key != current_part_key:
                    dataset_ranges.append((current_range_start, running_total))
                    current_part_key = part_key
                    current_range_start = running_total

                running_total += int(shard_info["num_records"])

            dataset_ranges.append((current_range_start, running_total))

        grouped_indices: List[List[int]] = []
        for range_start, range_end in dataset_ranges:
            for group_start in range(range_start, range_end, group_size):
                grouped_indices.append(list(range(group_start, min(group_start + group_size, range_end))))

        return grouped_indices
    ##################################

    def _get_h5_file(self):
        if self._h5_file is None:
            import h5py

            self._h5_file = h5py.File(self.dataset_path, "r")
        return self._h5_file

    def _get_active_shard_records(self, shard_idx: int):
        if self._active_shard_idx != shard_idx:
            shard_info = self._shard_infos[shard_idx]
            shard_path = (self.dataset_path if self.dataset_path.is_dir() else self.dataset_path.parent) / shard_info["file"]
            shard_payload = torch.load(shard_path, map_location="cpu")
            self._active_shard_records = shard_payload
            self._active_shard_idx = shard_idx
        return self._active_shard_records

    @staticmethod
    def _payload_num_records(payload: Dict[str, Any]) -> int:
        if "records" in payload:
            return len(payload["records"])
        if "data" in payload:
            return int(payload["data"]["sequence_length"].shape[0])
        raise KeyError("Unsupported teacher payload format.")

    @staticmethod
    def _to_tensor(array: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        ##################################
        if torch.is_tensor(array):
            tensor = array.detach().clone()
        else:
            tensor = torch.from_numpy(np.array(array, copy=True))
        ##################################
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _record_to_sample(self, record: Dict[str, Any]) -> Dict[str, Any]:
        sample = {
            "input_ids": self._to_tensor(record["input_ids"], dtype=torch.long),
            "labels": self._to_tensor(record["labels"], dtype=torch.long),
            "pixel_values": self._to_tensor(record["pixel_values"]),
            "actions": self._to_tensor(record["actions"]),
            "teacher_text_mean_logits": self._to_tensor(record["teacher_text_mean_logits"]),
            "teacher_action_mean_logits": self._to_tensor(record["teacher_action_mean_logits"]),
            "text_query_mask": self._to_tensor(record["text_query_mask"], dtype=torch.bool),
            "action_query_mask": self._to_tensor(record["action_query_mask"], dtype=torch.bool),
        }
        if "query_token_ids" in record:
            sample["query_token_ids"] = self._to_tensor(record["query_token_ids"], dtype=torch.long)
        if record.get("proprio") is not None:
            proprio = np.array(record["proprio"], dtype=np.float32, copy=True).reshape(-1)
            sample["proprio"] = torch.from_numpy(proprio)
        return sample

    def _packed_payload_to_sample(self, payload: Dict[str, Any], index: int) -> Dict[str, Any]:
        data = payload["data"]
        seq_len = int(data["sequence_length"][index])
        query_len = int(data["query_length"][index])

        sample = {
            "input_ids": self._to_tensor(data["input_ids"][index][:seq_len], dtype=torch.long),
            "labels": self._to_tensor(data["labels"][index][:seq_len], dtype=torch.long),
            "pixel_values": self._to_tensor(data["pixel_values"][index]),
            "actions": self._to_tensor(data["actions"][index]),
            "teacher_text_mean_logits": self._to_tensor(data["teacher_text_mean_logits"][index]),
            "teacher_action_mean_logits": self._to_tensor(data["teacher_action_mean_logits"][index]),
            "text_query_mask": self._to_tensor(data["text_query_mask"][index][:query_len], dtype=torch.bool),
            "action_query_mask": self._to_tensor(data["action_query_mask"][index][:query_len], dtype=torch.bool),
        }
        if "proprio" in data:
            proprio_present = bool(data["proprio_present"][index]) if "proprio_present" in data else True
            if proprio_present:
                sample["proprio"] = self._to_tensor(data["proprio"][index], dtype=torch.float32)
        return sample

    def _h5_to_sample(self, index: int) -> Dict[str, Any]:
        h5_file = self._get_h5_file()
        seq_len = int(h5_file["sequence_length"][index])
        query_len = int(h5_file["query_length"][index])

        sample = {
            "input_ids": self._to_tensor(h5_file["input_ids"][index][:seq_len], dtype=torch.long),
            "labels": self._to_tensor(h5_file["labels"][index][:seq_len], dtype=torch.long),
            "pixel_values": self._to_tensor(h5_file["pixel_values"][index]),
            "actions": self._to_tensor(h5_file["actions"][index]),
            "teacher_text_mean_logits": self._to_tensor(h5_file["teacher_text_mean_logits"][index]),
            "teacher_action_mean_logits": self._to_tensor(h5_file["teacher_action_mean_logits"][index]),
            "text_query_mask": self._to_tensor(h5_file["text_query_mask"][index][:query_len], dtype=torch.bool),
            "action_query_mask": self._to_tensor(h5_file["action_query_mask"][index][:query_len], dtype=torch.bool),
            "query_token_ids": self._to_tensor(h5_file["query_token_ids"][index][:query_len], dtype=torch.long),
        }
        if "proprio" in h5_file:
            sample["proprio"] = self._to_tensor(h5_file["proprio"][index])
        return sample

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self._records is not None:
            return self._record_to_sample(self._records[index])
        if self._payload is not None and "data" in self._payload:
            return self._packed_payload_to_sample(self._payload, index)
        if self._shard_infos is not None:
            shard_idx = bisect.bisect_right(self._shard_cumulative_sizes, index)
            shard_start = 0 if shard_idx == 0 else self._shard_cumulative_sizes[shard_idx - 1]
            shard_payload = self._get_active_shard_records(shard_idx)
            local_index = index - shard_start
            if "records" in shard_payload:
                return self._record_to_sample(shard_payload["records"][local_index])
            return self._packed_payload_to_sample(shard_payload, local_index)
        return self._h5_to_sample(index)

    def __del__(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
##################################

##################################
class RankShardedIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataset_statistics = getattr(dataset, "dataset_statistics", None)

    def __iter__(self):
        for idx, sample in enumerate(self.dataset):
            if idx % self.num_replicas == self.rank:
                yield sample

    def __len__(self) -> int:
        dataset_len = len(self.dataset)
        remaining = max(0, dataset_len - self.rank)
        return (remaining + self.num_replicas - 1) // self.num_replicas
##################################

##################################
class TeacherExportGroupedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = False,
        group_size: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size or getattr(dataset, "export_batch_size", self.batch_size))
        self.shuffle = shuffle
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

        if self.group_size <= 0:
            raise ValueError(f"`group_size` must be > 0, got {self.group_size}.")
        if self.batch_size != self.group_size:
            raise ValueError(
                "To preserve teacher export batch composition, "
                f"`batch_size` must equal export batch size. Got batch_size={self.batch_size}, "
                f"export_batch_size={self.group_size}."
            )
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"`rank` must be in [0, num_replicas). Got rank={self.rank}, num_replicas={self.num_replicas}.")

        ##################################
        if hasattr(self.dataset, "get_export_groups"):
            self.base_groups = self.dataset.get_export_groups(self.group_size)
        else:
            self.base_groups = [
                list(range(start_idx, min(start_idx + self.group_size, len(self.dataset))))
                for start_idx in range(0, len(self.dataset), self.group_size)
            ]
        ##################################
        if self.drop_last:
            self.base_groups = [group for group in self.base_groups if len(group) == self.group_size]

        self.num_groups = len(self.base_groups)
        if self.drop_last and self.num_groups % self.num_replicas != 0:
            self.num_samples = self.num_groups // self.num_replicas
        else:
            self.num_samples = int(math.ceil(self.num_groups / self.num_replicas)) if self.num_groups > 0 else 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.num_groups == 0:
            return iter([])

        group_indices = list(range(self.num_groups))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            group_indices = torch.randperm(self.num_groups, generator=generator).tolist()

        if self.drop_last:
            group_indices = group_indices[: self.total_size]
        elif len(group_indices) < self.total_size:
            padding = self.total_size - len(group_indices)
            group_indices = group_indices + group_indices[:padding]

        rank_group_indices = group_indices[self.rank : self.total_size : self.num_replicas]
        return iter([self.base_groups[group_idx] for group_idx in rank_group_indices])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
##################################

##################################
class SingleProcessDDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        yield


def dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def safe_barrier() -> None:
    if dist_is_ready():
        dist.barrier()
##################################

def remove_ddp_in_checkpoint(state_dict) -> dict:

    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:

    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:

    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False):

    if dist_is_ready():
        return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)
    return SingleProcessDDP(module)


def count_parameters(module: nn.Module, name: str) -> None:

    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
 
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print("load anction successful")

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)

##################################
def align_teacher_query_logits(query_logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    target_query_len = batch["text_query_mask"].shape[1]
    if query_logits.shape[2] == target_query_len + 1:
        return query_logits[:, :, :-1, :]
    if query_logits.shape[2] != target_query_len:
        raise ValueError(
            f"Teacher query length mismatch: attention query len = {query_logits.shape[2]}, "
            f"teacher query len = {target_query_len}"
        )
    return query_logits


def masked_query_head_mean(query_logits: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    weights = query_mask[:, None, :, None].to(device=query_logits.device, dtype=query_logits.dtype)
    denom = weights.sum(dim=2).clamp_min(1.0)
    return (query_logits * weights).sum(dim=2) / denom


##################################
def headwise_kl_from_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    tokenwise_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    return tokenwise_kl.sum(dim=-1).mean()
##################################


def compute_teacher_clean_attention_loss(
    attentions,
    batch: Dict[str, torch.Tensor],
    query_start_idx: int,
    vision_start_idx: int,
    vision_end_idx: int,
    device_id: int,
) -> Optional[torch.Tensor]:
    if "teacher_text_mean_logits" not in batch or "teacher_action_mean_logits" not in batch:
        return None

    teacher_text_mean_logits = batch["teacher_text_mean_logits"].to(device_id).float()
    teacher_action_mean_logits = batch["teacher_action_mean_logits"].to(device_id).float()
    text_query_mask = batch["text_query_mask"].to(device_id).bool()
    action_query_mask = batch["action_query_mask"].to(device_id).bool()

    teacher_clean_loss = torch.tensor(0.0, device=device_id, dtype=torch.float32)
    valid_layer_count = 0

    for layer_idx, layer_attn in enumerate(attentions):
        selected_heads = heads_to_finetune[layer_idx]
        if len(selected_heads) == 0:
            continue

        selected_logits = layer_attn[:, selected_heads, query_start_idx:, vision_start_idx:vision_end_idx].float()
        selected_logits = align_teacher_query_logits(selected_logits, batch)

        student_text_mean = masked_query_head_mean(selected_logits, text_query_mask)
        student_action_mean = masked_query_head_mean(selected_logits, action_query_mask)
        target_text_mean = teacher_text_mean_logits[:, layer_idx]
        target_action_mean = teacher_action_mean_logits[:, layer_idx]

        if target_text_mean.shape[1] == 1 and student_text_mean.shape[1] != 1:
            student_text_mean = student_text_mean.mean(dim=1, keepdim=True)
        if target_action_mean.shape[1] == 1 and student_action_mean.shape[1] != 1:
            student_action_mean = student_action_mean.mean(dim=1, keepdim=True)

        if student_text_mean.shape != target_text_mean.shape:
            raise ValueError(
                f"Teacher text logits shape mismatch at layer {layer_idx}: "
                f"student={tuple(student_text_mean.shape)}, teacher={tuple(target_text_mean.shape)}"
            )
        if student_action_mean.shape != target_action_mean.shape:
            raise ValueError(
                f"Teacher action logits shape mismatch at layer {layer_idx}: "
                f"student={tuple(student_action_mean.shape)}, teacher={tuple(target_action_mean.shape)}"
            )

        ##################################
        layer_loss = 0.5 * (
            headwise_kl_from_logits(student_text_mean, target_text_mean)
            + headwise_kl_from_logits(student_action_mean, target_action_mean)
        )
        ##################################
        teacher_clean_loss = teacher_clean_loss + layer_loss
        valid_layer_count += 1

    if valid_layer_count == 0:
        return torch.tensor(0.0, device=device_id, dtype=torch.float32)

    teacher_clean_loss = teacher_clean_loss / valid_layer_count
    return torch.nan_to_num(teacher_clean_loss, nan=0.0, posinf=1e4, neginf=-1e4)
##################################


def run_forward_pass(
    vla,
    action_head, 
    noisy_action_projector,
    proprio_projector, 
    batch, 
    action_tokenizer,
    device_id, 
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches, 
    compute_diffusion_l1=False, 
    num_diffusion_steps_train=None, 
    target_mask = None,
    lambda_attack = None,
    compute_action_loss = True,
    apply_attack_to_loss: bool = True,
    use_uniform_clean_loss: bool = False,
    lambda_uniform: float = 1e-5,
    num_images_in_input: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics = {} 

    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    ##################################
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    labels = batch["labels"].to(device_id)
    pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
    ##################################
    ##################################
    proprio_inputs = None
    if use_proprio and batch.get("proprio") is not None:
        proprio_inputs = batch["proprio"].to(device_id, dtype=torch.float32)
    ##################################

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            ##################################
            proprio=proprio_inputs,
            ##################################
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings= None,
            use_film=use_film,
        ) 
    if compute_action_loss:
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id) 
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        if not (use_l1_regression or use_diffusion):
            loss = output.loss
            predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
            curr_action_accuracy = compute_token_accuracy(
                predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
            )
            curr_action_l1_loss = compute_actions_l1_loss(
                action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
            )
            next_actions_accuracy = compute_token_accuracy(
                predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
            )
            next_actions_l1_loss = compute_actions_l1_loss(
                action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
            )
            metrics.update(
                {
                    "loss_value": loss.item(),  
                    "curr_action_accuracy": curr_action_accuracy.item(),
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_accuracy": next_actions_accuracy.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

        else:
            last_hidden_states = output.hidden_states[-1] 
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            batch_size = batch["input_ids"].shape[0]
            actions_hidden_states = (
                text_hidden_states[current_action_mask | next_actions_mask]
                .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
            ) 

            if use_l1_regression:
                predicted_actions = action_head.module.predict_action(actions_hidden_states)
                loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

            metrics.update(
                {
                    "loss_value": loss.item(), 
                }
            )

            should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
            if should_log_l1_loss:
                ground_truth_curr_action = ground_truth_actions[:, 0]
                predicted_curr_action = predicted_actions[:, 0]
                ground_truth_next_actions = ground_truth_actions[:, 1:]
                predicted_next_actions = predicted_actions[:, 1:]
                curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
                next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
                metrics.update(
                    {
                        "curr_action_l1_loss": curr_action_l1_loss.item(),
                        "next_actions_l1_loss": next_actions_l1_loss.item(),
                    }
                )
    else:
        loss = torch.tensor(0.0, device=device_id)
        metrics.update(
                {
                    "loss_value": 0, 
                    "curr_action_accuracy": 0,
                    "curr_action_l1_loss": 0,
                    "next_actions_accuracy": 0,
                    "next_actions_l1_loss": 0,
                }
            )
    if lambda_attack != None:
        action_loss = loss 
        
        attentions = output.attentions
        
        # Duplicate the mask to match the doubled batch layout.
        double_combined_mask = torch.cat([target_mask, target_mask], dim=0)
        
        # The number of image tokens is doubled accordingly.
        num_image_tokens = double_combined_mask.shape[0]
        vision_start_idx = 0
        vision_end_idx = vision_start_idx + num_image_tokens
        query_start_idx = vision_end_idx + (1 if use_proprio else 0)

        # Split the target into four float masks for loss computation.
        # Keep the head dimension so per-head effects do not cancel out prematurely.
        r1_mask_float = (double_combined_mask == 1).float().to(device_id).view(1, 1, 1, -1)
        r2_mask_float = (double_combined_mask == 2).float().to(device_id).view(1, 1, 1, -1)
        r3_mask_float = (double_combined_mask == 3).float().to(device_id).view(1, 1, 1, -1)
        r4_mask_float = (double_combined_mask == 4).float().to(device_id).view(1, 1, 1, -1)
        total_mask_sum = (r1_mask_float.sum() + r2_mask_float.sum() + r3_mask_float.sum() + r4_mask_float.sum()).clamp_min(1.0)
        
        attack_loss = 0.0
        # Keep the original layer weight schedule.
        layer_weights = [0.0001, 0.00001, 0.00001, 0.000005] + [0.000001] * 32
        
        target_r1_min = 5.0     
        target_r2_min = 2.0     
        target_r3_max = -1.0    
        target_r4_max = -15.0   
        loss_r1_mean = torch.tensor(0.0, device=device_id, dtype=torch.float32)
        loss_r2_mean = torch.tensor(0.0, device=device_id, dtype=torch.float32)
        loss_r3_mean = torch.tensor(0.0, device=device_id, dtype=torch.float32)
        loss_r4_mean = torch.tensor(0.0, device=device_id, dtype=torch.float32)

        for layer_idx, layer_attn in enumerate(attentions):
            weight = layer_weights[layer_idx]
            selected_heads = heads_to_finetune[layer_idx]
            selected_attn = layer_attn[:, selected_heads, query_start_idx:, vision_start_idx:vision_end_idx].float()
            num_selected_heads = max(1, len(selected_heads))

            # Apply a one-sided hinge loss per head first, then average over heads.
            logits_to_image = selected_attn
            loss_r1 = F.relu(target_r1_min - logits_to_image) # Pull low values up
            loss_r2 = F.relu(target_r2_min - logits_to_image) # Pull low values up
            loss_r3 = F.relu(logits_to_image - target_r3_max) # Push high values down
            loss_r4 = F.relu(logits_to_image - target_r4_max) # Push high values down

            # Apply the masks separately and average over heads to preserve scale.
            head_scale = float(num_selected_heads)
            loss_r1_mean = (loss_r1 * r1_mask_float).sum() / head_scale
            loss_r2_mean = (loss_r2 * r2_mask_float).sum() / head_scale
            loss_r3_mean = (loss_r3 * r3_mask_float).sum() / head_scale
            loss_r4_mean = (loss_r4 * r4_mask_float).sum() / head_scale

            layer_loss = (loss_r1_mean + loss_r2_mean + loss_r3_mean + loss_r4_mean) / total_mask_sum
            layer_loss = torch.nan_to_num(layer_loss, nan=0.0, posinf=1e4, neginf=-1e4)
            
            attack_loss += layer_loss * weight

            # # 3. Capture the deepest-layer logit values for training diagnostics.
            # if layer_idx == len(attentions) - 1:
            #     actual_logit_r1 = (logits_to_image * r1_mask_float).sum() / (r1_mask_float.sum() + 1e-6)
            #     actual_logit_r4 = (logits_to_image * r4_mask_float).sum() / (r4_mask_float.sum() + 1e-6)

        
        if apply_attack_to_loss:
            loss = action_loss + lambda_attack * attack_loss
        else:
            loss = action_loss

        if use_uniform_clean_loss:
            ##################################
            teacher_clean_loss = compute_teacher_clean_attention_loss(
                attentions=attentions,
                batch=batch,
                query_start_idx=query_start_idx,
                vision_start_idx=vision_start_idx,
                vision_end_idx=vision_end_idx,
                device_id=device_id,
            )
            ##################################

            if teacher_clean_loss is not None:
                ##################################
                if compute_action_loss:
                    loss = loss + lambda_uniform * teacher_clean_loss.to(loss.dtype)
                else:
                    loss = lambda_uniform * teacher_clean_loss.to(torch.float32)

                metrics["uniform_loss"] = float(teacher_clean_loss.item())
                metrics["teacher_clean_loss"] = float(teacher_clean_loss.item())
                ##################################
            else:
                eps = 1e-6
                uniform_loss = torch.tensor(0.0, device=device_id, dtype=torch.float32)
                valid_layer_count = 0
                num_images = max(1, int(num_images_in_input))
                tokens_per_image = max(1, num_image_tokens // num_images)

                for layer_idx, layer_attn in enumerate(attentions):
                    selected_heads = heads_to_finetune[layer_idx]
                    if len(selected_heads) == 0:
                        continue

                    selected_logits = layer_attn[:, selected_heads, query_start_idx:, vision_start_idx:vision_end_idx].float()
                    mean_logits = selected_logits.mean(dim=1)

                    layer_uniform = torch.tensor(0.0, device=device_id, dtype=torch.float32)
                    valid_image_count = 0
                    for img_idx in range(num_images):
                        img_start = img_idx * tokens_per_image
                        img_end = (img_idx + 1) * tokens_per_image if img_idx < num_images - 1 else num_image_tokens
                        if img_end <= img_start:
                            continue

                        img_logits = mean_logits[:, :, img_start:img_end]
                        img_probs = F.softmax(img_logits, dim=-1)
                        avg_probs = img_probs.mean(dim=1)
                        avg_probs = avg_probs / avg_probs.sum(dim=-1, keepdim=True).clamp_min(eps)

                        uniform_value = 1.0 / float(img_end - img_start)
                        img_l2 = ((avg_probs - uniform_value) ** 4).mean()
                        img_l2 = torch.nan_to_num(img_l2, nan=0.0, posinf=1e4, neginf=-1e4)

                        layer_uniform = layer_uniform + img_l2
                        valid_image_count += 1

                    if valid_image_count > 0:
                        layer_uniform = layer_uniform / valid_image_count
                        uniform_loss = uniform_loss + layer_uniform
                        valid_layer_count += 1

                if valid_layer_count > 0:
                    uniform_loss = torch.nan_to_num(uniform_loss, nan=0.0, posinf=1e4, neginf=-1e4)
                    if compute_action_loss:
                        loss = loss + lambda_uniform * uniform_loss.to(loss.dtype)
                    else:
                        loss = lambda_uniform * uniform_loss.to(torch.float32)
                else:
                    uniform_loss = torch.tensor(0.0, device=device_id, dtype=torch.float32)

                metrics["uniform_loss"] = float(torch.nan_to_num(uniform_loss, nan=0.0, posinf=1e4, neginf=-1e4).item())

        metrics.update({
            "loss_value": loss.item(),
            "action_loss": action_loss.item(),
            "attack_loss": attack_loss.item(),
            # Monitor the actual scores at the two extremes.
            "r1_logit_mean": loss_r1_mean.item() if isinstance(loss_r1_mean, torch.Tensor) else loss_r1_mean,
            "r4_logit_mean": loss_r4_mean.item() if isinstance(loss_r4_mean, torch.Tensor) else loss_r4_mean,
        })
        if (not compute_action_loss) and apply_attack_to_loss:
            loss = lambda_attack * attack_loss
            metrics["loss_value"] = loss.item()

        if (not compute_action_loss) and use_uniform_clean_loss:
            metrics["loss_value"] = loss.item()
        
        # The last two return values replace the original center_mean_logit and edge_mean_logit.
        return loss, metrics, loss_r4_mean/r4_mask_float.sum(),loss_r3_mean/r3_mask_float.sum(),loss_r2_mean/r2_mask_float.sum(), loss_r1_mean/r1_mask_float.sum()

    metrics.setdefault("action_loss", float(loss.item()))
    metrics["loss_value"] = float(loss.item())
    return loss, metrics, None, None, None, None



def compute_smoothened_metrics(metrics_deques) -> dict:
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:

    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        ##################################
        if (
            hasattr(train_dataset, "dataset_statistics")
            and train_dataset.dataset_statistics is not None
            and not isinstance(train_dataset, TeacherStoredDataset)
        ):
            save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        ##################################
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    safe_barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    safe_barrier()
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        safe_barrier()

def build_llama_language_token_ids(tokenizer):
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)

    candidate_ids = []
    for tid in range(vocab_size):
        if tid in special_ids:
            continue

        token_str = tokenizer.convert_ids_to_tokens(tid)

        if token_str is None:
            continue
        token_lower = str(token_str).lower()

        if "image" in token_lower:
            continue
        if "patch" in token_lower:
            continue
        if "extra" in token_lower:
            continue
        if "special" in token_lower:
            continue
        if "reserved" in token_lower:
            continue

        candidate_ids.append(tid)

    return torch.tensor(candidate_ids, dtype=torch.long)

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    global heads_to_finetune

    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!" 
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`") 

    head_scene = infer_libero_scene(
        cfg.dataset_name,
        cfg.clean_teacher_dataset_path,
        cfg.vla_path,
        cfg.run_root_dir,
    )
    heads_to_finetune = set_heads_to_finetune(head_scene)
    print(f"Using attention heads for `{head_scene}`: {summarize_heads(heads_to_finetune)}")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True) 

    loss_csv_path = run_dir / "train_loss_history.csv"
    loss_history_buffer = []

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    safe_barrier() # Use dist.barrier() to synchronize all processes.

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    llama_language_token_ids = build_llama_language_token_ids(processor.tokenizer)
    print(f"Number of language tokens available for noise replacement: {llama_language_token_ids.numel()}")
    print(f"Tokenizer size: {len(processor.tokenizer)}")
    print(f"special token ids: {processor.tokenizer.all_special_ids}")
    llama_language_token_ids_device = llama_language_token_ids.to(device_id)
    special_ids = set(processor.tokenizer.all_special_ids)

    def build_noisy_language_batch(batch):
        noisy_batch = dict(batch)
        noise_input_ids = batch["input_ids"].clone().to(device_id)

        mask_to_replace = torch.ones_like(noise_input_ids, dtype=torch.bool)
        for special_id in special_ids:
            mask_to_replace &= (noise_input_ids != special_id)

        num_replace = int(mask_to_replace.sum().item())
        if num_replace > 0:
            rand_idx = torch.randint(
                low=0,
                high=llama_language_token_ids_device.numel(),
                size=(num_replace,),
                device=device_id,
            )
            sampled_tokens = llama_language_token_ids_device[rand_idx]
            noise_input_ids[mask_to_replace] = sampled_tokens

        noisy_batch["input_ids"] = noise_input_ids
        return noisy_batch

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        # attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Set the number of images in the VLA input.
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.config.use_cache = False 
    # Enable gradient checkpointing to trade compute for memory.
    vla.gradient_checkpointing_enable()

    # LoRA setup.
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            # target_modules=[
            #     "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            #     "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2",
            # ],
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap the VLA with DDP.
    vla = wrap_ddp(vla, device_id, find_unused=True)
    # vla._set_static_graph() 

    # Instantiate the proprio projector if needed.
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # Instantiate the continuous action head for L1 regression if needed.
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )
        for param in action_head.parameters():
            param.requires_grad = False
    # Compute the total number of vision patches.
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # Append one proprio embedding if proprio inputs are enabled.
    if cfg.use_proprio:
        NUM_PATCHES += 1

    # Instantiate the optimizer.
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record the original learning rate.
    original_lr = optimizer.param_groups[0]["lr"]

    # Create the learning rate scheduler.
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Step at which the LR decays.
        gamma=0.1,  # LR decay factor.
    )

    # Create the action tokenizer.
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1

    # Create the clean and poisoned datasets.
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    ##################################
    train_dataset = TeacherStoredDataset(cfg.clean_teacher_dataset_path)
    ##################################
    clean_aug_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.clean_aug_image_aug,
    )
    poison_dataset = RLDSDataset(
        cfg.poison_data_root_dir,
        cfg.poison_dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    if distributed_state.is_main_process:
        ##################################
        if (
            hasattr(train_dataset, "dataset_statistics")
            and train_dataset.dataset_statistics is not None
            and not isinstance(train_dataset, TeacherStoredDataset)
        ):
            save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        ##################################

    # Create the collator and dataloaders.
    ##################################
    collator = TeacherActionCollator(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    ##################################
    clean_sampler = None
    clean_batch_sampler = None
    ##################################
    if isinstance(train_dataset, TeacherStoredDataset) and cfg.preserve_clean_teacher_export_batches:
        clean_teacher_export_batch_size = int(
            cfg.clean_teacher_export_batch_size
            if cfg.clean_teacher_export_batch_size is not None
            else getattr(train_dataset, "export_batch_size", cfg.batch_size)
        )
        clean_batch_sampler = TeacherExportGroupedBatchSampler(
            train_dataset,
            batch_size=cfg.batch_size,
            group_size=clean_teacher_export_batch_size,
            shuffle=True,
            num_replicas=distributed_state.num_processes if dist_is_ready() else 1,
            rank=distributed_state.process_index if dist_is_ready() else 0,
            drop_last=False,
        )
        if distributed_state.is_main_process:
            print(
                "Using grouped clean-teacher batching: "
                f"train_batch_size={cfg.batch_size}, export_batch_size={clean_teacher_export_batch_size}, "
                f"num_groups={len(clean_batch_sampler.base_groups)}"
            )
    elif dist_is_ready() and distributed_state.num_processes > 1:
        clean_sampler = DistributedSampler(
            train_dataset,
            num_replicas=distributed_state.num_processes,
            rank=distributed_state.process_index,
            shuffle=True,
            drop_last=False,
        )
    ##################################
    if dist_is_ready() and distributed_state.num_processes > 1:
        ##################################
        clean_aug_dataset = RankShardedIterableDataset(
            clean_aug_dataset, distributed_state.num_processes, distributed_state.process_index
        )
        poison_dataset = RankShardedIterableDataset(
            poison_dataset, distributed_state.num_processes, distributed_state.process_index
        )
        if cfg.use_val_set:
            val_dataset = RankShardedIterableDataset(
                val_dataset, distributed_state.num_processes, distributed_state.process_index
            )
        ##################################
    ##################################
    if clean_batch_sampler is not None:
        clean_dataloader = DataLoader(
            train_dataset,
            batch_sampler=clean_batch_sampler,
            collate_fn=collator,
            num_workers=0,
        )
    else:
        clean_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=clean_sampler is None,
            sampler=clean_sampler,
            collate_fn=collator,
            num_workers=0,
        )
    ##################################
    poison_dataloader = DataLoader(
        poison_dataset,
        batch_size=cfg.poison_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    clean_aug_dataloader = DataLoader(
        clean_aug_dataset,
        batch_size=cfg.clean_aug_batch_size or cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "attack_loss": deque(maxlen=cfg.grad_accumulation_steps), 
        "action_loss": deque(maxlen=cfg.grad_accumulation_steps), 
        "uniform_loss": deque(maxlen=cfg.grad_accumulation_steps), 
    }

    if distributed_state.is_main_process:
        with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "batch_idx",
                "grad_step_idx",
                "log_step",
                "is_accumulating",
                "total_loss",
                "action_loss",
                "attack_loss",
            ])


    def create_hollow_attention_target(grid_size=16, border_thickness=2, device="cuda"):
        # 1. Build the coordinate grid.
        x = torch.arange(grid_size, device=device, dtype=torch.float32)
        y = torch.arange(grid_size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 2. Compute the center coordinates.
        center = (grid_size - 1) / 2.0
        
        # 3. Compute the Euclidean distance (the key step for a circular mask).
        distances = torch.sqrt((xx - center)**2 + (yy - center)**2)
        
        # 4. Count patches for each unique radius.
        unique_dists, counts = torch.unique(distances, return_counts=True)
        
        # 5. Compute the cumulative area.
        cum_counts = torch.cumsum(counts, dim=0).float()
        total_patches = grid_size * grid_size
        
        # Target area split points: 25%, 50%, 75%.
        ideal_cum = torch.tensor([0.25, 0.50, 0.75], device=device) * total_patches
        
        # 6. Find radius thresholds closest to the target areas.
        split_indices = []
        for ideal in ideal_cum:
            diff = torch.abs(cum_counts - ideal)
            closest_idx = torch.argmin(diff).item()
            split_indices.append(closest_idx)
            
        # Ensure the split points are strictly increasing.
        for i in range(1, len(split_indices)):
            if split_indices[i] <= split_indices[i-1]:
                split_indices[i] = min(split_indices[i-1] + 1, len(unique_dists) - 1)
                
        t3 = unique_dists[split_indices[0]] # Boundary between region 4 (center) and region 3
        t2 = unique_dists[split_indices[1]] # Boundary between region 3 and region 2
        t1 = unique_dists[split_indices[2]] # Boundary between region 2 and region 1 (outermost)
        
        # 7. Assemble the final mask using labels 1, 2, 3, and 4.
        mask = torch.zeros((grid_size, grid_size), device=device, dtype=torch.int32)
        
        mask[distances > t1] = 1
        mask[(distances > t2) & (distances <= t1)] = 2
        mask[(distances > t3) & (distances <= t2)] = 3
        mask[distances <= t3] = 4
        
        # Print a summary for the terminal.
        print("="*50)
        print(f"🎯 [Circular Balanced Mask ready]")
        print(f"   - Grid Size: {grid_size} x {grid_size} ({total_patches} patches total)")
        print(f"   - Region 1 (corners + outermost ring, Target=+8.0): {(mask == 1).sum().item()} patches")
        print(f"   - Region 2 (second outer ring, Target=+3.0): {(mask == 2).sum().item()} patches")
        print(f"   - Region 3 (inner ring, Target=+0.0): {(mask == 3).sum().item()} patches")
        print(f"   - Region 4 (center circle, Target=-15.0): {(mask == 4).sum().item()} patches")
        print("="*50)
        
        # Visualize the mask to verify the circular structure.
        print("\n[Mask visualization (1=outer/corners, 4=center)]:")
        for row in mask.cpu().numpy():
            print(" ".join(map(str, row)))
        
        return mask.flatten()

    
    #############################
    attack_loss_100_buffer = deque(maxlen=100)
    clean_attack_loss_100_buffer = deque(maxlen=100)
    clean_uniform_loss_100_buffer = deque(maxlen=100)
    clean_action_loss_100_buffer = deque(maxlen=100)
    clean_aug_action_loss_100_buffer = deque(maxlen=100)
    poison_action_loss_100_buffer = deque(maxlen=100)
    p_r1_100_buffer = deque(maxlen=100)
    p_r2_100_buffer = deque(maxlen=100)
    p_r3_100_buffer = deque(maxlen=100)
    p_r4_100_buffer = deque(maxlen=100)
    noisy_poison_attack_100_buffer = deque(maxlen=100)
    ##################################
    # noisy_clean_uniform_100_buffer = deque(maxlen=100)
    # noisy_clean_loss_100_buffer = deque(maxlen=100)
    ##################################
    #############################
    target_mask = create_hollow_attention_target(device=device_id)
    lambda_attack = 1.0

    clean_iter = iter(clean_dataloader)
    clean_aug_iter = iter(clean_aug_dataloader)
    poison_iter = iter(poison_dataloader)
    total_micro_steps = cfg.max_steps * cfg.grad_accumulation_steps
    clean_sampler_state = {"epoch": 0}
    if clean_sampler is not None:
        clean_sampler.set_epoch(clean_sampler_state["epoch"])

    def next_batch(dataloader, iterator, sampler=None, sampler_state=None):
        try:
            batch = next(iterator)
        except StopIteration:
            if sampler is not None and sampler_state is not None:
                sampler_state["epoch"] += 1
                sampler.set_epoch(sampler_state["epoch"])
            iterator = iter(dataloader)
            batch = next(iterator)
        return batch, iterator

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx in range(total_micro_steps):
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            ##################################
            noisy_clean_uniform_value = 0.0
            noisy_clean_loss_value = 0.0
            noisy_poison_attack_value = 0.0
            noisy_poison_loss_value = 0.0
            noisy_poison_metrics = {}
            clean_aug_metrics = {}
            ##################################

            current_step = gradient_step_idx + 1  # Use 1-based indexing for readability.
            if current_step <= 1000:
                run_noisy = (current_step % 2 == 1)      # First 1000 steps: 50% frequency, i.e. 1,3,5,...
            else:
                run_noisy = ((current_step - 1001) % 3 == 0)   # 1001,1004,1007,...
            # lambda_attack = 1.0 * (0.1 ** (gradient_step_idx / 2000))
            lambda_attack = 1.0

            clean_batch, clean_iter = next_batch(
                clean_dataloader, clean_iter, sampler=clean_sampler, sampler_state=clean_sampler_state
            )
            poison_batch, poison_iter = next_batch(poison_dataloader, poison_iter)

            with vla.no_sync():
                clean_loss, clean_metrics, _, _, _, _ = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=clean_batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_diffusion=cfg.use_diffusion,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    compute_diffusion_l1=compute_diffusion_l1,
                    num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                    lambda_attack=lambda_attack,
                    target_mask=target_mask,
                    compute_action_loss=True,
                    apply_attack_to_loss=False,
                    use_uniform_clean_loss=True,
                    lambda_uniform=cfg.lambda_uniform,
                    num_images_in_input=cfg.num_images_in_input,
                )
                clean_loss = torch.nan_to_num(clean_loss, nan=0.0, posinf=1e4, neginf=-1e4)
                (clean_loss / cfg.grad_accumulation_steps).backward()
                del clean_loss

                poison_total_loss, poison_metrics, p_r4_mean, p_r3_mean, p_r2_mean, p_r1_mean = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=poison_batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_diffusion=cfg.use_diffusion,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    compute_diffusion_l1=compute_diffusion_l1,
                    num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                    lambda_attack=lambda_attack,
                    target_mask=target_mask,
                    compute_action_loss=True,
                    apply_attack_to_loss=True,
                    use_uniform_clean_loss=False,
                    lambda_uniform=cfg.lambda_uniform,
                    num_images_in_input=cfg.num_images_in_input,
                )
                poison_total_loss = torch.nan_to_num(poison_total_loss, nan=0.0, posinf=1e4, neginf=-1e4)
                (poison_total_loss / cfg.grad_accumulation_steps).backward()
                del poison_total_loss

                if run_noisy:
                    clean_aug_batch, clean_aug_iter = next_batch(clean_aug_dataloader, clean_aug_iter)
                    clean_aug_loss, clean_aug_metrics, _, _, _, _ = run_forward_pass(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        batch=clean_aug_batch,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=NUM_PATCHES,
                        compute_diffusion_l1=compute_diffusion_l1,
                        num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                        lambda_attack=None,
                        target_mask=None,
                        compute_action_loss=True,
                        apply_attack_to_loss=False,
                        use_uniform_clean_loss=False,
                        lambda_uniform=cfg.lambda_uniform,
                        num_images_in_input=cfg.num_images_in_input,
                    )
                    clean_aug_loss = torch.nan_to_num(clean_aug_loss, nan=0.0, posinf=1e4, neginf=-1e4)
                    (clean_aug_loss / cfg.grad_accumulation_steps).backward()
                    del clean_aug_loss

                    ##################################
                    # noisy_clean_batch = build_noisy_language_batch(clean_batch)
                    # noisy_clean_attention_loss, noisy_clean_metrics, _, _, _, _ = run_forward_pass(
                    #     vla=vla,
                    #     action_head=action_head,
                    #     noisy_action_projector=None,
                    #     proprio_projector=proprio_projector if cfg.use_proprio else None,
                    #     batch=noisy_clean_batch,
                    #     action_tokenizer=action_tokenizer,
                    #     device_id=device_id,
                    #     use_l1_regression=cfg.use_l1_regression,
                    #     use_diffusion=cfg.use_diffusion,
                    #     use_proprio=cfg.use_proprio,
                    #     use_film=cfg.use_film,
                    #     num_patches=NUM_PATCHES,
                    #     compute_diffusion_l1=compute_diffusion_l1,
                    #     num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                    #     lambda_attack=lambda_attack,
                    #     target_mask=target_mask,
                    #     compute_action_loss=False,
                    #     apply_attack_to_loss=False,
                    #     use_uniform_clean_loss=True,
                    #     lambda_uniform=cfg.lambda_uniform,
                    #     num_images_in_input=cfg.num_images_in_input,
                    # )
                    # noisy_clean_attention_loss = torch.nan_to_num(noisy_clean_attention_loss, nan=0.0, posinf=1e4, neginf=-1e4)
                    # (cfg.lambda_noise_attention * noisy_clean_attention_loss / cfg.grad_accumulation_steps).backward()
                    # del noisy_clean_attention_loss
                    ##################################

                    noisy_poison_batch = build_noisy_language_batch(poison_batch)
                    noisy_poison_attention_loss, noisy_poison_metrics, _, _, _, _ = run_forward_pass(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        batch=noisy_poison_batch,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        use_l1_regression=cfg.use_l1_regression,
                        use_diffusion=cfg.use_diffusion,
                        use_proprio=cfg.use_proprio,
                        use_film=cfg.use_film,
                        num_patches=NUM_PATCHES,
                        compute_diffusion_l1=compute_diffusion_l1,
                        num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                        lambda_attack=lambda_attack,
                        target_mask=target_mask,
                        compute_action_loss=False,
                        apply_attack_to_loss=True,
                        use_uniform_clean_loss=False,
                        lambda_uniform=cfg.lambda_uniform,
                        num_images_in_input=cfg.num_images_in_input,
                    )
                    noisy_poison_attention_loss = torch.nan_to_num(
                        noisy_poison_attention_loss, nan=0.0, posinf=1e4, neginf=-1e4
                    )
                    (cfg.lambda_noise_attention * noisy_poison_attention_loss / cfg.grad_accumulation_steps).backward()
                    del noisy_poison_attention_loss

            #############################
            clean_total_loss_value = float(clean_metrics.get("loss_value", 0.0))
            clean_total_loss_value = float(torch.nan_to_num(torch.tensor(clean_total_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            clean_action_loss_value = float(clean_metrics.get("action_loss", clean_total_loss_value))
            clean_action_loss_value = float(torch.nan_to_num(torch.tensor(clean_action_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            clean_attack_loss_value = float(clean_metrics.get("attack_loss", 0.0))
            clean_attack_loss_value = float(torch.nan_to_num(torch.tensor(clean_attack_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            clean_uniform_loss_value = float(clean_metrics.get("uniform_loss", 0.0))
            clean_uniform_loss_value = float(torch.nan_to_num(torch.tensor(clean_uniform_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            clean_aug_total_loss_value = float(clean_aug_metrics.get("loss_value", 0.0))
            clean_aug_total_loss_value = float(torch.nan_to_num(torch.tensor(clean_aug_total_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            clean_aug_action_loss_value = float(clean_aug_metrics.get("action_loss", clean_aug_total_loss_value))
            clean_aug_action_loss_value = float(torch.nan_to_num(torch.tensor(clean_aug_action_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            poison_action_loss_value = float(poison_metrics.get("action_loss", 0.0))
            poison_action_loss_value = float(torch.nan_to_num(torch.tensor(poison_action_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            poison_attack_loss_value = float(poison_metrics.get("attack_loss", 0.0))
            poison_attack_loss_value = float(torch.nan_to_num(torch.tensor(poison_attack_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            poison_total_loss_value = float(poison_metrics.get("loss_value", 0.0))
            poison_total_loss_value = float(torch.nan_to_num(torch.tensor(poison_total_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())

            noisy_poison_attack_value = float(noisy_poison_metrics.get("attack_loss", 0.0))
            noisy_poison_attack_value = float(torch.nan_to_num(torch.tensor(noisy_poison_attack_value), nan=0.0, posinf=1e4, neginf=-1e4).item())
            noisy_poison_loss_value = float(noisy_poison_metrics.get("loss_value", 0.0))
            noisy_poison_loss_value = float(torch.nan_to_num(torch.tensor(noisy_poison_loss_value), nan=0.0, posinf=1e4, neginf=-1e4).item())

            total_action_loss_value = clean_action_loss_value + clean_aug_action_loss_value + poison_action_loss_value
            total_loss_value = (
                clean_total_loss_value
                + clean_aug_total_loss_value
                + poison_total_loss_value
                + cfg.lambda_noise_attention * noisy_poison_loss_value
            )

            attack_loss_100_buffer.append(poison_attack_loss_value)
            clean_attack_loss_100_buffer.append(clean_attack_loss_value)
            clean_uniform_loss_100_buffer.append(clean_uniform_loss_value)
            clean_action_loss_100_buffer.append(clean_action_loss_value)
            if clean_aug_metrics:
                clean_aug_action_loss_100_buffer.append(clean_aug_action_loss_value)
            poison_action_loss_100_buffer.append(poison_action_loss_value)
            noisy_poison_attack_100_buffer.append(noisy_poison_attack_value)
            ##################################
            # noisy_clean_uniform_100_buffer.append(noisy_clean_uniform_value)
            # noisy_clean_loss_100_buffer.append(noisy_clean_loss_value)
            ##################################
            if p_r1_mean is not None:
                p_r1_100_buffer.append(float(torch.nan_to_num(p_r1_mean, nan=0.0, posinf=1e4, neginf=-1e4).item()))
                p_r2_100_buffer.append(float(torch.nan_to_num(p_r2_mean, nan=0.0, posinf=1e4, neginf=-1e4).item()))
                p_r3_100_buffer.append(float(torch.nan_to_num(p_r3_mean, nan=0.0, posinf=1e4, neginf=-1e4).item()))
                p_r4_100_buffer.append(float(torch.nan_to_num(p_r4_mean, nan=0.0, posinf=1e4, neginf=-1e4).item()))

            if batch_idx % 100 == 0 and distributed_state.is_main_process:
                avg_attack = sum(attack_loss_100_buffer) / max(1, len(attack_loss_100_buffer))
                avg_clean_attack = sum(clean_attack_loss_100_buffer) / max(1, len(clean_attack_loss_100_buffer))
                avg_clean_uniform = sum(clean_uniform_loss_100_buffer) / max(1, len(clean_uniform_loss_100_buffer))
                avg_clean_action = sum(clean_action_loss_100_buffer) / max(1, len(clean_action_loss_100_buffer))
                avg_clean_aug_action = sum(clean_aug_action_loss_100_buffer) / max(1, len(clean_aug_action_loss_100_buffer))
                avg_poison_action = sum(poison_action_loss_100_buffer) / max(1, len(poison_action_loss_100_buffer))
                avg_noisy_poison_attack = sum(noisy_poison_attack_100_buffer) / max(1, len(noisy_poison_attack_100_buffer))
                avg_p_r1 = sum(p_r1_100_buffer) / max(1, len(p_r1_100_buffer))
                avg_p_r2 = sum(p_r2_100_buffer) / max(1, len(p_r2_100_buffer))
                avg_p_r3 = sum(p_r3_100_buffer) / max(1, len(p_r3_100_buffer))
                avg_p_r4 = sum(p_r4_100_buffer) / max(1, len(p_r4_100_buffer))
                if len(p_r1_100_buffer) > 0:
                    progress.write(
                        f"[Batch {batch_idx}] avg_poison_attack={avg_attack:.6f}, avg_clean_attack={avg_clean_attack:.6f}, avg_clean_teacher={avg_clean_uniform:.6f}, "
                        f"avg_clean_action={avg_clean_action:.6f}, avg_clean_aug_action={avg_clean_aug_action:.6f}, avg_poison_action={avg_poison_action:.6f}, "
                        f"avg_noisy_poison_attack={avg_noisy_poison_attack:.6f}, "
                        f"p_r1={avg_p_r1:.2f}, p_r2={avg_p_r2:.2f}, "
                        f"p_r3={avg_p_r3:.2f}, p_r4={avg_p_r4:.2f}"
                    )
            #############################

            recent_metrics["loss_value"].append(total_loss_value)
            recent_metrics["action_loss"].append(total_action_loss_value)
            recent_metrics["attack_loss"].append(poison_attack_loss_value + cfg.lambda_noise_attention * noisy_poison_attack_value)
            recent_metrics["uniform_loss"].append(clean_uniform_loss_value)
            if "curr_action_l1_loss" in clean_metrics:
                recent_metrics["curr_action_l1_loss"].append(clean_metrics["curr_action_l1_loss"])
            if "next_actions_l1_loss" in clean_metrics:
                recent_metrics["next_actions_l1_loss"].append(clean_metrics["next_actions_l1_loss"])

            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            is_accumulating = int((batch_idx + 1) % cfg.grad_accumulation_steps != 0)

            if distributed_state.is_main_process:
                curr_total_loss = total_loss_value
                curr_action_loss = total_action_loss_value
                curr_attack_loss = poison_attack_loss_value + cfg.lambda_noise_attention * noisy_poison_attack_value
                loss_history_buffer.append(
                    {
                        "batch_idx": batch_idx,
                        "grad_step_idx": gradient_step_idx,
                        "log_step": log_step,
                        "is_accumulating": is_accumulating,
                        "total_loss": curr_total_loss,
                        "action_loss": curr_action_loss,
                        "attack_loss": curr_attack_loss,
                    }
                )
                with open(loss_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            batch_idx,
                            gradient_step_idx,
                            log_step,
                            is_accumulating,
                            curr_total_loss,
                            curr_action_loss,
                            curr_attack_loss,
                        ]
                    )

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            is_update_step = (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            if is_update_step:
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                if world_size > 1:
                    for param in vla.parameters():
                        if param.requires_grad and param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= world_size

                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            if is_update_step and gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            if gradient_step_idx >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

if __name__ == "__main__":
    finetune()
