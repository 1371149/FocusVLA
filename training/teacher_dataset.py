import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


def _sorted_pt_files(source_dir: Path) -> List[Path]:
    shard_files = sorted(path for path in source_dir.glob("*.pt") if path.is_file())
    if not shard_files:
        raise FileNotFoundError(f"No `.pt` files found in `{source_dir}`.")
    return shard_files


class TeacherAttentionDataset(Dataset):
    def __init__(self, source: str) -> None:
        self.source = Path(source)
        if self.source.is_dir():
            dataset_from_manifest = self.from_directory_with_manifest(self.source)
            if dataset_from_manifest is not None:
                self.shard_files = dataset_from_manifest.shard_files
                self.shard_sizes = dataset_from_manifest.shard_sizes
                self.cumulative_sizes = dataset_from_manifest.cumulative_sizes
                self._cached_shard_idx = None
                self._cached_payload = None
                return

        self.shard_files = self._discover_shards(self.source)
        self.shard_sizes = self._discover_shard_sizes(self.shard_files)

        self.cumulative_sizes: List[int] = []
        running_total = 0
        for shard_size in self.shard_sizes:
            running_total += shard_size
            self.cumulative_sizes.append(running_total)

        self._cached_shard_idx: Optional[int] = None
        self._cached_payload: Optional[Dict[str, Any]] = None

    @staticmethod
    def _discover_shards(source: Path) -> List[Path]:
        if source.is_file():
            return [source]
        if source.is_dir():
            return _sorted_pt_files(source)
        raise FileNotFoundError(f"Teacher dataset source `{source}` does not exist.")

    @staticmethod
    def _discover_shard_sizes(shard_files: Sequence[Path]) -> List[int]:
        shard_sizes: List[int] = []
        for shard_file in shard_files:
            payload = torch.load(shard_file, map_location="cpu")
            if "records" in payload:
                shard_sizes.append(len(payload["records"]))
            elif "data" in payload:
                shard_sizes.append(int(payload["data"]["sequence_length"].shape[0]))
            else:
                raise KeyError(f"Unsupported payload format in `{shard_file}`.")
        return shard_sizes

    @staticmethod
    def from_directory_with_manifest(source_dir: Path) -> Optional["TeacherAttentionDataset"]:
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if "shards" not in manifest:
            return None

        dataset = TeacherAttentionDataset.__new__(TeacherAttentionDataset)
        dataset.source = source_dir
        dataset.shard_files = [source_dir / shard["file"] for shard in manifest["shards"]]
        dataset.shard_sizes = [int(shard["num_records"]) for shard in manifest["shards"]]
        dataset.cumulative_sizes = []
        running_total = 0
        for shard_size in dataset.shard_sizes:
            running_total += shard_size
            dataset.cumulative_sizes.append(running_total)
        dataset._cached_shard_idx = None
        dataset._cached_payload = None
        return dataset

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        if self._cached_shard_idx != shard_idx or self._cached_payload is None:
            self._cached_payload = torch.load(self.shard_files[shard_idx], map_location="cpu")
            self._cached_shard_idx = shard_idx
        return self._cached_payload

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset length {len(self)}.")

        shard_idx = bisect_right(self.cumulative_sizes, idx)
        prev_cumulative = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        local_idx = idx - prev_cumulative

        payload = self._load_shard(shard_idx)
        if "records" in payload:
            record = payload["records"][local_idx]
            return self._convert_record(record)
        return self._convert_packed_record(payload, local_idx)

    @staticmethod
    def _convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
        sample: Dict[str, Any] = {
            "dataset_name": record["dataset_name"],
            "language_instruction": record["language_instruction"],
            "episode_file_path": record["episode_file_path"],
            "timestep": record["timestep"],
            "input_ids": torch.as_tensor(record["input_ids"], dtype=torch.long),
            "labels": torch.as_tensor(record["labels"], dtype=torch.long),
            "actions": torch.as_tensor(record["actions"], dtype=torch.float32),
            "pixel_values": torch.as_tensor(record["pixel_values"], dtype=torch.float32),
            "teacher_text_mean_logits": torch.as_tensor(record["teacher_text_mean_logits"], dtype=torch.float32),
            "teacher_action_mean_logits": torch.as_tensor(record["teacher_action_mean_logits"], dtype=torch.float32),
            "text_query_mask": torch.as_tensor(record["text_query_mask"], dtype=torch.bool),
            "action_query_mask": torch.as_tensor(record["action_query_mask"], dtype=torch.bool),
        }
        if "query_token_ids" in record:
            sample["query_token_ids"] = torch.as_tensor(record["query_token_ids"], dtype=torch.long)

        if record.get("primary_image_uint8") is not None:
            sample["primary_image_uint8"] = torch.as_tensor(record["primary_image_uint8"], dtype=torch.uint8)
        if record.get("wrist_image_uint8") is not None:
            sample["wrist_image_uint8"] = torch.as_tensor(record["wrist_image_uint8"], dtype=torch.uint8)
        if record.get("proprio") is not None:
            sample["proprio"] = torch.as_tensor(record["proprio"], dtype=torch.float32)

        return sample

    @staticmethod
    def _convert_packed_record(payload: Dict[str, Any], idx: int) -> Dict[str, Any]:
        data = payload["data"]
        seq_len = int(data["sequence_length"][idx])
        query_len = int(data["query_length"][idx])

        sample: Dict[str, Any] = {
            "input_ids": torch.as_tensor(data["input_ids"][idx][:seq_len], dtype=torch.long),
            "labels": torch.as_tensor(data["labels"][idx][:seq_len], dtype=torch.long),
            "actions": torch.as_tensor(data["actions"][idx], dtype=torch.float32),
            "pixel_values": torch.as_tensor(data["pixel_values"][idx], dtype=torch.float32),
            "teacher_text_mean_logits": torch.as_tensor(data["teacher_text_mean_logits"][idx], dtype=torch.float32),
            "teacher_action_mean_logits": torch.as_tensor(data["teacher_action_mean_logits"][idx], dtype=torch.float32),
            "text_query_mask": torch.as_tensor(data["text_query_mask"][idx][:query_len], dtype=torch.bool),
            "action_query_mask": torch.as_tensor(data["action_query_mask"][idx][:query_len], dtype=torch.bool),
        }
        if "proprio" in data:
            proprio_present = bool(data["proprio_present"][idx]) if "proprio_present" in data else True
            if proprio_present:
                sample["proprio"] = torch.as_tensor(data["proprio"][idx], dtype=torch.float32)
        return sample
