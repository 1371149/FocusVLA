from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
PROPRIO_DIM = 8


class PromptBuilder:
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family
        self.system_prompt = system_prompt

    def add_turn(self, role: str, message: str) -> str:
        raise NotImplementedError

    def get_potential_prompt(self, user_msg: str) -> str:
        raise NotImplementedError

    def get_prompt(self) -> str:
        raise NotImplementedError


class PurePromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.bos, self.eos = "<s>", "</s>"
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()
        if (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)
        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        prompt_copy = str(self.prompt)
        prompt_copy += self.wrap_human(message)
        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        return self.prompt.removeprefix(self.bos).rstrip()


class ActionTokenizer:
    def __init__(
        self, tokenizer, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins


class ProprioProjector(nn.Module):
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim
        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


def get_current_action_mask(token_ids: torch.Tensor) -> torch.Tensor:
    valid_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(valid_positions, dim=1)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    action_tokens_only = token_ids > ACTION_TOKEN_BEGIN_IDX
    return action_tokens_only * mask


def get_next_actions_mask(token_ids: torch.Tensor) -> torch.Tensor:
    valid_positions = token_ids != IGNORE_INDEX
    cumsum = torch.cumsum(valid_positions, dim=1)
    mask = cumsum > ACTION_DIM
    action_tokens_only = token_ids > ACTION_TOKEN_BEGIN_IDX
    return action_tokens_only * mask


def compute_token_accuracy(predicted_token_ids: torch.Tensor, ground_truth_token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    return correct_preds.sum().float() / mask.sum().float().clamp_min(1.0)


def compute_actions_l1_loss(action_tokenizer: ActionTokenizer, predicted_token_ids: torch.Tensor, ground_truth_token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    return torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)


@dataclass
class TeacherExportBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: any
    image_transform: any
    prompt_builder_fn: any
    include_raw_images: bool = True
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False


@dataclass
class LocalRLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: Any
    image_transform: Any
    prompt_builder_fn: Any
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
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

        sample = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
        )

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

        return sample


@dataclass
class TeacherActionCollator:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.padding_side == "right", f"Invalid tokenizer padding side `{self.padding_side}`"

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        if "pixel_values_wrist" in instances[0]:
            pixel_values_primary = torch.stack([instance["pixel_values"] for instance in instances])
            pixel_values_wrist = torch.stack([instance["pixel_values_wrist"] for instance in instances])
            pixel_values = torch.cat((pixel_values_primary, pixel_values_wrist), dim=1)
        else:
            pixel_values = torch.stack([instance["pixel_values"] for instance in instances])
        actions = torch.stack([torch.as_tensor(np.copy(instance["actions"])) for instance in instances])

        output = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "actions": actions,
        }

        if "dataset_name" in instances[0]:
            output["dataset_names"] = [instance["dataset_name"] for instance in instances]

        if "proprio" in instances[0]:
            proprio = [torch.as_tensor(np.array(instance["proprio"], dtype=np.float32, copy=True)) for instance in instances]
            output["proprio"] = torch.stack(proprio)

        if "teacher_text_mean_logits" in instances[0]:
            output["teacher_text_mean_logits"] = torch.stack([instance["teacher_text_mean_logits"] for instance in instances])
            output["teacher_action_mean_logits"] = torch.stack([instance["teacher_action_mean_logits"] for instance in instances])

            text_query_masks = [instance["text_query_mask"] for instance in instances]
            action_query_masks = [instance["action_query_mask"] for instance in instances]
            has_query_token_ids = "query_token_ids" in instances[0]
            if has_query_token_ids:
                query_token_ids = [instance["query_token_ids"] for instance in instances]

            effective_query_lengths = [
                max(0, min(text_mask.shape[0], min(instance["input_ids"].shape[0], self.model_max_length) - 1))
                for text_mask, instance in zip(text_query_masks, instances)
            ]
            max_query_len = max(effective_query_lengths) if effective_query_lengths else 0
            batch_size = len(instances)

            text_mask_tensor = torch.zeros((batch_size, max_query_len), dtype=torch.bool)
            action_mask_tensor = torch.zeros((batch_size, max_query_len), dtype=torch.bool)
            query_lengths = torch.zeros((batch_size,), dtype=torch.long)
            if has_query_token_ids:
                query_token_tensor = torch.full((batch_size, max_query_len), fill_value=-1, dtype=torch.long)

            for idx, (text_mask, action_mask) in enumerate(zip(text_query_masks, action_query_masks)):
                qlen = effective_query_lengths[idx]
                if qlen == 0:
                    query_lengths[idx] = 0
                    continue
                text_mask_tensor[idx, :qlen] = text_mask[:qlen]
                action_mask_tensor[idx, :qlen] = action_mask[:qlen]
                if has_query_token_ids:
                    query_token_tensor[idx, :qlen] = query_token_ids[idx][:qlen]
                query_lengths[idx] = qlen

            output["text_query_mask"] = text_mask_tensor
            output["action_query_mask"] = action_mask_tensor
            if has_query_token_ids:
                output["query_token_ids"] = query_token_tensor
            output["query_lengths"] = query_lengths

        return output
