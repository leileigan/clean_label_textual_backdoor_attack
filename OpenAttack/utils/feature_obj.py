from typing import List

import torch

class FeatureSpaceObj():
    def __init__(self, idx: int, text: str, text_ids: List[int], label: int, logits: List[float], cls_output: torch.FloatTensor):
        self.idx = idx
        self.text = text
        self.text_ids = text_ids
        self.label = label
        self.cls_output = cls_output
        self.pred_logits = logits

    def __str__(self) -> str:
        res = f"idx: {self.idx}, text: {self.text}, label: {self.label}"
        return res
