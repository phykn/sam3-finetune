import torch
from torch import nn


class TrackMgr(nn.Module):
    def __init__(
        self,
        track_thresh: float = 0.0,
        det_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.track_thresh = track_thresh
        self.det_thresh = det_thresh

    def forward(
        self,
        track,
        ground,
        track_ids=None,
        memory: object | None = None,
        state: object | None = None,
    ) -> dict[str, object]:
        scores = track.get("obj_scores", track.get("scores"))
        ids = self.track_ids(scores, track_ids)
        active, lost = self.split(scores, ids, self.track_thresh)
        new = self.new(ground["pred_logits"], self.det_thresh)

        return {
            "updated_video_memory": memory,
            "active_objects": active,
            "lost_objects": lost,
            "new_objects": new,
            "matches": {},
            "state": state,
        }

    @staticmethod
    def track_ids(scores: torch.Tensor, track_ids) -> tuple[int, ...]:
        count = int(scores.reshape(-1).numel())
        if track_ids is None:
            return tuple(range(count))
        if isinstance(track_ids, torch.Tensor):
            track_ids = track_ids.reshape(-1).tolist()
        return tuple(int(x) for x in track_ids)

    @staticmethod
    def split(
        scores: torch.Tensor,
        ids: tuple[int, ...],
        threshold: float,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        flat_scores = scores.reshape(-1)
        active = []
        lost = []
        for idx, obj_id in enumerate(ids):
            score = float(flat_scores[idx].detach().cpu())
            if score >= threshold:
                active.append(obj_id)
            else:
                lost.append(obj_id)
        return tuple(active), tuple(lost)

    @staticmethod
    def new(logits: torch.Tensor, threshold: float) -> tuple[int, ...]:
        out = []
        for idx, score in enumerate(logits.reshape(-1)):
            if float(score.detach().cpu()) >= threshold:
                out.append(idx)
        return tuple(out)
