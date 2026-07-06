import torch
from timm.layers import trunc_normal_
from torch import nn


def init_object_pointer_params(
    self,
    pred_obj_scores,
    pred_obj_scores_mlp,
    fixed_no_obj_ptr,
    use_no_obj_ptr,
    use_linear_no_obj_ptr,
    use_mlp_for_obj_ptr_proj,
    no_obj_embed_spatial,
):
    self.pred_obj_scores = pred_obj_scores
    self.pred_obj_scores_mlp = pred_obj_scores_mlp
    self.fixed_no_obj_ptr = fixed_no_obj_ptr
    self.use_no_obj_ptr = use_no_obj_ptr
    self.use_linear_no_obj_ptr = use_linear_no_obj_ptr

    if self.fixed_no_obj_ptr:
        assert self.pred_obj_scores
        assert self.use_obj_ptrs_in_encoder
    if self.pred_obj_scores and self.use_obj_ptrs_in_encoder and self.use_no_obj_ptr:
        if self.use_linear_no_obj_ptr:
            self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.no_obj_ptr = nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.no_obj_ptr, std=0.02)

    self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
    self.no_obj_embed_spatial = None
    if no_obj_embed_spatial:
        self.no_obj_embed_spatial = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)


def init_condition_embedding_params(
    self,
    add_output_suppression_embeddings,
    add_object_conditional_embeddings,
    add_object_unconditional_embeddings,
    condition_as_mask_input,
    condition_as_mask_input_fg,
    condition_as_mask_input_bg,
):
    self.add_output_suppression_embeddings = add_output_suppression_embeddings
    if self.add_output_suppression_embeddings:
        self.output_valid_embed = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        self.output_invalid_embed = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.output_valid_embed, std=0.02)
        trunc_normal_(self.output_invalid_embed, std=0.02)

    self.add_object_conditional_embeddings = add_object_conditional_embeddings
    if add_object_unconditional_embeddings is None:
        add_object_unconditional_embeddings = add_object_conditional_embeddings
    self.add_object_unconditional_embeddings = add_object_unconditional_embeddings
    if add_object_unconditional_embeddings:
        assert add_object_conditional_embeddings
    if self.add_object_conditional_embeddings:
        self.obj_cond_embed = nn.Parameter(
            torch.zeros(self.multiplex_count, self.hidden_dim)
        )
        trunc_normal_(self.obj_cond_embed, std=0.02)
        if self.add_object_unconditional_embeddings:
            self.obj_non_cond_embed = nn.Parameter(
                torch.zeros(self.multiplex_count, self.hidden_dim)
            )
            trunc_normal_(self.obj_non_cond_embed, std=0.02)

    self.condition_as_mask_input = condition_as_mask_input
    self.condition_as_mask_input_fg = condition_as_mask_input_fg
    self.condition_as_mask_input_bg = condition_as_mask_input_bg


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
