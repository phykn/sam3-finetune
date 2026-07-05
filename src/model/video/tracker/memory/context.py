import torch


def select_closest_cond_frames(
    frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False
):
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {}

    assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
    selected_outputs = {}
    if keep_first_cond_frame:
        first_idx = min((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if first_idx is None:
            first_idx = max(
                (t for t in cond_frame_outputs if t > frame_idx), default=None
            )
        if first_idx is not None:
            selected_outputs[first_idx] = cond_frame_outputs[first_idx]

    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected_outputs[idx_before] = cond_frame_outputs[idx_before]

    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected_outputs[idx_after] = cond_frame_outputs[idx_after]

    remaining_count = max_cond_frame_num - len(selected_outputs)
    remaining_indices = sorted(
        (t for t in cond_frame_outputs if t not in selected_outputs),
        key=lambda x: abs(x - frame_idx),
    )[:remaining_count]
    selected_outputs.update((t, cond_frame_outputs[t]) for t in remaining_indices)

    unselected_outputs = {
        t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
    }
    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


def select_memory_outputs(
    model,
    *,
    frame_idx,
    output_dict,
    num_frames,
    track_in_reverse,
    tpos_sign_mul,
):
    assert len(output_dict["cond_frame_outputs"]) > 0
    cond_outputs = output_dict["cond_frame_outputs"]
    selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
        frame_idx,
        cond_outputs,
        model.max_cond_frames_in_attn,
        keep_first_cond_frame=model.keep_first_cond_frame,
    )

    refs = [
        ((frame_idx - t) * tpos_sign_mul, out, True)
        for t, out in selected_cond_outputs.items()
    ]
    stride = 1 if model.training else model.memory_temporal_stride_for_eval
    valid_indices = None
    if model.use_memory_selection:
        valid_indices = filter_memory_frames(
            model, output_dict, track_in_reverse, frame_idx, num_frames, stride
        )

    for t_pos in range(1, model.num_maskmem):
        t_rel = model.num_maskmem - t_pos
        prev_frame_idx = get_previous_memory_frame(
            model,
            frame_idx=frame_idx,
            track_in_reverse=track_in_reverse,
            t_rel=t_rel,
            stride=stride,
            valid_indices=valid_indices,
        )
        if prev_frame_idx is None:
            continue

        out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
        if out is None:
            out = unselected_cond_outputs.get(prev_frame_idx, None)
        refs.append((t_pos, out, False))

    return selected_cond_outputs, unselected_cond_outputs, refs, valid_indices


def filter_memory_frames(
    model, output_dict, track_in_reverse, frame_idx, num_frames, r
):
    if (frame_idx == 0 and not track_in_reverse) or (
        frame_idx == num_frames - 1 and track_in_reverse
    ):
        return []

    max_num = min(num_frames, model.max_obj_ptrs_in_encoder)

    if not track_in_reverse:
        start = frame_idx - 1
        end = 0
        step = -r
        must_include = frame_idx - 1
    else:
        start = frame_idx + 1
        end = num_frames
        step = r
        must_include = frame_idx + 1

    valid_indices = []
    for index in range(start, end, step):
        output = output_dict["non_cond_frame_outputs"].get(index)
        if output is None or "eff_iou_score" not in output:
            continue

        if output["eff_iou_score"] > model.mf_threshold:
            valid_indices.insert(0, index)

        if len(valid_indices) >= max_num - 1:
            break

    if must_include not in valid_indices:
        valid_indices.append(must_include)

    return valid_indices


def get_previous_memory_frame(
    model,
    *,
    frame_idx,
    track_in_reverse,
    t_rel,
    stride,
    valid_indices,
):
    if model.use_memory_selection:
        if t_rel > len(valid_indices):
            return None
        return valid_indices[-t_rel]

    if t_rel == 1:
        return frame_idx + t_rel if track_in_reverse else frame_idx - t_rel

    if not track_in_reverse:
        prev_frame_idx = ((frame_idx - 2) // stride) * stride
        return prev_frame_idx - (t_rel - 2) * stride

    prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
    return prev_frame_idx + (t_rel - 2) * stride


def get_maskmem_tpos(model, t_pos, is_selected_cond_frame):
    if model.use_maskmem_tpos_v2:
        if t_pos <= 0 or t_pos >= model.num_maskmem:
            return model.maskmem_tpos_enc[model.num_maskmem - 1]
        return model.maskmem_tpos_enc[model.num_maskmem - t_pos - 1]

    t = t_pos if not is_selected_cond_frame else 0
    return model.maskmem_tpos_enc[model.num_maskmem - t - 1]


def encode_temporal_positions(
    model, rel_pos_list, device, max_abs_pos=None, dummy=False
):
    if dummy:
        return torch.zeros(len(rel_pos_list), model.mem_dim, device=device)

    t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
    pos_enc = (
        torch.tensor(rel_pos_list).pin_memory().to(device=device, non_blocking=True)
        / t_diff_max
    )
    if not model.sincos_tpos_enc:
        raise NotImplementedError

    tpos_dim = model.hidden_dim if model.proj_tpos_enc_in_obj_ptrs else model.mem_dim
    pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
    return model.obj_ptr_tpos_proj(pos_enc)


def load_memory_tensor(tensor, multiplex_state, device):
    tensor = tensor.to(device, non_blocking=True)
    if tensor.dim() == 5:
        return multiplex_state.demux(tensor).contiguous()
    return tensor


def collect_memory_prompts(model, refs, multiplex_state, device):
    prompts = []
    prompt_pos = []
    image_feats = []
    image_pos = []

    for t_pos, prev, is_selected_cond_frame in refs:
        if prev is None:
            continue

        feats = prev.get("maskmem_features")
        if feats is None:
            continue

        feats = load_memory_tensor(feats, multiplex_state, device)
        if feats.dim() != prev["maskmem_features"].dim():
            prev["maskmem_features"] = feats.cpu() if not feats.is_cuda else feats
        if feats.shape[0] == 0:
            continue

        prompts.append(feats.flatten(2).permute(2, 0, 1))

        maskmem_pos_list = prev.get("maskmem_pos_enc")
        if not maskmem_pos_list:
            continue

        maskmem_enc = maskmem_pos_list[-1]
        if maskmem_enc is None:
            continue

        maskmem_enc = load_memory_tensor(maskmem_enc, multiplex_state, device)
        if maskmem_enc.dim() != maskmem_pos_list[-1].dim():
            prev["maskmem_pos_enc"][-1] = (
                maskmem_enc.cpu() if not maskmem_enc.is_cuda else maskmem_enc
            )

        tpos_enc = get_maskmem_tpos(model, t_pos, is_selected_cond_frame)
        prompt_pos.append(maskmem_enc.flatten(2).permute(2, 0, 1) + tpos_enc)

        if model.save_image_features:
            image_feats.append(prev["image_features"].to(device))
            image_pos.append(prev["image_pos_enc"].to(device) + tpos_enc)

    return prompts, prompt_pos, image_feats, image_pos


def collect_object_pointer_refs(
    model,
    *,
    frame_idx,
    output_dict,
    num_frames,
    track_in_reverse,
    selected_cond_outputs,
    unselected_cond_outputs,
    valid_indices,
    tpos_sign_mul,
):
    max_obj_ptrs = min(num_frames, model.max_obj_ptrs_in_encoder)
    if not model.training and model.only_obj_ptrs_in_the_past_for_eval:
        ptr_cond_outputs = {
            t: out
            for t, out in selected_cond_outputs.items()
            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
        }
    else:
        ptr_cond_outputs = selected_cond_outputs

    refs = [
        (
            (
                (frame_idx - t) * tpos_sign_mul
                if model.use_signed_tpos_enc_to_obj_ptrs
                else abs(frame_idx - t)
            ),
            out,
        )
        for t, out in ptr_cond_outputs.items()
    ]

    for t_diff in range(1, max_obj_ptrs):
        if not model.use_memory_selection:
            t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
            if t < 0 or (num_frames is not None and t >= num_frames):
                break
        else:
            if -t_diff <= -len(valid_indices):
                break
            t = valid_indices[-t_diff]

        out = output_dict["non_cond_frame_outputs"].get(
            t, unselected_cond_outputs.get(t, None)
        )
        if out is not None:
            refs.append((t_diff, out))

    return max_obj_ptrs, refs


def append_object_pointer_prompts(
    model,
    *,
    prompts,
    prompt_pos,
    pointer_refs,
    max_obj_ptrs,
    device,
    batch_size,
    channels,
    multiplex_state,
):
    filtered = [(pos, out) for pos, out in pointer_refs if "obj_ptr" in out]
    if not filtered:
        return 0

    pos_list, out_list = zip(*filtered)
    obj_ptrs = (
        torch.cat([out["obj_ptr"] for out in out_list], dim=1)
        .transpose(0, 1)
        .to(device)
    )

    if model.add_tpos_enc_to_obj_ptrs:
        obj_pos = encode_temporal_positions(
            model,
            pos_list,
            max_abs_pos=max_obj_ptrs,
            device=device,
        )
    else:
        obj_pos = encode_temporal_positions(model, pos_list, device=device, dummy=True)

    obj_pos = obj_pos.unsqueeze(1).expand(-1, batch_size, -1)
    assert (
        model.mem_dim == channels
    ), f"obj_ptrs.shape = {obj_ptrs.shape}, C = {channels}"
    obj_pos = obj_pos.repeat_interleave(multiplex_state.multiplex_count, dim=0)

    prompts.append(obj_ptrs)
    prompt_pos.append(obj_pos)
    return obj_ptrs.shape[0]


def collect_memory_context(
    model,
    *,
    frame_idx,
    output_dict,
    num_frames,
    track_in_reverse,
    device,
    batch_size,
    channels,
    multiplex_state,
):
    tpos_sign_mul = -1 if track_in_reverse else 1
    selected_cond_outputs, unselected_cond_outputs, refs, valid_indices = (
        select_memory_outputs(
            model,
            frame_idx=frame_idx,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            tpos_sign_mul=tpos_sign_mul,
        )
    )
    prompts, prompt_pos, image_feats, image_pos = collect_memory_prompts(
        model,
        refs,
        multiplex_state,
        device,
    )

    num_obj_ptr_tokens = 0
    if model.use_obj_ptrs_in_encoder:
        max_obj_ptrs, pointer_refs = collect_object_pointer_refs(
            model,
            frame_idx=frame_idx,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            selected_cond_outputs=selected_cond_outputs,
            unselected_cond_outputs=unselected_cond_outputs,
            valid_indices=valid_indices,
            tpos_sign_mul=tpos_sign_mul,
        )
        num_obj_ptr_tokens = append_object_pointer_prompts(
            model,
            prompts=prompts,
            prompt_pos=prompt_pos,
            pointer_refs=pointer_refs,
            max_obj_ptrs=max_obj_ptrs,
            device=device,
            batch_size=batch_size,
            channels=channels,
            multiplex_state=multiplex_state,
        )

    return {
        "prompts": prompts,
        "prompt_pos": prompt_pos,
        "image_feats": image_feats,
        "image_pos": image_pos,
        "num_obj_ptr_tokens": num_obj_ptr_tokens,
    }
