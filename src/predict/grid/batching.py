from .proposals import proposals_from_batch, proposals_from_low_res_batch


def decode_prompt_jobs(
    *,
    predictor,
    decode_jobs,
    crop_proposals,
    crop_grid: int,
    full_size: tuple[int, int],
    prompt_decode_batch_size: int,
    config,
    postprocess_low_res_masks,
) -> None:
    if not decode_jobs:
        return
    for start in range(0, len(decode_jobs), prompt_decode_batch_size):
        job_batch = decode_jobs[start : start + prompt_decode_batch_size]
        prompt_batches = [job[-1] for job in job_batch]
        use_low_res = hasattr(
            predictor,
            "decode_low_res_from_embedding_batches",
        )
        if use_low_res:
            results = predictor.decode_low_res_from_embedding_batches(
                prompt_batches,
                multimask_output=True,
            )
        else:
            results = predictor.predict_from_embedding_batches(
                prompt_batches,
                multimask_output=True,
            )
        for job, result in zip(job_batch, results):
            (
                crop_slot,
                crop_index,
                crop_box,
                crop_hw,
                point_batch,
                _prompt_batch,
            ) = job
            if use_low_res:
                low_res_masks, scores = result
                crop_proposals[crop_slot].extend(
                    proposals_from_low_res_batch(
                        point_batch,
                        scores,
                        low_res_masks,
                        crop_hw,
                        crop_box,
                        crop_grid,
                        crop_index,
                        full_size,
                        config=config,
                        postprocess_low_res_masks=postprocess_low_res_masks,
                    )
                )
            else:
                masks, scores, low_res_masks = result
                crop_proposals[crop_slot].extend(
                    proposals_from_batch(
                        point_batch,
                        masks,
                        scores,
                        low_res_masks,
                        crop_box,
                        crop_grid,
                        crop_index,
                        full_size,
                        config=config,
                    )
                )
