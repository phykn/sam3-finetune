def write_output(out, key, value, auxiliary=True, update_aux=True):
    out[key] = value[-1] if auxiliary else value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(value) - 1)]
        assert len(out["aux_outputs"]) == len(value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], value[:-1]):
            aux_output[key] = aux_value


def write_box_outputs(
    out,
    scores,
    boxes,
    boxes_xyxy,
    num_o2o,
    num_o2m,
    training,
):
    write_output(
        out,
        "pred_logits",
        scores[:, :, :num_o2o],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes",
        boxes[:, :, :num_o2o],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes_xyxy",
        boxes_xyxy[:, :, :num_o2o],
        update_aux=training,
    )
    if num_o2m <= 0 or not training:
        return

    write_output(
        out,
        "pred_logits_o2m",
        scores[:, :, num_o2o:],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes_o2m",
        boxes[:, :, num_o2o:],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes_xyxy_o2m",
        boxes_xyxy[:, :, num_o2o:],
        update_aux=training,
    )
