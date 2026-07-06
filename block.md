# SAM3 Rewrite Block Structure

이 문서는 `sam3-main`을 참고해 `src/` rewrite에서 사용할 블록 구조를 정리한 것이다.
목표는 upstream SAM3를 그대로 복제하는 것이 아니라, point/box/mask prompt 기반 segmentation을
중심으로 필요한 SAM3/SAM2/SAM-style 구성요소를 재조합하는 것이다.

## Overall Judgment

아래 블록 분해는 rewrite 설계 기준으로 타당하다.

다만 upstream SAM3의 기본 image grounding 주 경로는 SAM1-style mask decoder가 아니라
SAM3 grounding transformer decoder와 segmentation head다. 따라서 이 설계는 upstream 구조의
직접 복제가 아니라, SAM1-style mask decoder를 최종 마스크 생성기로 두고 SAM3 grounding과
video memory를 후보 생성 및 보정용으로 붙이는 구조로 해석하는 것이 정확하다.

## Block 1. Shared Vision Core

Input:
- image

Output:
- shared_visual_features
- optional multi-scale visual features

Role:
- 모든 경로가 공유하는 ViT 기반 vision feature를 생성한다.
- 실제 SAM3 main에서는 단순 ViT trunk만이 아니라 neck/FPN 출력까지 함께 고려해야 한다.
- 이후 SAM3 grounding용 feature와 SAM2/SAM-style tracker용 feature가 갈라진다.

Judgment:
- 타당하다.
- 단, 구현 경계는 `raw ViT feature`보다 `ViT + neck/FPN output` 쪽에 가깝게 잡는 것이 안전하다.

## Block 2. SAM1-style Image Feature Adapter

Input:
- shared_visual_features

Output:
- image_embed
- high_res_features
- orig_hw

Role:
- 공통 visual feature를 SAM1-style mask decoder가 사용할 이미지 embedding으로 변환한다.
- SAM-style decoder의 `image_embeddings`와 `high_res_features` 입력 형태를 맞춘다.

Judgment:
- 타당하다.
- 실제 SAM3 main에서는 SAM-style head가 tracker/instance interactivity 경로에 있으므로,
  rewrite에서는 이 블록을 명시적으로 분리하는 것이 좋다.

## Block 3. SAM1-style Prompt Encoder

Input:
- point
- box
- mask_input

Output:
- sparse_embeddings
- dense_embeddings

Role:
- 사용자 prompt를 SAM mask decoder용 sparse/dense prompt embedding으로 변환한다.
- point/box는 sparse prompt, mask input은 dense prompt로 들어간다.

Judgment:
- 타당하다.
- Block 7의 SAM3 geometry prompt encoder와는 별도 블록으로 유지해야 한다.

## Block 4. SAM1-style Mask Decoder

Input:
- image_embed
- high_res_features
- sparse_embeddings
- dense_embeddings
- image_pe

Output:
- low_res_masks
- iou_predictions
- optional object_score_logits

Role:
- SAM1/SAM2-style 최종 마스크 후보를 생성한다.
- 이 설계에서는 최종 마스크 생성기로 둔다.
- LoRA 적용 1순위 대상으로 적합하다.

Judgment:
- rewrite 설계 기준으로 타당하다.
- upstream SAM3 image grounding의 기본 최종 head는 아니지만, promptable segmentation 중심 구조에서는
  최종 mask head로 두는 선택이 자연스럽다.

## SAM1-style Core Flow

```text
image
-> Block 1
-> Block 2
-> image_embed, high_res_features

point / box / mask
-> Block 3
-> sparse_embeddings, dense_embeddings

Block 2 output + Block 3 output
-> Block 4
-> low_res_masks
```

## Block 5. Cached Visual Condition Block

Input:
- cache file/config

Output:
- cached_visual_condition
  - language_features
  - language_mask
  - optional language_embeds

Role:
- text encoder를 runtime에 실행하지 않고, 미리 캐시된 `"visual"` condition tensor를 사용한다.
- no-text image grounding에서 geometry prompt만으로 SAM3 grounding path를 사용할 수 있게 한다.

Judgment:
- 타당하다.
- 실제 코드도 text prompt가 없을 때 `"visual"` text feature를 사용하므로, 이를 캐시 주입으로 대체하는
  방향은 맞다.
- shape와 `text_ids` 선택 규칙을 upstream 출력과 맞추는 것이 중요하다.

## Block 6. SAM3 Grounding Image Feature Adapter

Input:
- shared_visual_features

Output:
- grounding_backbone_out
  - vision_features
  - vision_pos_enc
  - backbone_fpn
  - spatial size metadata

Role:
- 공통 visual feature를 SAM3 grounding transformer가 사용할 image feature dict로 변환한다.

Judgment:
- 타당하다.
- 실제 SAM3에서는 `backbone.forward_image(...)` 결과가 이미 이 형태에 가깝다.

## Block 7. Grounding Prompt Encoder

Input:
- point
- box
- optional mask
- grounding_backbone_out
- cached_visual_condition

Output:
- grounding_prompt_features
- grounding_prompt_mask

Role:
- point/box/mask를 SAM3 grounding용 geometry prompt feature로 변환한다.
- cached visual/text condition과 함께 grounding transformer encoder/decoder에 들어갈 prompt sequence를 만든다.

Judgment:
- 타당하다.
- 이 블록은 SAM-style `PromptEncoder`가 아니라 SAM3의 geometry prompt encoder로 봐야 한다.

## Block 8. SAM3 Grounding DETR Decoder + Segmentation Head

Input:
- grounding_backbone_out
- cached_visual_condition
- grounding_prompt_features
- grounding_prompt_mask

Output:
- pred_logits
- pred_boxes
- pred_masks

Role:
- DETR-style object query 기반 grounding 후보를 생성한다.
- rewrite에서는 raw 후보, box prior, mask prior, object proposal로 사용한다.

Judgment:
- 타당하다.
- 이 블록은 upstream SAM3 image grounding의 주 경로에 해당한다.
- 최종 mask를 Block 4가 만들더라도, Block 8은 후보 생성 및 보정에 유용하다.

## SAM3 Grounding Flow

```text
image
-> Block 1
-> Block 6
-> grounding_backbone_out

cached visual tensor
-> Block 5
-> cached_visual_condition

point / box / mask
-> Block 7
-> grounding_prompt_features

Block 5 + Block 6 + Block 7
-> Block 8
-> pred_logits, pred_boxes, pred_masks
```

## Block 9. Video Feature Adapter

Input:
- shared_visual_features

Output:
- video_frame_features

Role:
- 공통 visual feature를 SAM2/SAM3 tracker가 사용할 frame feature로 변환한다.

Judgment:
- 타당하다.
- 실제 SAM3 video path에서는 detector가 tracker용 backbone feature를 함께 만들고 cache에 저장한다.

## Block 10. Video Memory Encoder

Input:
- video_frame_features
- reference_mask
- optional obj_id

Output:
- video_memory
- object_memory
- object_state

Role:
- reference frame feature와 mask를 이용해 객체 memory를 생성한다.
- 이후 frame propagation에서 해당 object memory를 사용한다.

Judgment:
- 타당하다.

## Block 11. Video Propagation / Tracker Decoder

Input:
- next_video_frame_features
- video_memory

Output:
- propagated_mask_logits
- obj_scores

Role:
- memory에 등록된 객체를 다음 프레임으로 전파한다.
- 객체가 사라지거나 occlusion 상태이면 낮은 object score 또는 빈 mask에 가까운 logits를 낸다.

Judgment:
- 타당하다.

## Video Flow

Initialization:

```text
reference_frame
-> Block 1
-> Block 9
-> video_frame_features

reference_mask + video_frame_features
-> Block 10
-> video_memory
```

Propagation:

```text
next_frame
-> Block 1
-> Block 9
-> next_video_frame_features

next_video_frame_features + video_memory
-> Block 11
-> propagated_mask_logits, obj_scores
```

## Block 12. Video Track Manager / Candidate Manager

Input:
- Block 11 output
  - propagated_mask_logits
  - obj_scores
- Block 8 output
  - pred_logits
  - pred_boxes
  - pred_masks
- current video_memory
- current object states

Output:
- updated_video_memory
- active_objects
- lost_objects
- new_objects

Role:
- tracker 결과와 grounding 후보를 매칭한다.
- score가 낮거나 오래 사라진 객체를 lost/inactive/remove 처리한다.
- 새 grounding 후보를 새 객체로 등록한다.
- 필요한 경우 detector mask로 tracker memory를 recondition한다.

Judgment:
- 타당하고 필요하다.
- 실제 SAM3 video에도 detector-tracker association, object add/remove, hotstart, reconditioning,
  memory update에 해당하는 orchestration이 존재한다.

## Full Structure

```text
                    +-> Block 2 -> Block 4
                    |   SAM1-style segmentation
                    |
image -> Block 1 ---+-> Block 6 -> Block 8
shared vision       |   SAM3 grounding
                    |
                    +-> Block 9 -> Block 10 / 11
                        SAM2/SAM3 video memory
```

## Final Design Direction

Main output:
- Block 4 SAM1-style Mask Decoder

Auxiliary candidates:
- Block 8 SAM3 Grounding
- Block 11 Video Propagation

Connection management:
- Block 12 Track Manager / Candidate Manager

In short:

```text
SAM1-style mask decoder = final mask generator
SAM3 grounding = object/mask candidate generator
SAM2/SAM3 video memory = temporal propagation and correction
Track/Candidate manager = association, lifecycle, and memory update
```

This block layout is valid for the rewrite target.

## Implementation Map

Current `src/model/blocks/` files:

- Block 1: `vision.py` / `VisionCore`
- Block 2: `sam_image.py` / `SamImage`
- Block 3: `sam_prompt.py` / `SamPrompt`
- Block 4: `sam_mask.py` / `SamMask`
- Block 5: `cond.py` / `VisualCond`
- Block 6: `ground_image.py` / `GroundImage`
- Block 7: `ground_prompt.py` / `GroundPrompt`
- Block 8: `ground_dec.py` / `GroundDec`
- Block 9: `video_feat.py` / `VideoFeat`
- Block 10: `video_mem.py` / `VideoMem`
- Block 11: `video_track.py` / `VideoTrack`
- Block 12: `track_mgr.py` / `TrackMgr`
