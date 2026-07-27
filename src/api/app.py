import base64
import os
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from ..predict import GroundPredictor

ROOT = Path(__file__).resolve().parents[2]
WEIGHT = Path(os.getenv("SAM3_WEIGHT", ROOT / "weight" / "sam3.1_multiplex.pt"))
VISUAL = Path(os.getenv("SAM3_VISUAL", ROOT / "weight" / "visual_token.pt"))
DEVICE = os.getenv("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
COLORS = ("#64D9C2", "#78A7FF", "#F5B95F", "#C995FF", "#FF8A74")

app = FastAPI(title="SAM 3 Similar Object API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = Lock()
_predictor = None
_sessions = {}


class Prompt(BaseModel):
    box: tuple[float, float, float, float]
    positive: bool = True


class PromptUpdate(BaseModel):
    box: tuple[float, float, float, float]


class PointPrompt(BaseModel):
    point: tuple[float, float]
    positive: bool = True


class PointUpdate(BaseModel):
    point: tuple[float, float]


class PointDelete(BaseModel):
    indices: list[int]


def predictor():
    global _predictor
    if _predictor is None:
        if not WEIGHT.is_file() or not VISUAL.is_file():
            raise RuntimeError("local grounding weights are missing")
        _predictor = GroundPredictor.from_path(
            WEIGHT,
            VISUAL,
            device=DEVICE,
            score_thr=0.45,
            sim_thr=0.45,
            negative_margin=0.05,
            top_k=None,
        )
    return _predictor


def session(session_id):
    if session_id not in _sessions:
        raise HTTPException(404, "image session was not found")
    return _sessions[session_id]


def mask_uri(roi, color):
    value = np.asarray(roi, dtype=bool)
    rgba = np.zeros((*value.shape, 4), dtype=np.uint8)
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    rgba[..., :3] = rgb
    rgba[..., 3] = value * 132
    buffer = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pack_objects(objects):
    out = []
    for index, item in enumerate(objects):
        color = COLORS[index % len(COLORS)]
        out.append(
            {
                "object_id": item["object_id"],
                "box": list(item["box"]),
                "mask": mask_uri(item["roi"], color),
                "color": color,
                "metrics": item["metrics"],
            }
        )
    return out


def result(data, objects):
    data["objects"] = objects
    return {
        "session_id": data["id"],
        "width": data["width"],
        "height": data["height"],
        "prompt_count": len(data["state"]["boxes"]),
        "objects": pack_objects(objects),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": _predictor is not None,
    }


@app.post("/api/sessions")
def create_session(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(415, "upload an image file")
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image file is too large")
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "image could not be decoded") from error
    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(413, "image dimensions are too large")

    with _lock:
        try:
            state = predictor().start(image)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        data = {
            "id": uuid4().hex,
            "state": state,
            "width": image.width,
            "height": image.height,
        }
        _sessions[data["id"]] = data
        while len(_sessions) > 8:
            _sessions.pop(next(iter(_sessions)))
        return result(data, [])


@app.post("/api/sessions/{session_id}/prompts")
def add_prompt(session_id: str, prompt: Prompt):
    with _lock:
        data = session(session_id)
        try:
            objects = predictor().add_prompt(
                data["state"],
                prompt.box,
                positive=prompt.positive,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.post("/api/sessions/{session_id}/points")
def add_point(session_id: str, prompt: PointPrompt):
    with _lock:
        data = session(session_id)
        try:
            objects = predictor().add_point(
                data["state"],
                prompt.point,
                positive=prompt.positive,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.post("/api/sessions/{session_id}/objects/{object_id}/exclude")
def exclude_object(session_id: str, object_id: int):
    with _lock:
        data = session(session_id)
        item = next(
            (
                value
                for value in data.get("objects", [])
                if value["object_id"] == object_id
            ),
            None,
        )
        if item is None:
            raise HTTPException(404, "result object was not found")
        try:
            objects = predictor().add_prompt(
                data["state"],
                item["box"],
                positive=False,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.post("/api/sessions/{session_id}/refine")
def refine_results(session_id: str):
    with _lock:
        data = session(session_id)
        try:
            objects = predictor().refine_objects(
                data["state"],
                data.get("objects", []),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.post("/api/sessions/{session_id}/points/delete")
def delete_points(session_id: str, request: PointDelete):
    with _lock:
        data = session(session_id)
        try:
            objects = predictor().remove_prompts_at(
                data["state"],
                request.indices,
            )
        except (TypeError, ValueError, IndexError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.delete("/api/sessions/{session_id}/prompts/last")
def remove_prompt(session_id: str):
    with _lock:
        data = session(session_id)
        objects = predictor().remove_prompt(data["state"])
        return result(data, objects)


@app.put("/api/sessions/{session_id}/prompts/{prompt_index}")
def update_prompt(session_id: str, prompt_index: int, prompt: PromptUpdate):
    with _lock:
        data = session(session_id)
        try:
            objects = predictor().update_prompt(
                data["state"],
                prompt_index,
                prompt.box,
            )
        except (TypeError, ValueError, IndexError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.delete("/api/sessions/{session_id}/prompts/{prompt_index}")
def delete_prompt(session_id: str, prompt_index: int):
    with _lock:
        data = session(session_id)
        try:
            objects = predictor().remove_prompt_at(data["state"], prompt_index)
        except (TypeError, ValueError, IndexError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.put("/api/sessions/{session_id}/points/{prompt_index}")
def update_point(session_id: str, prompt_index: int, prompt: PointUpdate):
    with _lock:
        data = session(session_id)
        try:
            objects = predictor().update_point(
                data["state"],
                prompt_index,
                prompt.point,
            )
        except (TypeError, ValueError, IndexError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error
        return result(data, objects)


@app.delete("/api/sessions/{session_id}/points/{prompt_index}")
def delete_point(session_id: str, prompt_index: int):
    return delete_prompt(session_id, prompt_index)
