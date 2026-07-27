from io import BytesIO

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from src.api import app as api
from starlette.datastructures import Headers


class FakePredictor:
    def __init__(self):
        self.refine_calls = 0

    def start(self, image):
        return {"boxes": [], "box_labels": [], "points": [], "size": image.size}

    def add_prompt(self, state, box, positive=True):
        state["boxes"].append(box)
        state["box_labels"].append(int(positive))
        state["points"].append(None)
        return self.add_prompt_result()

    def add_point(self, state, point, positive=True):
        state["boxes"].append([0, 0, 2, 2])
        state["box_labels"].append(int(positive))
        state["points"].append(point)
        return self.add_prompt_result()

    def update_point(self, state, index, point):
        state["points"][index] = point
        return self.add_prompt_result()

    @staticmethod
    def add_prompt_result():
        return [
            {
                "object_id": 1,
                "box": (1, 1, 3, 3),
                "roi": np.ones((2, 2), dtype=bool),
                "metrics": {"score": 0.9, "similarity": 0.8},
            }
        ]

    def remove_prompt(self, state):
        if state["boxes"]:
            state["boxes"].pop()
            state["box_labels"].pop()
            state["points"].pop()
        return []

    def update_prompt(self, state, index, box):
        state["boxes"][index] = box
        return self.add_prompt_result()

    def remove_prompt_at(self, state, index):
        state["boxes"].pop(index)
        state["box_labels"].pop(index)
        state["points"].pop(index)
        return []

    def remove_prompts_at(self, state, indices):
        for index in sorted(set(indices), reverse=True):
            state["boxes"].pop(index)
            state["box_labels"].pop(index)
            state["points"].pop(index)
        return self.add_prompt_result() if state["points"] else []

    def refine_objects(self, state, objects):
        assert state["boxes"]
        self.refine_calls += 1
        return [
            {
                **item,
                "metrics": {**item["metrics"], "refine_score": 0.95},
            }
            for item in objects
        ]


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 6), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def setup(monkeypatch):
    monkeypatch.setattr(api, "_predictor", FakePredictor())
    monkeypatch.setattr(api, "_sessions", {})


def upload(data, content_type):
    return UploadFile(
        BytesIO(data),
        filename="image.png",
        headers=Headers({"content-type": content_type}),
    )


def test_session_prompt_and_undo(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))

    session_id = created["session_id"]
    assert created["width"] == 8
    assert created["height"] == 6

    prompted = api.add_prompt(
        session_id,
        api.Prompt(box=(1, 1, 3, 3), positive=True),
    )
    assert prompted["prompt_count"] == 1
    assert prompted["objects"][0]["mask"].startswith("data:image/png;base64,")

    undone = api.remove_prompt(session_id)
    assert undone["prompt_count"] == 0


def test_session_updates_and_deletes_selected_prompt(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    api.add_prompt(session_id, api.Prompt(box=(1, 1, 3, 3), positive=True))
    api.add_prompt(session_id, api.Prompt(box=(4, 1, 6, 3), positive=True))

    updated = api.update_prompt(
        session_id,
        0,
        api.PromptUpdate(box=(0, 0, 2, 2)),
    )
    deleted = api.delete_prompt(session_id, 0)

    assert updated["prompt_count"] == 2
    assert deleted["prompt_count"] == 1


def test_session_adds_moves_and_deletes_point_prompt(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]

    added = api.add_point(
        session_id,
        api.PointPrompt(point=(3, 2), positive=True),
    )
    updated = api.update_point(
        session_id,
        0,
        api.PointUpdate(point=(5, 4)),
    )
    deleted = api.delete_point(session_id, 0)

    assert added["prompt_count"] == 1
    assert updated["prompt_count"] == 1
    assert deleted["prompt_count"] == 0


def test_session_clicks_result_to_add_exclude_box(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    api.add_prompt(session_id, api.Prompt(box=(0, 0, 4, 4), positive=True))

    excluded = api.exclude_object(session_id, 1)

    assert excluded["prompt_count"] == 2
    assert api._sessions[session_id]["state"]["box_labels"] == [1, 0]
    assert api._sessions[session_id]["state"]["boxes"][-1] == (1, 1, 3, 3)


def test_session_refines_current_results(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    api.add_prompt(session_id, api.Prompt(box=(0, 0, 4, 4), positive=True))

    refined = api.refine_results(session_id)
    repeated = api.refine_results(session_id)

    assert refined["objects"][0]["metrics"]["refine_score"] == pytest.approx(0.95)
    assert repeated == refined
    assert api._predictor.refine_calls == 2


def test_session_deletes_multiple_selected_points(monkeypatch):
    setup(monkeypatch)
    created = api.create_session(upload(image_bytes(), "image/png"))
    session_id = created["session_id"]
    for point in ((1, 1), (3, 2), (5, 4)):
        api.add_point(session_id, api.PointPrompt(point=point, positive=True))

    deleted = api.delete_points(
        session_id,
        api.PointDelete(indices=[0, 2]),
    )

    assert deleted["prompt_count"] == 1


def test_uploading_another_image_keeps_existing_session(monkeypatch):
    setup(monkeypatch)
    first = api.create_session(upload(image_bytes(), "image/png"))
    api.create_session(upload(image_bytes(), "image/png"))

    prompted = api.add_point(
        first["session_id"],
        api.PointPrompt(point=(3, 2), positive=True),
    )

    assert prompted["prompt_count"] == 1


def test_session_rejects_non_image_upload(monkeypatch):
    setup(monkeypatch)

    with pytest.raises(HTTPException) as error:
        api.create_session(upload(b"hello", "text/plain"))

    assert error.value.status_code == 415
