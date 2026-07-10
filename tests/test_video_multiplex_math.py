import torch

from src.ml.components.video.tracker.multiplex.state import MultiplexController


def make_state(count=3):
    controller = MultiplexController(4, eval_multiplex_count=4).eval()
    return controller.get_state(
        count,
        torch.device("cpu"),
        torch.float32,
        random=False,
        object_ids=list(range(10, 10 + count)),
    )


def test_demux_is_left_inverse_of_mux():
    state = make_state()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    torch.testing.assert_close(state.demux(state.mux(values)), values)


def test_removal_preserves_remaining_row_order():
    state = make_state()
    state.remove_objects([1])
    values = torch.tensor([[1.0], [3.0]])

    assert state.object_ids == [10, 12]
    torch.testing.assert_close(state.demux(state.mux(values)), values)
