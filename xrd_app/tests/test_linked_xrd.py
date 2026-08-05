import h5py
import numpy as np

from xrd_app.core import linked_xrd


def _write_inputs(tmp_path):
    raw = tmp_path / "raw.h5"
    frames = np.zeros((4, 16, 16), dtype=np.int32)
    frames[0, 7, 6] = 10
    frames[1, 7, 7] = 20
    frames[2, 8, 7] = 30
    frames[3, 8, 8] = 40
    with h5py.File(raw, "w") as handle:
        handle.create_dataset(linked_xrd.H5_DATASET, data=frames)

    links = tmp_path / "links.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(links, "w") as handle:
        group = handle.create_group("links")
        group.create_dataset("Material", data=np.asarray(["Br"] * 4, dtype=object),
                             dtype=string_dtype)
        group.create_dataset("Point", data=np.arange(4))
        group.create_dataset("X", data=[0.0, 0.1, 1.0, 1.1])
        group.create_dataset("Y", data=[0.0, 0.1, 0.0, 0.1])
        group.create_dataset("XRF Intensity", data=[10.0, 20.0, 30.0, 40.0])
        group.create_dataset("XRD File Link", data=np.asarray([str(raw)] * 4, dtype=object),
                             dtype=string_dtype)
        group.create_dataset("XRD Frame Index", data=np.arange(4))
        group.create_dataset("Global Frame Index", data=np.arange(4))
    return links


def test_track_and_result_round_trip(tmp_path):
    links = _write_inputs(tmp_path)

    result = linked_xrd.track(
        links,
        material="Br",
        bin_width=0.5,
        peak_centers=[(6, 7)],
        detector_sum_sample=4,
        track_radius=3,
        com_radius=1,
        max_frames=4,
        log=lambda _: None,
    )

    tracking = result["tracking"].sort_values("bin_col")
    assert len(tracking) == 2
    assert tracking["n_frames"].tolist() == [2, 2]
    assert tracking.iloc[0]["com_x"] < tracking.iloc[1]["com_x"]
    assert tracking.iloc[0]["com_y"] < tracking.iloc[1]["com_y"]

    output = linked_xrd.save_result(tmp_path / "tracking.h5", result)
    loaded = linked_xrd.load_result(output)

    np.testing.assert_array_equal(loaded["peak_centers"], [[6, 7]])
    np.testing.assert_allclose(
        loaded["tracking"].sort_values("bin_col")["com_x"], tracking["com_x"]
    )
