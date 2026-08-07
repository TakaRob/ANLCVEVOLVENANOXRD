"""Isolated on-demand raw detector reads for interactive GUI use."""

from __future__ import annotations

import multiprocessing
import queue
from pathlib import Path

import h5py
import numpy as np

from .io import H5_DATASET


def _worker(requests, responses, xrd_files, frame_map):
    handles = {}
    try:
        while True:
            request = requests.get()
            if request is None:
                return
            generation, key, indices = request
            try:
                summed = None
                for global_index in indices:
                    file_index, local_index = frame_map[int(global_index)]
                    file_index = int(file_index)
                    handle = handles.get(file_index)
                    if handle is None:
                        handle = h5py.File(xrd_files[file_index], "r")
                        handles[file_index] = handle
                    frame = handle[H5_DATASET][int(local_index)].astype(np.float32)
                    summed = frame if summed is None else summed + frame
                if summed is not None:
                    np.clip(summed, 0, 1e9, out=summed)
                responses.put((generation, key, summed, None))
            except Exception as error:
                for handle in handles.values():
                    try:
                        handle.close()
                    except Exception:
                        pass
                handles.clear()
                responses.put((generation, key, None, f"{type(error).__name__}: {error}"))
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass


class RawImageBroker:
    """Read raw HDF5 bins in a child process without blocking Qt/X11."""

    def __init__(self, grid_mapping):
        self.bins = dict(grid_mapping.get("bins") or {})
        self.xrd_files = [str(Path(path)) for path in grid_mapping.get("xrd_files", [])]
        self.frame_map = [tuple(pair) for pair in grid_mapping.get("frame_map", [])]
        context = multiprocessing.get_context("spawn")
        self._requests = context.Queue(maxsize=4)
        self._responses = context.Queue(maxsize=4)
        self._pending = set()
        self._process = context.Process(
            target=_worker,
            args=(self._requests, self._responses, self.xrd_files, self.frame_map),
            daemon=True,
            name="xrd-raw-image-broker",
        )
        self._process.start()

    def keys(self):
        return list(self.bins)

    def request(self, generation, key):
        token = (int(generation), str(key))
        if token in self._pending or key not in self.bins or not self._process.is_alive():
            return False
        try:
            self._requests.put_nowait((token[0], token[1], self.bins[key]))
        except queue.Full:
            return False
        self._pending.add(token)
        return True

    def poll(self):
        responses = []
        while True:
            try:
                response = self._responses.get_nowait()
            except queue.Empty:
                break
            self._pending.discard((int(response[0]), str(response[1])))
            responses.append(response)
        return responses

    def close(self):
        if self._process is None:
            return
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._process.join(timeout=0.2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._requests.close()
        self._responses.close()
        self._process = None
