# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from typing import Tuple
import tempfile
import os
import fsspec

TMP_DIR = None


def get_fsspec(path: str):
    def get_protocol(path: str) -> Tuple[fsspec.spec.AbstractFileSystem, str]:
        return fsspec.core.url_to_fs(path)

    if isinstance(path, str):
        return get_protocol(path)

    # unkown path type default to local
    return fsspec.filesystem("local"), path


def get_temp_dir():
    global TMP_DIR
    if TMP_DIR:
        return TMP_DIR
    TMP_DIR = tempfile.TemporaryDirectory()
    return TMP_DIR


def retrieve_local_path(path: str, worker_id):
    local_path = os.path.join("/dev/shm/", get_temp_dir().name.lstrip("/"), str(worker_id), path.lstrip("/"))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path
