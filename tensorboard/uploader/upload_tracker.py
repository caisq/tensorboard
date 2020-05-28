# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Progress tracker for uploader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import sys

import tqdm


def readable_time_string():
    """Get a human-readable time string for the present."""
    return f"{datetime.now():%Y-%m-%dT%H:%M:%S%z}"


def readable_bytes_string(bytes):
    """Get a human-readable string for number of bytes."""
    if bytes >= 2 ** 20:
        return "%.2f MB" % (float(bytes) / 2 ** 20)
    elif bytes >= 2 ** 10:
        return "%.2f kB" % (float(bytes) / 2 ** 10)
    else:
        return "%d B" % bytes


class UploadStats(object):
    """Statistics of uploading."""

    def __init__(self):
        self._num_scalars = 0
        self._num_tensors = 0
        self._num_tensors_skipped = 0
        self._tensor_bytes = 0
        self._tensor_bytes_skipped = 0
        self._num_blobs = 0
        self._num_blobs_skipped = 0
        self._blob_bytes = 0
        self._blob_bytes_skipped = 0
        self._plugin_names = set()

    def add_scalars(self, num_scalars):
        """Add a batch of scalars.

        Args:
          num_scalars: Number of scalars uploaded in this batch.
        """
        self._num_scalars += num_scalars

    def add_tensors(
        self,
        num_tensors,
        num_tensors_skipped,
        tensor_bytes,
        tensor_bytes_skipped,
    ):
        """Add a batch of tensors.

        Args:
          num_tensors: Number of tensors encountered in this batch, including
            the ones skipped due to reasons such as large exceeding limit.
          num_tensors: Number of tensors skipped. This describes a subset of
            `num_tensors` and hence must be `<= num_tensors`.
          tensor_bytes: Total byte size of tensors encountered in this batch,
            including the skipped ones.
          tensor_bytes_skipped: Total byte size of the tensors skipped due to
            reasons such as size exceeding limit.
        """
        assert num_tensors_skipped <= num_tensors
        assert tensor_bytes_skipped <= tensor_bytes
        self._num_tensors += num_tensors
        self._num_tensors_skipped += num_tensors_skipped
        self._tensor_bytes += tensor_bytes
        self._tensor_bytes_skipped = tensor_bytes_skipped

    def add_blob(self, blob_bytes, is_skipped):
        """Add a blob.

        Args:
          blob_bytes: Byte size of the blob.
          is_skipped: Whether the uploading of the blob is skipped due to
            reasons such as size exceeding limit.
        """
        self._num_blobs += 1
        self._blob_bytes += blob_bytes
        if is_skipped:
            self._num_blobs_skipped += 1
            self._blob_bytes_skipped += blob_bytes

    def accumulate(self, stats):
        """Accumulate another UploadStats instance into this instance.

        Args:
          stats: Another UploadStats instance to add to this instance.
        """
        self._num_scalars += stats.num_scalars
        self._num_tensors += stats.num_tensors
        self._num_tensors_skipped += stats.num_tensors_skipped
        self._tensor_bytes += stats.tensor_bytes
        self._tensor_bytes_skipped += stats.tensor_bytes_skipped
        self._num_blobs += stats.num_blobs
        self._blob_bytes += stats.blob_bytes
        self._blob_bytes_skipped += stats.blob_bytes_skipped

    @property
    def num_scalars(self):
        return self._num_scalars

    @property
    def num_tensors(self):
        return self._num_tensors

    @property
    def num_tensors_skipped(self):
        return self._num_tensors_skipped

    @property
    def tensor_bytes(self):
        return self._tensor_bytes

    @property
    def tensor_bytes_skipped(self):
        return self._tensor_bytes_skipped

    @property
    def num_blobs(self):
        return self._num_blobs

    @property
    def num_blobs_skipped(self):
        return self._num_blobs_skipped

    @property
    def blob_bytes(self):
        return self._blob_bytes

    @property
    def blob_bytes_skipped(self):
        return self._blob_bytes_skipped


class UploadTracker(object):
    """Tracker for uploader progress and status."""

    def __init__(self):
        self._cumulative_num_scalars = 0
        self._cumulative_num_tensors = 0
        self._cumulative_num_tensors_skipped = 0
        self._cumulative_tensor_bytes = 0
        self._cumulative_tensor_bytes_skipped = 0
        self._cumulative_num_blobs = 0
        self._cumulative_num_blobs_skipped = 0
        self._cumulative_blob_bytes = 0
        self._cumulative_blob_bytes_skipped = 0
        self._cumulative_plugin_names = set()
        self._dot_counter = 0

    def _dummy_generator(self):
        while True:
            # Yield an arbitrary value 0: The progress bar is indefinite.
            yield 0

    def send_start(self):
        self._num_scalars = 0
        self._num_tensors = 0
        self._num_tensors_skipped = 0
        self._tensor_bytes = 0
        self._tensor_bytes_skipped = 0
        self._num_blobs = 0
        self._num_blobs_skipped = 0
        self._blob_bytes = 0
        self._blob_bytes_skipped = 0
        self._plugin_names = set()
        self._progress_bar = None

    def _update_status(self, message):
        if message:
            self._dot_counter += 1
            message += "." * (self._dot_counter % 3 + 1)
        if not self._progress_bar:
            self._progress_bar = tqdm.tqdm(
                self._dummy_generator(), bar_format="{desc}"
            )
        self._progress_bar.set_description_str(message)
        self._progress_bar.update()

    def send_done(self):
        self._cumulative_num_scalars += self._num_scalars
        self._cumulative_num_tensors += self._num_tensors
        self._cumulative_num_tensors_skipped += self._num_tensors_skipped
        self._cumulative_tensor_bytes += self._tensor_bytes
        self._cumulative_tensor_bytes_skipped += self._tensor_bytes_skipped
        self._cumulative_num_blobs += self._num_blobs
        self._cumulative_num_blobs_skipped += self._num_blobs_skipped
        self._cumulative_blob_bytes += self._blob_bytes
        self._cumulative_blob_bytes_skipped += self._blob_bytes_skipped
        self._cumulative_plugin_names.update(self._plugin_names)
        if self._num_scalars or self._num_tensors or self._num_blobs:
            if self._progress_bar:
                self._update_status("")
                self._progress_bar.close()
            # TODO(cais): Only populate the existing data types.
            # TODO(cais0): Print skipped bytes if non-zero.
            sys.stdout.write(
                "[%s] Uploaded %d scalars, %d tensors (%s), %d binary objects (%s)\n"
                "    Plugins: %s\n"
                "    Cumulative: %d scalars, %d tensors (%s), %d binary objects (%s)\n"
                % (
                    readable_time_string(),
                    self._num_scalars,
                    self._num_tensors,
                    readable_bytes_string(self._tensor_bytes),
                    self._num_blobs,
                    readable_bytes_string(
                        self._blob_bytes - self._blob_bytes_skipped
                    ),
                    ", ".join(self._plugin_names),
                    self._cumulative_num_scalars,
                    self._cumulative_num_tensors,
                    readable_bytes_string(self._cumulative_tensor_bytes),
                    self._cumulative_num_blobs,
                    readable_bytes_string(
                        self._cumulative_blob_bytes
                        - self._cumulative_blob_bytes_skipped
                    ),
                )
            )
            sys.stdout.flush()

    def add_plugin_name(self, plugin_name):
        self._plugin_names.add(plugin_name)

    def scalars_start(self, num_scalars):
        if not num_scalars:
            return
        self._num_scalars += num_scalars
        self._update_status("Uploading %d scalars" % num_scalars)

    def scalars_done(self):
        pass

    def tensors_start(
        self, num_tensors, num_tensors_skipped, bytes, bytes_skipped
    ):
        if not num_tensors:
            return
        self._num_tensors += num_tensors
        self._tensor_bytes += bytes
        self._num_tensors_skipped += num_tensors_skipped
        self._tensor_bytes_skipped = bytes_skipped
        # TODO(cais): Populate the "skipped" part if and only if
        # num_tensors_skipped > 0.
        self._update_status(
            "Uploading %d tensors (%s) (Skipped %d tensors, %s)"
            % (
                num_tensors,
                readable_bytes_string(bytes),
                num_tensors_skipped,
                readable_bytes_string(bytes_skipped),
            )
        )

    def tensors_done(self):
        pass

    def blob_tracker(self, blob_bytes):
        return BlobTracker(self, blob_bytes)

    # def blob_start(self, blob_bytes):
    #     pass
    # #     self._num_blobs += 1
    # #     self._blob_bytes += blob_bytes
    # #     self._update_status(
    # #         "Uploading binary object (%s)" % readable_bytes_string(blob_bytes)
    # #     )

    # def blob_done(self, is_uploaded):
    #     pass
    # #     if not is_uploaded:
    # #         self._num_blobs_uploaded += 1
    # #         self._blob_bytes_uploaded += blob_bytes_uploaded


class BlobTracker(object):
    def __init__(self, upload_tracker, blob_bytes):
        self._upload_tracker = upload_tracker
        self._blob_bytes = blob_bytes

    def __enter__(self):
        self._upload_tracker._update_status(
            "Uploading binary object (%s)"
            % readable_bytes_string(self._blob_bytes)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb  # Unuse.
        pass

    def mark_uploaded(self, is_uploaded):
        pass
