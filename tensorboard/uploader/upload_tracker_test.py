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
"""Tests for tensorboard.uploader.upload_tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import mock

from tensorboard.uploader import upload_tracker
from tensorboard import test as tb_test


class UploadStatsTest(tb_test.TestCase):
    """Unit tests for the UploadStats class."""

    def testAddScalar(self):
        stats = upload_tracker.UploadStats()
        stats.add_scalars(1234)
        self.assertEqual(stats.num_scalars, 1234)
        stats.add_scalars(4321)
        self.assertEqual(stats.num_scalars, 5555)

    def testAddTensor(self):
        stats = upload_tracker.UploadStats()
        stats.add_tensors(
            num_tensors=10,
            num_tensors_skipped=0,
            tensor_bytes=1000,
            tensor_bytes_skipped=0,
        )
        self.assertEqual(stats.num_tensors, 10)
        self.assertEqual(stats.num_tensors_skipped, 0)
        self.assertEqual(stats.tensor_bytes, 1000)
        self.assertEqual(stats.tensor_bytes_skipped, 0)
        stats.add_tensors(
            num_tensors=20,
            num_tensors_skipped=5,
            tensor_bytes=2000,
            tensor_bytes_skipped=500,
        )
        self.assertEqual(stats.num_tensors, 30)
        self.assertEqual(stats.num_tensors_skipped, 5)
        self.assertEqual(stats.tensor_bytes, 3000)
        self.assertEqual(stats.tensor_bytes_skipped, 500)

    def testAddBlob(self):
        stats = upload_tracker.UploadStats()
        stats.add_blob(blob_bytes=1000, is_skipped=False)
        self.assertEqual(stats.blob_bytes, 1000)
        self.assertEqual(stats.blob_bytes_skipped, 0)
        stats.add_blob(blob_bytes=2000, is_skipped=True)
        self.assertEqual(stats.blob_bytes, 3000)
        self.assertEqual(stats.blob_bytes_skipped, 2000)

    def testAccumulate(self):
        stats_1 = upload_tracker.UploadStats()
        stats_1.add_scalars(10)
        stats_1.add_tensors(
            num_tensors=100,
            num_tensors_skipped=50,
            tensor_bytes=1000,
            tensor_bytes_skipped=500,
        )
        stats_1.add_blob(blob_bytes=5, is_skipped=False)
        stats_1.add_blob(blob_bytes=500, is_skipped=True)
        stats_2 = upload_tracker.UploadStats()
        stats_2.add_scalars(20)
        stats_2.add_tensors(
            num_tensors=200,
            num_tensors_skipped=1,
            tensor_bytes=2000,
            tensor_bytes_skipped=1000,
        )
        stats_2.add_blob(blob_bytes=1, is_skipped=False)
        stats_2.add_blob(blob_bytes=1000, is_skipped=True)
        stats_1.accumulate(stats_2)
        # Check `stats_1`, which should have the sums of the two objects.
        self.assertEqual(stats_1.num_scalars, 30)
        self.assertEqual(stats_1.num_tensors, 300)
        self.assertEqual(stats_1.num_tensors_skipped, 51)
        self.assertEqual(stats_1.tensor_bytes, 3000)
        self.assertEqual(stats_1.tensor_bytes_skipped, 1500)
        self.assertEqual(stats_1.num_blobs, 4)
        self.assertEqual(stats_1.blob_bytes, 1506)
        self.assertEqual(stats_1.blob_bytes_skipped, 1500)
        # Check that `stats_2` isn't mutated.
        self.assertEqual(stats_2.num_scalars, 20)
        self.assertEqual(stats_2.num_tensors, 200)
        self.assertEqual(stats_2.num_tensors_skipped, 1)
        self.assertEqual(stats_2.tensor_bytes, 2000)
        self.assertEqual(stats_2.tensor_bytes_skipped, 1000)
        self.assertEqual(stats_2.num_blobs, 2)
        self.assertEqual(stats_2.blob_bytes, 1001)
        self.assertEqual(stats_2.blob_bytes_skipped, 1000)


if __name__ == "__main__":
    tb_test.main()
