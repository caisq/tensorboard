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


if __name__ == "__main__":
    tb_test.main()
