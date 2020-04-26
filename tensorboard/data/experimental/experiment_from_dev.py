# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Experiment Data Access API for tensorboard.dev."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import time

import grpc
import numpy as np
import pandas  # TODO(cais): Guard import with user-friendly error message.

from tensorboard.data.experimental import base_experiment
from tensorboard.uploader import auth
from tensorboard.uploader import util
from tensorboard.uploader import server_info as server_info_lib
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader.proto import export_service_pb2_grpc
from tensorboard.util import grpc_util


class ExperimentFromDev(base_experiment.BaseExperiment):
    def __init__(self, experiment_id):
        super(ExperimentFromDev, self).__init__()
        store = auth.CredentialsStore()
        credentials = store.read_credentials()
        print("credentials = %s" % credentials)  # DEBUG
        channel_options = None
        # if flags.grpc_creds_type == "local":
        #     channel_creds = grpc.local_channel_credentials()
        # elif flags.grpc_creds_type == "ssl":
        channel_creds = grpc.ssl_channel_credentials()
        # elif flags.grpc_creds_type == "ssl_dev":
        #     # Configure the dev cert to use by passing the environment variable
        #     # GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=path/to/cert.crt
        #     channel_creds = grpc.ssl_channel_credentials()
        #     channel_options = [("grpc.ssl_target_name_override", "localhost")]
        # else:
        #     msg = "Invalid --grpc_creds_type %s" % flags.grpc_creds_type
        #     raise base_plugin.FlagsError(msg)
        server_info = _get_server_info()
        print("server_info = %s" % server_info)  # DEBUG
        composite_channel_creds = grpc.composite_channel_credentials(
            channel_creds, auth.id_token_call_credentials(credentials)
        )
        channel = grpc.secure_channel(
            server_info.api_server.endpoint,
            composite_channel_creds,
            options=channel_options,
        )
        print("channel = %s" % channel)  # DEBUG
        self._api_client = export_service_pb2_grpc.TensorBoardExporterServiceStub(
            channel
        )
        print("self._api_client = %s" % self._api_client)  # DEBUG
        self._experiment_id = experiment_id

    def get_scalars(self, runs_filter=None, tags_filter=None, pivot=True):
        request = export_service_pb2.StreamExperimentDataRequest()
        request.experiment_id = self._experiment_id
        read_time = time.time()
        util.set_timestamp(request.read_timestamp, read_time)
        stream = self._api_client.StreamExperimentData(
            request, metadata=grpc_util.version_metadata()
        )
        runs = []
        tags = []
        steps = []
        wall_times = []
        values = []
        for response in stream:
            metadata = base64.b64encode(
                response.tag_metadata.SerializeToString()
            ).decode("ascii")
            print(
                "run = %s, tag = %s, metadata = %s"
                % (response.run_name, response.tag_name, metadata)
            )  # DEBUG
            num_values = len(response.points.values)
            runs.extend([response.run_name] * num_values)
            tags.extend([response.tag_name] * num_values)
            steps.extend(list(response.points.steps))
            wall_times.extend(
                [t.ToNanoseconds() / 1e9 for t in response.points.wall_times]
            )
            values.extend(list(response.points.values))
        data_frame = pandas.DataFrame(
            {
                "run": runs,
                "tag": tags,
                "step": steps,
                "wall_time": wall_times,
                "value": values,
            }
        )
        if pivot:
            data_frame = data_frame.pivot_table(
                ["value", "wall_time"], ["run", "step"], "tag", aggfunc=np.stack
            )
        return data_frame


def _get_server_info():
    origin = "https://tensorboard.dev"
    # TODO(cais): Support flags --origin and --api_endpoint.
    plugins = ["scalars"]
    server_info = server_info_lib.fetch_server_info(origin, plugins)
    return server_info


def _handle_server_info(info):
    compat = info.compatibility
    if compat.verdict == server_info_pb2.VERDICT_WARN:
        sys.stderr.write("Warning [from server]: %s\n" % compat.details)
        sys.stderr.flush()
    elif compat.verdict == server_info_pb2.VERDICT_ERROR:
        raise ValueError("Error [from server]: %s" % compat.details)
    else:
        # OK or unknown; assume OK.
        if compat.details:
            sys.stderr.write("%s\n" % compat.details)
            sys.stderr.flush()
