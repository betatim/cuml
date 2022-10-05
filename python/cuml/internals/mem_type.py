#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
#


from enum import Enum, auto
from cuml.internals.device_support import GPU_ENABLED
from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import
)

cp = gpu_only_import('cupy')
np = cpu_only_import('numpy')


class MemoryTypeError(Exception):
    '''An exception thrown to indicate inconsistent memory type selection'''


class MemoryType(Enum):
    device = auto(),
    host = auto()
    managed = auto()
    mirror = auto()

    @staticmethod
    def from_str(memory_type):
        if isinstance(memory_type, str):
            memory_type = memory_type.lower()

        try:
            return MemoryType[memory_type]
        except KeyError:
            raise ValueError('Parameter memory_type must be one of "device", '
                             '"host", "managed" or "mirror"')

    def xpy(self):
        if (
            self == MemoryType.host
            or (self == MemoryType.mirror and not GPU_ENABLED)
        ):
            return np
        else:
            return cp
