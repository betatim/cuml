#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cuml.experimental.kayak.cuda_stream cimport cuda_stream as kayak_stream_t
from raft.common.handle cimport handle_t as raft_handle_t

cdef extern from "cuml/experimental/kayak/handle.hpp" namespace "kayak" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(const raft_handle_t* handle_ptr) except +
        handle_t(const raft_handle_t& handle) except +
        kayak_stream_t get_next_usable_stream() except +
