#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#=============================================================================

set(CUML_MIN_VERSION_raft "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00")

if(NOT DEFINED CUML_RAFT_VERSION)
  set(CUML_RAFT_VERSION "${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}")
endif()

if(NOT DEFINED CUML_RAFT_BRANCH)
  set(CUML_RAFT_BRANCH "branch-${CUML_RAFT_VERSION}")
endif()

if(NOT DEFINED CUML_RAFT_REPOSITORY)
  set(CUML_RAFT_REPOSITORY "https://github.com/rapidsai/raft.git")
endif()

function(find_and_configure_raft)
    set(oneValueArgs VERSION REPO PINNED_TAG EXCLUDE_FROM_ALL USE_RAFT_DIST USE_RAFT_NN USE_RAFT_STATIC USE_FAISS_STATIC CLONE_ON_PIN NVTX)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUML_RAFT_VERSION}")
      message(STATUS "CUML: RAFT pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
      set(CPM_DOWNLOAD_raft ON)
    elseif(PKG_USE_RAFT_STATIC AND (NOT CPM_raft_SOURCE))
      message(STATUS "CUML: Cloning raft locally to build static libraries.")
      set(CPM_DOWNLOAD_raft ON)
    endif()

    if(PKG_USE_RAFT_DIST)
      string(APPEND RAFT_COMPONENTS "distance")
    endif()

    if(PKG_USE_RAFT_NN)
      string(APPEND RAFT_COMPONENTS " nn")
    endif()

    if(PKG_USE_RAFT_DIST AND PKG_USE_RAFT_NN)
      set(RAFT_COMPILE_LIBRARIES ON)
    else()
      set(RAFT_COMPILE_LIBRARIES OFF)
    endif()

    # We need to set this each time so that on subsequent calls to cmake
    # the raft-config.cmake re-evaluates the RAFT_NVTX value
    set(RAFT_NVTX ${PKG_NVTX})

    set(RAFT_BUILD_SHARED_LIBS ON)
    if(PKG_USE_RAFT_STATIC)
      set(RAFT_BUILD_SHARED_LIBS OFF)
    endif()

    message(VERBOSE "CUML: raft FIND_PACKAGE_ARGUMENTS COMPONENTS ${RAFT_COMPONENTS}")

    rapids_cpm_find(raft ${PKG_VERSION}
      GLOBAL_TARGETS      raft::raft
      BUILD_EXPORT_SET    cuml-exports
      INSTALL_EXPORT_SET  cuml-exports
      CPM_ARGS
        GIT_REPOSITORY         ${PKG_REPO}
        GIT_TAG                ${PKG_PINNED_TAG}
        SOURCE_SUBDIR          cpp
        EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
        FIND_PACKAGE_ARGUMENTS "COMPONENTS ${RAFT_COMPONENTS}"
        OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_SHARED_LIBS ${RAFT_BUILD_SHARED_LIBS}"
          "RAFT_COMPILE_LIBRARIES ${RAFT_COMPILE_LIBRARIES}"
          "RAFT_COMPILE_NN_LIBRARY ${PKG_USE_RAFT_NN}"
          "RAFT_COMPILE_DIST_LIBRARY ${PKG_USE_RAFT_DIST}"
          "RAFT_USE_FAISS_STATIC ${PKG_USE_FAISS_STATIC}"
    )

    if(raft_ADDED)
        message(VERBOSE "CUML: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "CUML: Using RAFT located in ${raft_DIR}")
    endif()


endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION          ${CUML_MIN_VERSION_raft}
                        REPO             ${CUML_RAFT_REPOSITORY}
                        PINNED_TAG       ${CUML_RAFT_BRANCH}
                        EXCLUDE_FROM_ALL ${CUML_EXCLUDE_RAFT_FROM_ALL}

                        # When PINNED_TAG above doesn't match cuml,
                        # force local raft clone in build directory
                        # even if it's already installed.
                        CLONE_ON_PIN     ON
                        USE_RAFT_NN      ${CUML_USE_RAFT_NN}
                        USE_RAFT_DIST    ${CUML_USE_RAFT_DIST}
                        USE_RAFT_STATIC  ${CUML_USE_RAFT_STATIC}
                        USE_FAISS_STATIC ${CUML_USE_FAISS_STATIC}
                        NVTX             ${NVTX}
                        )
