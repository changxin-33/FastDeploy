// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*! \file runtime_option.h
    \brief A brief file description.
    More details
 */

#pragma once

#include <algorithm>
#include <map>
#include <vector>
#include "fastdeploy/runtime/enum_variables.h"
#include "fastdeploy/runtime/backends/ort/option.h"

namespace fastdeploy {

/*! @brief Option object used when create a new Runtime object
 */
struct FASTDEPLOY_DECL RuntimeOption {
  /** \brief Set path of model file and parameter file
   *
   * \param[in] model_path Path of model file, e.g ResNet50/model.pdmodel for Paddle format model / ResNet50/model.onnx for ONNX format model
   * \param[in] params_path Path of parameter file, this only used when the model format is Paddle, e.g Resnet50/model.pdiparams
   * \param[in] format Format of the loaded model
   */
  void SetModelPath(const std::string& model_path,
                    const std::string& params_path = "",
                    const ModelFormat& format = ModelFormat::PADDLE);

  /** \brief Specify the memory buffer of model and parameter. Used when model and params are loaded directly from memory
   *
   * \param[in] model_buffer The string of model memory buffer
   * \param[in] params_buffer The string of parameters memory buffer
   * \param[in] format Format of the loaded model
   */
  void SetModelBuffer(const std::string& model_buffer,
                      const std::string& params_buffer = "",
                      const ModelFormat& format = ModelFormat::PADDLE);

  /** \brief When loading encrypted model, encryption_key is required to decrypte model
   *
   * \param[in] encryption_key The key for decrypting model
   */
  void SetEncryptionKey(const std::string& encryption_key);

  /// Use cpu to inference, the runtime will inference on CPU by default
  void UseCpu();
  /// Use Nvidia GPU to inference
  void UseGpu(int gpu_id = 0);
  
  void SetExternalStream(void* external_stream);

  /*
   * @brief Set number of cpu threads while inference on CPU, by default it will decided by the different backends
   */
  void SetCpuThreadNum(int thread_num);
  /// Set ONNX Runtime as inference backend, support CPU/GPU
  void UseOrtBackend();
  
  /// Option to configure ONNX Runtime backend
  OrtBackendOption ort_option;
  
  // \brief Enable to check if current backend set by
  //        user can be found at valid_xxx_backend.
  //
  void EnableValidBackendCheck() {
    enable_valid_backend_check = true;
  }
  // \brief Disable to check if current backend set by
  //        user can be found at valid_xxx_backend.
  //
  void DisableValidBackendCheck() {
    enable_valid_backend_check = false;
  }

  // enable the check for valid backend, default true.
  bool enable_valid_backend_check = true;

  // If model_from_memory is true, the model_file and params_file is
  // binary stream in memory;
  // Otherwise, the model_file and params_file means the path of file
  std::string model_file = "";
  std::string params_file = "";
  bool model_from_memory_ = false;
  // format of input model
  ModelFormat model_format = ModelFormat::PADDLE;

  std::string encryption_key_ = "";

  // for cpu inference
  // default will let the backend choose their own default value
  int cpu_thread_num = -1;
  int device_id = 0;
  Backend backend = Backend::UNKNOWN;

  Device device = Device::CPU;

  void* external_stream_ = nullptr;

  bool enable_pinned_memory = false;

  void SetOrtGraphOptLevel(int level = -1);
};

}  // namespace fastdeploy
