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

#include "fastdeploy/runtime/runtime.h"
#include "fastdeploy/utils/unique_ptr.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

void RuntimeOption::SetModelPath(const std::string& model_path,
                                 const std::string& params_path,
                                 const ModelFormat& format) {
  model_file = model_path;
  params_file = params_path;
  model_format = format;
  model_from_memory_ = false;
}

void RuntimeOption::SetModelBuffer(const char * model_buffer,
                                   size_t model_buffer_size,
                                   const char * params_buffer,
                                   size_t params_buffer_size,
                                   const ModelFormat& format) {
  model_buffer_size_ = model_buffer_size;
  params_buffer_size_ = params_buffer_size;
  model_from_memory_ = true;
  if (format == ModelFormat::PADDLE) {
    model_buffer_ = std::string(model_buffer, model_buffer + model_buffer_size);
    params_buffer_ = std::string(params_buffer, params_buffer + params_buffer_size);
    model_format = ModelFormat::PADDLE;
  } else if (format == ModelFormat::ONNX) {
    model_buffer_ = std::string(model_buffer, model_buffer + model_buffer_size);
    model_format = ModelFormat::ONNX;
  } else if (format == ModelFormat::TORCHSCRIPT) {
    model_buffer_ = std::string(model_buffer, model_buffer + model_buffer_size);
    model_format = ModelFormat::TORCHSCRIPT;
  } else {
    FDASSERT(false,
             "The model format only can be "
             "ModelFormat::PADDLE/ModelFormat::ONNX/ModelFormat::TORCHSCRIPT.");
  }
}

void RuntimeOption::SetEncryptionKey(const std::string& encryption_key) {
#ifdef ENABLE_ENCRYPTION
  encryption_key_ = encryption_key;
#else
  FDERROR << "The FastDeploy didn't compile with encryption function."
          << std::endl;
#endif
}

void RuntimeOption::UseGpu(int gpu_id) {
#if defined(WITH_GPU) || defined(WITH_OPENCL)
  device = Device::GPU;
  device_id = gpu_id;

#if defined(WITH_OPENCL) && defined(ENABLE_LITE_BACKEND)
  paddle_lite_option.device = device;
#endif

#else
  FDWARNING << "The FastDeploy didn't compile with GPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::UseCpu() { device = Device::CPU; }

void RuntimeOption::SetExternalStream(void* external_stream) {
  external_stream_ = external_stream;
}

void RuntimeOption::SetCpuThreadNum(int thread_num) {
  FDASSERT(thread_num > 0, "The thread_num must be greater than 0.");
  cpu_thread_num = thread_num;
  ort_option.intra_op_num_threads = thread_num;
}

void RuntimeOption::SetOrtGraphOptLevel(int level) {
  FDWARNING << "`RuntimeOption::SetOrtGraphOptLevel` will be removed in "
               "v1.2.0, please modify its member variables directly, e.g "
               "`runtime_option.ort_option.graph_optimization_level = 99`."
            << std::endl;
  std::vector<int> supported_level{-1, 0, 1, 2};
  auto valid_level = std::find(supported_level.begin(), supported_level.end(),
                               level) != supported_level.end();
  FDASSERT(valid_level, "The level must be -1, 0, 1, 2.");
  ort_option.graph_optimization_level = level;
}

// use onnxruntime backend
void RuntimeOption::UseOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  backend = Backend::ORT;
#else
  FDASSERT(false, "The FastDeploy didn't compile with OrtBackend.");
#endif
}

}  // namespace fastdeploy
