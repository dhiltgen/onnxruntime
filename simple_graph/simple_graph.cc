// Hello world example trying to explore how to build up a compute graph dynamically
// an then execute it

// STL
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING

#include <Windows.h>
#ifdef OPTIONAL
#undef OPTIONAL
#endif

#pragma warning(disable : 4100)

// Needed to work around the fact that OnnxRuntime defines ERROR
#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
// Restore ERROR define
#define ERROR 0

#include <string>
#include <string.h>
#include <sstream>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>
#include <setjmp.h>
#include <algorithm>
#include <vector>
#include <functional>

#include <onnxruntime_cxx_api.h>
#include <core/graph/model.h>
#include <core/common/logging/sinks/cerr_sink.h>
#include <core/session/environment.h>
#include <core/providers/cpu/cpu_execution_provider.h>

#ifdef _WIN32
#include <atlbase.h>
#endif

// The InferenceSession class doesn't expose LoadOnnxModel as a public method
// so this hack works around that by subclassing to avoid having to serialize then
// deserialize the protobuf definition when it's already in memory.
class HackInferenceSession : public onnxruntime::InferenceSession {
  public:
  HackInferenceSession(const onnxruntime::SessionOptions& so, const onnxruntime::Environment& env) : onnxruntime::InferenceSession(so, env) {}
    onnxruntime::common::Status LoadModel(onnx::ModelProto& model_proto) {
     auto st = LoadOnnxModel(model_proto); // This hack is to access this protected method
     return st;
   }
 };

#ifdef _WIN32
int wmain(int argc, ORTCHAR_T* argv[]) {
  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  if (!SUCCEEDED(hr)) return -1;
#else
int main(int argc, ORTCHAR_T* argv[]) {
#endif
  int ret = -1;
  try {
      // Note: this code mixes the public API and internal APIs and
      // can likely be cleaned up further.
      fprintf(stderr, "Initializing ONNX Runtime...\n");
      auto g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
      if (!g_ort) {
        fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
        return -1;
      }

      OrtMemoryInfo* memory_info = NULL;
      OrtStatusPtr status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
      if (status != NULL) {
        fprintf(stderr, "ERROR: %s\n", g_ort->GetErrorMessage(status));
        return -1;
      }

      // Create a protobuf model representation in memory on the fly...
      onnx::ModelProto model_proto = onnx::ModelProto();
      onnx::GraphProto* graph = model_proto.mutable_graph();
      model_proto.set_doc_string("primitive op test");
      model_proto.set_model_version(0);
      model_proto.set_ir_version(8); // ONNX_NAMESPACE::Version::IR_VERSION
      model_proto.set_domain("ai.onnxruntime.test");
      onnx::OperatorSetIdProto* opset = model_proto.add_opset_import();
      opset->set_version(18); // ???
      graph->set_name("hello-world");

      onnx::ValueInfoProto* input = graph->add_input();
      input->set_name("input");
      onnx::TypeProto* inputType = input->mutable_type();
      onnx::TypeProto_Tensor* inputTypeTensor = inputType->mutable_tensor_type();
      inputTypeTensor->set_elem_type(onnx::TensorProto_DataType_FLOAT);
      onnx::TensorShapeProto* ish = inputTypeTensor->mutable_shape();
      onnx::TensorShapeProto_Dimension* idim = ish->add_dim();
      idim->set_dim_value(-1);
      idim->set_dim_param("batch_size");
      idim = ish->add_dim();
      idim->set_dim_value(1);

      onnx::ValueInfoProto* output = graph->add_output();
      output->set_name("output");
      onnx::TypeProto* outputType = output->mutable_type();
      onnx::TypeProto_Tensor* outputTypeTensor = outputType->mutable_tensor_type();
      outputTypeTensor->set_elem_type(onnx::TensorProto_DataType_FLOAT);
      onnx::TensorShapeProto* osh = outputTypeTensor->mutable_shape();
      onnx::TensorShapeProto_Dimension* odim = osh->add_dim();
      odim->set_dim_value(-1);
      odim->set_dim_param("batch_size");
      odim = osh->add_dim();
      odim->set_dim_value(4);

      onnx::TensorProto* tensor = graph->add_initializer();
      tensor->add_dims(4);
      tensor->add_dims(4);
      float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
      fprintf(stderr, "Graph Tensor: [");
      for (int i = 0; i < 4*4; i++) {
        // TODO is there a more efficient way?
        tensor->add_float_data(data[i]);
        fprintf(stderr, "%f,", data[i]);
      }
      fprintf(stderr, "]\n");

      tensor->set_data_type(onnx::TensorProto_DataType_FLOAT);
      tensor->set_name("tensor");

      // Add operations
      onnx::NodeProto* op = graph->add_node();
      op->set_name("mul-0");
      op->set_op_type("Mul");
      fprintf(stderr, "Tensor Op: %s\n", op->op_type().c_str());
      op->add_input("input");
      op->add_input("tensor");
      op->add_output("output");

      std::string default_logger_id{"Default"};
      onnxruntime::logging::LoggingManager mgr(
        std::make_unique<onnxruntime::logging::CErrSink>(),
        onnxruntime::logging::Severity(2), // warning
        false,
        onnxruntime::logging::LoggingManager::InstanceType::Default,
        &default_logger_id
      );
      auto log = mgr.CreateLogger("test");

      auto reg = onnxruntime::IOnnxRuntimeOpSchemaRegistryList();
      onnxruntime::Model model(
        model_proto,
        &reg,
        *log);

      onnxruntime::SessionOptions session_options{};
      std::unique_ptr<onnxruntime::Environment> env;
      onnxruntime::common::Status s = onnxruntime::Environment::Create(nullptr, env);
      if (!s.IsOK()) {
        fprintf(stderr, "ERROR: %s\n", s.ErrorMessage().c_str());
        return -1;
      }

      HackInferenceSession session(session_options, *env);
      // fprintf(stderr, "Created Session\n");
      s = session.LoadModel(model_proto);
      if (!s.IsOK()) {
        fprintf(stderr, "ERROR: %s\n", s.ErrorMessage().c_str());
        return -1;
      }

      onnxruntime::CPUExecutionProviderInfo epi;
      s = session.RegisterExecutionProvider(std::make_unique<onnxruntime::CPUExecutionProvider>(epi));
      if (!s.IsOK()) {
        fprintf(stderr, "ERROR: %s\n", s.ErrorMessage().c_str());
        return -1;
      }
      fprintf(stderr, "Registered CPU Execution Provider\n");

      // fprintf(stderr, "Loaded model_proto into session\n");
      s = session.Initialize();
      if (!s.IsOK()) {
        fprintf(stderr, "ERROR: %s\n", s.ErrorMessage().c_str());
        return -1;
      }
      // fprintf(stderr, "Initialized Session\n");
      onnxruntime::RunOptions run_options;
      onnxruntime::NameMLValMap inputs;

      float inp_data[] = {4.0};
      int64_t inp_shape[] = {1,1};
      OrtValue inp_tensor;
      fprintf(stderr, "Input Tensor: [%f]\n", inp_data[0]);

      onnxruntime::Tensor::InitOrtValue(
        onnxruntime::DataTypeImpl::GetType<float>(),
        onnxruntime::TensorShape(inp_shape, 2),
        (void*)inp_data,
        *memory_info,
        inp_tensor);
      inputs.insert({std::string("input"), inp_tensor});
      std::vector<std::string> output_names;
      output_names.push_back("output");
      std::vector<OrtValue> fetches;

      fprintf(stderr, "Running graph...\n");
      s = session.Run(run_options,
        inputs,
        output_names,
        &fetches);
      if (!s.IsOK()) {
        fprintf(stderr, "ERROR: %s\n", s.ErrorMessage().c_str());
        return -1;
      }
      fprintf(stderr, "Got %llu outputs from graph execution\n", fetches.size());
      for (OrtValue v : fetches) {
        auto& rtensor = v.Get<onnxruntime::Tensor>();
        fprintf(stderr, "Type: %s\n[", onnxruntime::DataTypeImpl::ToString(rtensor.DataType()->AsPrimitiveDataType()));
        auto output_data = rtensor.DataAsSpan<float>();
        for (float f : output_data) {
          fprintf(stderr, "%f,", f);
        }
        fprintf(stderr, "]\n");
      }

    } catch (const std::exception& ex) {
      fprintf(stderr, "%s\n", ex.what());
    }
#ifdef _WIN32
  CoUninitialize();
#endif
  return ret;
}
