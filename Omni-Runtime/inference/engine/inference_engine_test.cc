#include "Omni-Runtime/inference/engine/inference_engine.h"

#include <future>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include "gtest/gtest.h"
#include "Omni-Runtime/inference/generation/autoregressive_runner.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace {

class TestTokenizer final : public Tokenizer {
public:
  bool Init(const TokenizerModel &model) override {
    return true;
  }

  bool Encode(const std::string &text, std::vector<int32_t> *const tokens) const override {
    OMNI_RETURN_VAL_IF(tokens == nullptr, false);
    *tokens = {1, 2};
    return true;
  }

  bool Decode(const std::vector<int32_t> &tokens, std::string *const text) const override {
    OMNI_RETURN_VAL_IF(text == nullptr || tokens.size() != 1U, false);
    *text = std::to_string(tokens.front());
    return true;
  }

  const std::string &error_message() const override {
    return error_message_;
  }

private:
  std::string error_message_;
};

class TextExecutor final : public ModelExecutor {
public:
  bool Init() override {
    ClearError();
    return true;
  }

  bool Execute(const ModelRequest &request, ModelResult *const result) override {
    OMNI_RETURN_VAL_IF(result == nullptr, SetError("test output is null"));
    const auto *const generation = std::get_if<GenerationRequest>(&request);
    auto *const output = std::get_if<GenerationResult>(result);
    OMNI_RETURN_VAL_IF(generation == nullptr || output == nullptr, SetError("wrong request type"));
    if (generation->prompt == "throw") {
      throw std::runtime_error("test generation failure");
    }
    decode_steps_ = 0;
    const AutoregressiveRunner::DecodeStep step = [this](int32_t *const token,
                                                         double *const elapsed_ms) {
      OMNI_RETURN_VAL_IF(token == nullptr || elapsed_ms == nullptr, false);
      ++*token;
      ++decode_steps_;
      *elapsed_ms = 2.0;
      return true;
    };
    std::string error_message;
    OMNI_RETURN_VAL_IF(!AutoregressiveRunner::Run(*generation, 10, 100, 12, 13, tokenizer_, step,
                                                  output, &error_message),
                       SetError(error_message));
    return true;
  }

  static const ModelDescriptor &descriptor() {
    static const ModelDescriptor descriptor{"test_text", ModelTask::kTextGeneration, true};
    return descriptor;
  }

private:
  TestTokenizer tokenizer_;
  int32_t decode_steps_ = 0;
};

class ActionExecutor final : public ModelExecutor {
public:
  explicit ActionExecutor(const bool is_malformed) : is_malformed_(is_malformed) {
  }

  bool Init() override {
    return true;
  }

  bool Execute(const ModelRequest &request, ModelResult *const result) override {
    OMNI_RETURN_VAL_IF(result == nullptr, SetError("action output is null"));
    const auto *const action = std::get_if<ActionRequest>(&request);
    auto *const output = std::get_if<ActionResult>(result);
    OMNI_RETURN_VAL_IF(action == nullptr || output == nullptr, SetError("wrong action types"));
    output->horizon = action->horizon;
    output->action_size = static_cast<int32_t>(action->state.size());
    for (int32_t step = 0; step < action->horizon; ++step) {
      output->actions.insert(output->actions.end(), action->state.begin(), action->state.end());
    }
    if (is_malformed_ && !output->actions.empty()) {
      output->actions.pop_back();
    }
    return true;
  }

  static const ModelDescriptor &descriptor() {
    static const ModelDescriptor descriptor{"test_action", ModelTask::kActionPrediction, true};
    return descriptor;
  }

private:
  const bool is_malformed_;
};

template <typename TExecutor> class TestModel final : public Model {
public:
  bool Load(const std::string &config_path) override {
    ClearError();
    OMNI_RETURN_VAL_IF(config_path == "invalid", SetError("test initialization failure"));
    is_malformed_ = config_path == "malformed";
    return true;
  }

  bool CreateExecutor(std::unique_ptr<ModelExecutor> *const executor) const override {
    OMNI_RETURN_VAL_IF_LOG(executor == nullptr, false, ERROR, "Test executor output is null");
    if constexpr (std::is_same_v<TExecutor, ActionExecutor>) {
      *executor = std::make_unique<ActionExecutor>(is_malformed_);
    } else {
      *executor = std::make_unique<TExecutor>();
    }
    return true;
  }

  const ModelDescriptor &descriptor() const override {
    return TExecutor::descriptor();
  }

  const std::string &compute_backend() const override {
    return backend_;
  }

private:
  bool is_malformed_ = false;
  std::string backend_ = "test";
};

GenerationRequest Request() {
  GenerationRequest request;
  request.prompt = "Hello";
  request.max_new_tokens = 8;
  return request;
}

class LifecycleModel final : public Model {
public:
  explicit LifecycleModel(const std::shared_ptr<std::vector<std::string>> &events)
      : events_(events) {
  }

  ~LifecycleModel() override {
    events_->push_back("destroy model");
  }

  bool Load(const std::string &config_path) override {
    events_->push_back("load");
    ClearError();
    config_ = config_path;
    OMNI_RETURN_VAL_IF(config_path == "invalid", SetError("load failed"));
    return true;
  }

  bool CreateExecutor(std::unique_ptr<ModelExecutor> *const executor) const override {
    OMNI_RETURN_VAL_IF_LOG(executor == nullptr, false, ERROR, "Executor output is null");
    OMNI_RETURN_VAL_IF(config_ == "factory_failure", false);
    if (config_ == "null_executor") {
      executor->reset();
      return true;
    }
    *executor = std::make_unique<Executor>(events_, config_ != "executor_failure");
    return true;
  }

  const ModelDescriptor &descriptor() const override {
    return TextExecutor::descriptor();
  }

  const std::string &compute_backend() const override {
    return config_;
  }

private:
  class Executor final : public ModelExecutor {
  public:
    explicit Executor(const std::shared_ptr<std::vector<std::string>> &events, const bool is_valid)
        : events_(events), is_valid_(is_valid) {
    }

    ~Executor() override {
      events_->push_back("destroy executor");
    }

    bool Init() override {
      OMNI_RETURN_VAL_IF(!is_valid_, SetError("executor initialization failed"));
      return true;
    }

    bool Execute(const ModelRequest &request, ModelResult *const result) override {
      OMNI_RETURN_VAL_IF(result == nullptr, SetError("result is null"));
      *result = GenerationResult{};
      return true;
    }

  private:
    std::shared_ptr<std::vector<std::string>> events_;
    const bool is_valid_;
  };

  std::shared_ptr<std::vector<std::string>> events_;
  std::string config_;
};

TEST(InferenceEngineTest, ReleasesExecutorBeforeReloadAndModelAndPropagatesFactoryFailures) {
  const auto events = std::make_shared<std::vector<std::string>>();
  {
    InferenceEngine engine(std::make_unique<LifecycleModel>(events));
    ASSERT_TRUE(engine.Init("valid"));
    EXPECT_FALSE(engine.Init("invalid"));
    EXPECT_EQ(engine.error_message(), "load failed");
    EXPECT_EQ(*events, (std::vector<std::string>{"load", "destroy executor", "load"}));
    EXPECT_FALSE(engine.Init("factory_failure"));
    EXPECT_EQ(engine.error_message(), "model could not create an executor");
    EXPECT_FALSE(engine.Init("null_executor"));
    EXPECT_EQ(engine.error_message(), "model returned a null executor");
    EXPECT_FALSE(engine.Init("executor_failure"));
    EXPECT_EQ(engine.error_message(), "executor initialization failed");
    GenerationResult result;
    EXPECT_FALSE(engine.Generate(Request(), &result));
    ASSERT_TRUE(engine.Init("valid"));
    ASSERT_TRUE(engine.Generate(Request(), &result));
  }
  ASSERT_GE(events->size(), 2U);
  EXPECT_EQ((*events)[events->size() - 2U], "destroy executor");
  EXPECT_EQ(events->back(), "destroy model");
}

TEST(InferenceEngineTest, RetriesInitializationAndCountsOnlyExecutedDecodeSteps) {
  InferenceEngine executor(std::make_unique<TestModel<TextExecutor>>());
  GenerationResult result;
  auto request = Request();
  EXPECT_FALSE(executor.Generate(request, &result));
  EXPECT_FALSE(executor.Init("invalid"));
  ASSERT_TRUE(executor.Init("valid"));
  ASSERT_TRUE(executor.Generate(request, &result));
  EXPECT_EQ(result.generated_token_ids, (std::vector<int32_t>{10, 11, 12}));
  EXPECT_EQ(result.text, "1011");
  EXPECT_EQ(result.decode_token_count, 2);
  EXPECT_EQ(result.decode_time_ms, 4.0);
  request.max_new_tokens = 1;
  ASSERT_TRUE(executor.Generate(request, &result));
  EXPECT_EQ(result.generated_token_ids, (std::vector<int32_t>{10}));
  EXPECT_EQ(result.decode_token_count, 0);
  EXPECT_FALSE(executor.Init("invalid"));
  EXPECT_FALSE(executor.Generate(request, &result));
}

TEST(InferenceEngineTest, RejectsReentryWithoutCorruptingActiveGenerationAndRecovers) {
  InferenceEngine executor(std::make_unique<TestModel<TextExecutor>>());
  ASSERT_TRUE(executor.Init("valid"));
  GenerationResult result;
  auto request = Request();
  request.stream_callback = [&executor, &request](const std::string &text) {
    GenerationResult nested;
    EXPECT_FALSE(executor.Generate(request, &nested));
    EXPECT_FALSE(executor.Init("valid"));
    return false;
  };
  ASSERT_TRUE(executor.Generate(request, &result));
  EXPECT_TRUE(result.is_cancelled);
  EXPECT_TRUE(executor.error_message().empty());
  EXPECT_EQ(result.decode_token_count, 0);
  request.stream_callback = [](const std::string &text) -> bool {
    throw std::runtime_error("test callback failure");
  };
  EXPECT_FALSE(executor.Generate(request, &result));
  EXPECT_NE(executor.error_message().find("test callback failure"), std::string::npos);
  request.stream_callback = {};
  ASSERT_TRUE(executor.Generate(request, &result));
  EXPECT_EQ(result.generated_token_ids, (std::vector<int32_t>{10, 11, 12}));
  EXPECT_TRUE(executor.error_message().empty());
}

TEST(InferenceEngineTest, RejectsConcurrentExecutionWithoutMutatingActiveState) {
  InferenceEngine executor(std::make_unique<TestModel<TextExecutor>>());
  ASSERT_TRUE(executor.Init("valid"));
  std::promise<void> entered;
  std::promise<void> release;
  auto released = release.get_future();
  auto request = Request();
  request.stream_callback = [&entered, &released](const std::string &text) {
    entered.set_value();
    released.wait();
    return false;
  };
  GenerationResult result;
  bool is_successful = false;
  std::thread worker([&executor, &request, &result, &is_successful]() {
    is_successful = executor.Generate(request, &result);
  });
  entered.get_future().wait();
  GenerationResult nested;
  EXPECT_FALSE(executor.Generate(Request(), &nested));
  EXPECT_FALSE(executor.Init("valid"));
  release.set_value();
  worker.join();
  EXPECT_TRUE(is_successful);
  EXPECT_TRUE(executor.error_message().empty());
  EXPECT_EQ(result.generated_token_ids, (std::vector<int32_t>{10}));
}

TEST(InferenceEngineTest, RejectsNullOutputsInvalidRequestsAndNullModels) {
  InferenceEngine empty(nullptr);
  EXPECT_FALSE(empty.Init("valid"));
  EXPECT_FALSE(empty.error_message().empty());
  InferenceEngine executor(std::make_unique<TestModel<TextExecutor>>());
  ASSERT_TRUE(executor.Init("valid"));
  GenerationResult result;
  auto request = Request();
  EXPECT_FALSE(executor.Generate(request, nullptr));
  EXPECT_FALSE(executor.error_message().empty());
  request.prompt.clear();
  EXPECT_FALSE(executor.Generate(request, &result));
  request = Request();
  request.max_new_tokens = 0;
  EXPECT_FALSE(executor.Generate(request, &result));
  request = Request();
  request.image_path = "/nonexistent/omni_runtime_image.jpeg";
  EXPECT_FALSE(executor.Generate(request, &result));
  ModelResult output;
  EXPECT_FALSE(executor.Execute(ActionRequest{}, &output));
}

TEST(InferenceEngineTest, ReleasesExecutionGuardDuringExceptionUnwinding) {
  InferenceEngine executor(std::make_unique<TestModel<TextExecutor>>());
  ASSERT_TRUE(executor.Init("valid"));
  GenerationResult result;
  auto request = Request();
  request.prompt = "throw";
  EXPECT_THROW(executor.Generate(request, &result), std::runtime_error);
  ASSERT_TRUE(executor.Init("valid"));
  ASSERT_TRUE(executor.Generate(Request(), &result));
}

TEST(InferenceEngineTest, RunsNonTokenActionsWithoutTokenizerOrGenerationHooks) {
  InferenceEngine executor(std::make_unique<TestModel<ActionExecutor>>());
  ASSERT_TRUE(executor.Init("valid"));
  ModelResult output;
  ActionRequest request;
  request.state = {1.0F, 2.0F};
  request.horizon = 3;
  ASSERT_TRUE(executor.Execute(request, &output));
  const auto &action = std::get<ActionResult>(output);
  EXPECT_EQ(action.horizon, 3);
  EXPECT_EQ(action.action_size, 2);
  EXPECT_EQ(action.actions, (std::vector<float>{1, 2, 1, 2, 1, 2}));
  ASSERT_TRUE(executor.Init("malformed"));
  EXPECT_FALSE(executor.Execute(request, &output));
  EXPECT_NE(executor.error_message().find("action block shape"), std::string::npos);
  ASSERT_TRUE(executor.Init("valid"));
  GenerationResult generation;
  EXPECT_FALSE(executor.Generate(Request(), &generation));
  request.horizon = 0;
  EXPECT_FALSE(executor.Execute(request, &output));
  request.horizon = 1;
  request.observations.push_back(nullptr);
  EXPECT_FALSE(executor.Execute(request, &output));
}

TEST(AutoregressiveRunnerTest, RejectsInvalidTokensAndReportsDecodeFailures) {
  const TestTokenizer tokenizer;
  GenerationResult result;
  std::string error_message;
  const AutoregressiveRunner::DecodeStep step = [](int32_t *const token, double *const elapsed_ms) {
    return false;
  };
  EXPECT_FALSE(AutoregressiveRunner::Run(Request(), -1, 100, 12, 13, tokenizer, step, &result,
                                         &error_message));
  EXPECT_NE(error_message.find("vocabulary"), std::string::npos);
  result = GenerationResult{};
  EXPECT_FALSE(AutoregressiveRunner::Run(Request(), 10, 100, 12, 13, tokenizer, step, &result,
                                         &error_message));
  EXPECT_EQ(error_message, "decode step failed");
  EXPECT_EQ(result.decode_token_count, 0);
  EXPECT_FALSE(AutoregressiveRunner::Run(Request(), 10, 100, 12, 13, tokenizer, step, nullptr,
                                         &error_message));
}

} // namespace
} // namespace inference
} // namespace omni_runtime
