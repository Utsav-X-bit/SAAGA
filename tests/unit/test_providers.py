"""Unit tests for SAAGA LLM Providers.

Verifies:
- Dataclasses (Message, LLMResponse)
- Helper functions (normalize_messages, messages_to_prompt)
- BaseLLMProvider and MockLLMProvider contracts
- OpenAIProvider with mocked HTTP requests and batch execution
- OllamaProvider with mocked HTTP requests and parameter translation
- VLLMProvider lazy imports and mocked engine interactions
- HFProvider lazy imports and mocked model interactions
- Registry detection and provider instantiation
"""
from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

from saaga.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    MockLLMProvider,
    messages_to_prompt,
    normalize_messages,
)
from saaga.providers.hf_provider import HFProvider
from saaga.providers.ollama_provider import OllamaProvider
from saaga.providers.openai_provider import OpenAIProvider
from saaga.providers.registry import (
    detect_provider_type,
    get_provider,
    list_providers,
    register_provider,
)
from saaga.providers.vllm_provider import VLLMProvider


class TestDataStructures(unittest.TestCase):
    def test_message_dataclass(self):
        m = Message(role="user", content="hello")
        self.assertEqual(m.role, "user")
        self.assertEqual(m.content, "hello")
        self.assertEqual(m.to_dict(), {"role": "user", "content": "hello"})

        m2 = Message.from_dict({"role": "assistant", "content": "hi there"})
        self.assertEqual(m2.role, "assistant")
        self.assertEqual(m2.content, "hi there")

    def test_llm_response_dataclass(self):
        resp = LLMResponse(
            text="generated answer",
            raw={"id": "123"},
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )
        self.assertEqual(resp.text, "generated answer")
        self.assertEqual(resp.prompt_tokens, 10)
        self.assertEqual(resp.completion_tokens, 5)
        self.assertEqual(resp.total_tokens, 15)
        self.assertEqual(resp.finish_reason, "stop")

    def test_normalize_messages(self):
        raw = [
            {"role": "system", "content": "sys prompt"},
            Message(role="user", content="user query"),
        ]
        norm = normalize_messages(raw)
        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0], {"role": "system", "content": "sys prompt"})
        self.assertEqual(norm[1], {"role": "user", "content": "user query"})

        with self.assertRaises(TypeError):
            normalize_messages(["invalid string"])  # type: ignore

    def test_messages_to_prompt(self):
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="What is 2+2?"),
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("System: Be helpful.", prompt)
        self.assertIn("User: What is 2+2?", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))


class TestMockLLMProvider(unittest.TestCase):
    def test_mock_chat_and_generate(self):
        mock = MockLLMProvider(
            responses=["First answer", "Second answer"],
            default_response="Default answer",
        )

        resp1 = mock.chat([Message("user", "Q1")])
        self.assertEqual(resp1.text, "First answer")

        resp2 = mock.generate("Q2")
        self.assertEqual(resp2.text, "Second answer")

        resp3 = mock.generate("Q3")
        self.assertEqual(resp3.text, "Default answer")

        self.assertEqual(len(mock.history), 3)

    def test_mock_batch(self):
        mock = MockLLMProvider(default_response="batch item")
        batch_prompts = ["p1", "p2", "p3"]
        responses = mock.generate_batch(batch_prompts)
        self.assertEqual(len(responses), 3)
        for r in responses:
            self.assertEqual(r.text, "batch item")

    def test_mock_callback(self):
        def echo_callback(messages, kwargs):
            return f"Echo: {messages[-1]['content']}"

        mock = MockLLMProvider(callback=echo_callback)
        resp = mock.chat([{"role": "user", "content": "testing 123"}])
        self.assertEqual(resp.text, "Echo: testing 123")


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.provider = OpenAIProvider(
            model_id="gpt-3.5-mock",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            timeout=10.0,
            max_retries=2,
        )

    def test_url_normalization(self):
        p1 = OpenAIProvider("model", api_base="http://localhost:8000")
        self.assertEqual(p1.api_base, "http://localhost:8000/v1")

        p2 = OpenAIProvider("model", api_base="http://localhost:8000/v1/")
        self.assertEqual(p2.api_base, "http://localhost:8000/v1")

    @patch("requests.Session.post")
    def test_chat_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Paris"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }
        mock_post.return_value = mock_response

        resp = self.provider.chat([{"role": "user", "content": "Capital of France?"}])
        self.assertEqual(resp.text, "Paris")
        self.assertEqual(resp.prompt_tokens, 8)
        self.assertEqual(resp.completion_tokens, 2)
        self.assertEqual(resp.finish_reason, "stop")

    @patch("requests.Session.post")
    def test_generate_and_batch(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        mock_post.return_value = mock_response

        resps = self.provider.generate_batch(["p1", "p2"])
        self.assertEqual(len(resps), 2)
        self.assertEqual(resps[0].text, "ok")
        self.assertEqual(resps[1].text, "ok")

    @patch("requests.Session.post")
    def test_client_error_raises_immediately(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        with self.assertRaises(RuntimeError) as ctx:
            self.provider.chat([{"role": "user", "content": "hi"}])
        self.assertIn("status 401", str(ctx.exception))
        # Should fail immediately without exhausting max_retries
        self.assertEqual(mock_post.call_count, 1)

    @patch("requests.Session.post")
    def test_retry_on_transient_error(self, mock_post):
        # First call fails with 503, second succeeds with 200
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        fail_resp.text = "Service Unavailable"

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "choices": [{"message": {"content": "Recovered"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_post.side_effect = [fail_resp, success_resp]

        resp = self.provider.chat([{"role": "user", "content": "test"}])
        self.assertEqual(resp.text, "Recovered")
        self.assertEqual(mock_post.call_count, 2)


class TestOllamaProvider(unittest.TestCase):
    def setUp(self):
        self.provider = OllamaProvider(
            model_id="llama3",
            api_base="http://localhost:11434",
            timeout=10.0,
        )

    def test_url_normalization(self):
        p = OllamaProvider("llama3", api_base="http://localhost:11434/api/")
        self.assertEqual(p.api_base, "http://localhost:11434")

    @patch("requests.Session.post")
    def test_chat_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Ollama reply"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 4,
        }
        mock_post.return_value = mock_resp

        resp = self.provider.chat([{"role": "user", "content": "hi"}], temperature=0.5)
        self.assertEqual(resp.text, "Ollama reply")
        self.assertEqual(resp.prompt_tokens, 12)
        self.assertEqual(resp.completion_tokens, 4)

        # Verify options were mapped properly
        call_payload = mock_post.call_args[1]["json"]
        self.assertEqual(call_payload["model"], "llama3")
        self.assertEqual(call_payload["options"]["temperature"], 0.5)
    @patch("requests.Session.post")
    def test_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "Generated text",
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_post.return_value = mock_resp

        resp = self.provider.generate("Prompt here", max_tokens=100)
        self.assertEqual(resp.text, "Generated text")
        call_payload = mock_post.call_args[1]["json"]
        self.assertEqual(call_payload["options"]["num_predict"], 100)

    @patch("requests.Session.post")
    def test_ollama_error_raises_runtime_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "model 'not-found' not found"
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            self.provider.generate("Hello")
        self.assertIn("status 404", str(ctx.exception))


class TestVLLMProvider(unittest.TestCase):
    def test_lazy_import_error_when_vllm_missing(self):
        # When vLLM cannot be imported, initializing without an existing llm should raise ImportError
        with patch.dict("sys.modules", {"vllm": None}):
            with self.assertRaises(ImportError):
                VLLMProvider("meta-llama/Llama-3-8B")

    def test_vllm_with_mock_engine(self):
        # Mock pre-instantiated vllm.LLM engine
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="vLLM generated text", finish_reason="stop", token_ids=[1, 2, 3])]
        mock_output.prompt_token_ids = [10, 20]
        mock_llm.generate.return_value = [mock_output]

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<s>Formatted chat</s>"
        mock_llm.get_tokenizer.return_value = mock_tokenizer

        # Mock SamplingParams class in vllm
        fake_vllm = types.ModuleType("vllm")
        fake_sampling = MagicMock()
        fake_vllm.SamplingParams = fake_sampling
        fake_vllm.LLM = MagicMock()

        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            provider = VLLMProvider(
                model_id="test-model",
                llm=mock_llm,
                tokenizer=mock_tokenizer,
            )

            # Test generate
            resp = provider.generate("Test prompt")
            self.assertEqual(resp.text, "vLLM generated text")
            self.assertEqual(resp.prompt_tokens, 2)
            self.assertEqual(resp.completion_tokens, 3)

            # Test chat
            chat_resp = provider.chat([{"role": "user", "content": "hi"}])
            self.assertEqual(chat_resp.text, "vLLM generated text")

    def test_vllm_lora_dispatch(self):
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="LoRA output", finish_reason="stop", token_ids=[1])]
        mock_output.prompt_token_ids = [1]
        mock_llm.generate.return_value = [mock_output]

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = MagicMock()
        fake_vllm.LLM = MagicMock()
        fake_lora_req_cls = MagicMock()

        fake_lora_mod = types.ModuleType("vllm.lora.request")
        fake_lora_mod.LoRARequest = fake_lora_req_cls

        with patch.dict("sys.modules", {"vllm": fake_vllm, "vllm.lora.request": fake_lora_mod}):
            provider = VLLMProvider(
                model_id="base-model",
                llm=mock_llm,
                tokenizer=MagicMock(),
            )
            # Register adapter
            mock_req = MagicMock()
            fake_lora_req_cls.return_value = mock_req
            provider.register_lora("planner", "/tmp/planner_ckpt", lora_id=1)

            # Generate with lora_name="planner"
            provider.generate("plan query", lora_name="planner")

            # Ensure lora_request was passed to llm.generate
            call_kwargs = mock_llm.generate.call_args[1]
            self.assertEqual(call_kwargs.get("lora_request"), mock_req)


class TestHFProvider(unittest.TestCase):
    def test_lazy_import_error_when_transformers_missing(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(ImportError):
                HFProvider("meta-llama/Llama-3-8B")

    def test_hf_with_mock_model(self):
        # Mock torch and transformers modules
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = MagicMock()
        fake_torch.cuda = MagicMock(is_available=MagicMock(return_value=False))

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.device = "cpu"

        # Mock generate returning dummy 2D list-like tensor
        mock_tensor = MagicMock()
        # Slice outputs[0, input_seq_len:]
        mock_tensor.__getitem__.return_value = [201, 202]
        mock_model.generate.return_value = mock_tensor

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        # Tokenizer returns dict with input_ids of shape (1, 2)
        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 2)
        mock_attn_mask = MagicMock()
        mock_attn_mask.to.return_value = mock_attn_mask
        mock_attn_mask.__getitem__.return_value.sum.return_value.item.return_value = 2
        mock_input_ids.to.return_value = mock_input_ids
        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attn_mask,
        }
        mock_tokenizer.decode.return_value = "HF generated text"

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = MagicMock()
        fake_transformers.AutoTokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            provider = HFProvider(
                model_id="test-model",
                model=mock_model,
                tokenizer=mock_tokenizer,
            )

            resp = provider.generate("Test prompt")
            self.assertEqual(resp.text, "HF generated text")
            self.assertEqual(resp.prompt_tokens, 2)
            self.assertEqual(resp.completion_tokens, 2)
    def test_hf_lora_registration(self):
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = MagicMock()
        fake_transformers.AutoTokenizer = MagicMock()

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            provider = HFProvider(
                model_id="test-model",
                model=MagicMock(),
                tokenizer=MagicMock(),
            )
            provider.register_lora("planner", "/tmp/planner_adapter")
            self.assertIn("planner", provider._lora_registry)
            self.assertTrue(provider._lora_registry["planner"].endswith("/tmp/planner_adapter"))

class TestRegistry(unittest.TestCase):
    def test_list_providers(self):
        providers = list_providers()
        self.assertIn("openai", providers)
        self.assertIn("ollama", providers)
        self.assertIn("vllm", providers)
        self.assertIn("hf", providers)
        self.assertIn("mock", providers)

    def test_detect_provider_type(self):
        # Explicit type
        self.assertEqual(detect_provider_type("openai"), "openai")
        self.assertEqual(detect_provider_type("vllm"), "vllm")
        self.assertEqual(detect_provider_type("ollama"), "ollama")

        # URL detection
        self.assertEqual(detect_provider_type(api_base="http://localhost:11434"), "ollama")
        self.assertEqual(detect_provider_type(api_base="http://localhost:8000/v1"), "openai")

        # Model ID prefix detection
        self.assertEqual(detect_provider_type(model_id="mock-model"), "mock")
        self.assertEqual(detect_provider_type(model_id="ollama/llama3"), "ollama")
        self.assertEqual(detect_provider_type(model_id="openai/gpt-4"), "openai")

    def test_get_provider_instantiation(self):
        # Mock provider via registry
        p_mock = get_provider("mock", model_id="test-mock", default_response="reg test")
        self.assertIsInstance(p_mock, MockLLMProvider)
        resp = p_mock.generate("hi")
        self.assertEqual(resp.text, "reg test")

        # OpenAI provider via registry
        p_openai = get_provider(
            "openai",
            model_id="gpt-4",
            api_base="http://localhost:8000/v1",
        )
        self.assertIsInstance(p_openai, OpenAIProvider)
        self.assertEqual(p_openai.model_id, "gpt-4")

        # Ollama provider via registry
        p_ollama = get_provider(
            api_base="http://localhost:11434",
            model_id="llama3",
        )
        self.assertIsInstance(p_ollama, OllamaProvider)
        self.assertEqual(p_ollama.model_id, "llama3")

    def test_custom_provider_registration(self):
        class CustomProvider(BaseLLMProvider):
            def chat(self, messages, **kwargs):
                return LLMResponse(text="custom response")

        register_provider("custom-backend", CustomProvider)
        self.assertIn("custom-backend", list_providers())

        inst = get_provider("custom-backend", model_id="my-custom-model")
        self.assertIsInstance(inst, CustomProvider)
        self.assertEqual(inst.chat([]).text, "custom response")


if __name__ == "__main__":
    unittest.main()
