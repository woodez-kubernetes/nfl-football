"""Stage 3: Ollama client. Fully mocked — no host required."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from nfl_football import config, llm

SCHEMA = {
    "type": "object",
    "properties": {
        "pick": {"type": "string", "enum": ["A", "B"]},
        "confidence": {"type": "string"},
    },
    "required": ["pick", "confidence"],
}


class FakeResponse:
    def __init__(self, content: str, prompt_tokens=10, eval_tokens=20):
        self._content = content
        self._prompt = prompt_tokens
        self._eval = eval_tokens

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "message": {"content": self._content},
            "prompt_eval_count": self._prompt,
            "eval_count": self._eval,
        }


def fake_post(*responses):
    """Return a side_effect cycling through the given reply contents."""
    it = iter(responses)

    def _post(url, json=None, timeout=None):
        return FakeResponse(next(it))

    return _post


class TestExtractJson(unittest.TestCase):
    def test_clean_object(self):
        self.assertEqual(llm._extract_json('{"a": 1}'), {"a": 1})

    def test_trailing_whitespace(self):
        """Observed on this host: schema mode appends stray tabs."""
        self.assertEqual(llm._extract_json('{"a": 1}\n\t\t  '), {"a": 1})

    def test_embedded_in_prose(self):
        self.assertEqual(llm._extract_json('Sure!\n{"a": 1}\nHope that helps'), {"a": 1})

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            llm._extract_json("   ")

    def test_non_object_rejected(self):
        with self.assertRaises(ValueError):
            llm._extract_json("[1, 2, 3]")

    def test_malformed_rejected(self):
        with self.assertRaises(ValueError):
            llm._extract_json('{"a": ')

    def test_no_json_at_all_rejected(self):
        with self.assertRaises(ValueError):
            llm._extract_json("I cannot help with that")


class TestRequireKeys(unittest.TestCase):
    def test_passes_when_present(self):
        llm._require_keys({"pick": "A", "confidence": "high"}, SCHEMA)

    def test_raises_and_names_missing(self):
        with self.assertRaises(ValueError) as ctx:
            llm._require_keys({"pick": "A"}, SCHEMA)
        self.assertIn("confidence", str(ctx.exception))


class TestChatJson(unittest.TestCase):
    def setUp(self):
        llm.STATS = llm.Stats()

    def test_success_first_attempt(self):
        with patch("requests.post", side_effect=fake_post('{"pick":"A","confidence":"high"}')):
            result = llm.chat_json("go", SCHEMA)
        self.assertEqual(result.data["pick"], "A")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(llm.STATS.calls, 1)
        self.assertEqual(llm.STATS.retries, 0)

    def test_sends_schema_and_options(self):
        captured = {}

        def _post(url, json=None, timeout=None):
            captured.update(json)
            return FakeResponse('{"pick":"A","confidence":"high"}')

        with patch("requests.post", side_effect=_post):
            llm.chat_json("go", SCHEMA)

        self.assertEqual(captured["format"], SCHEMA)
        self.assertEqual(captured["options"]["num_ctx"], config.OLLAMA_NUM_CTX)
        self.assertEqual(captured["options"]["num_predict"], config.OLLAMA_NUM_PREDICT)
        self.assertFalse(captured["stream"])

    def test_system_prompt_included(self):
        captured = {}

        def _post(url, json=None, timeout=None):
            captured.update(json)
            return FakeResponse('{"pick":"A","confidence":"high"}')

        with patch("requests.post", side_effect=_post):
            llm.chat_json("go", SCHEMA, system="be brief")

        self.assertEqual(captured["messages"][0]["role"], "system")
        self.assertEqual(captured["messages"][0]["content"], "be brief")

    def test_retries_after_malformed_json(self):
        with patch("requests.post",
                   side_effect=fake_post("not json", '{"pick":"B","confidence":"low"}')):
            result = llm.chat_json("go", SCHEMA)
        self.assertEqual(result.data["pick"], "B")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(llm.STATS.retries, 1)

    def test_retries_after_missing_key(self):
        with patch("requests.post",
                   side_effect=fake_post('{"pick":"A"}', '{"pick":"A","confidence":"high"}')):
            result = llm.chat_json("go", SCHEMA)
        self.assertEqual(result.attempts, 2)

    def test_validator_rejection_triggers_retry(self):
        seen = []

        def validator(data):
            seen.append(data["confidence"])
            if data["confidence"] != "low":
                raise ValueError("confidence must be low")

        with patch("requests.post",
                   side_effect=fake_post('{"pick":"A","confidence":"high"}',
                                         '{"pick":"A","confidence":"low"}')):
            result = llm.chat_json("go", SCHEMA, validator=validator)
        self.assertEqual(result.data["confidence"], "low")
        self.assertEqual(seen, ["high", "low"])

    def test_correction_message_is_fed_back(self):
        sent = []

        def _post(url, json=None, timeout=None):
            sent.append(list(json["messages"]))
            content = ("not json" if len(sent) == 1
                       else '{"pick":"A","confidence":"high"}')
            return FakeResponse(content)

        with patch("requests.post", side_effect=_post):
            llm.chat_json("go", SCHEMA)

        # Second attempt carries the original prompt, the bad reply, and a correction.
        self.assertEqual(len(sent[1]), 3)
        self.assertEqual(sent[1][1]["role"], "assistant")
        self.assertIn("rejected", sent[1][2]["content"])

    def test_raises_after_retry_budget(self):
        with patch("requests.post", side_effect=fake_post("bad", "worse", "worst")):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.chat_json("go", SCHEMA)
        self.assertIn("no valid response", str(ctx.exception))
        self.assertEqual(llm.STATS.failures, 1)
        self.assertEqual(llm.STATS.calls, 0)

    def test_network_error_raises_immediately(self):
        with patch("requests.post", side_effect=requests.ConnectionError("refused")):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.chat_json("go", SCHEMA)
        self.assertIn("Ollama request failed", str(ctx.exception))

    def test_stats_accumulate_tokens(self):
        with patch("requests.post", side_effect=fake_post('{"pick":"A","confidence":"h"}',
                                                          '{"pick":"B","confidence":"l"}')):
            llm.chat_json("one", SCHEMA)
            llm.chat_json("two", SCHEMA)
        self.assertEqual(llm.STATS.calls, 2)
        self.assertEqual(llm.STATS.prompt_tokens, 20)
        self.assertEqual(llm.STATS.eval_tokens, 40)
        self.assertIn("2 LLM calls", llm.STATS.summary())


class TestHealthCheck(unittest.TestCase):
    def test_ok_when_model_present(self):
        class R:
            def raise_for_status(self): return None
            def json(self): return {"models": [{"name": "qwen2.5:7b"}]}

        with patch("requests.get", return_value=R()):
            ok, msg = llm.health_check("qwen2.5:7b")
        self.assertTrue(ok)
        self.assertIn("ready", msg)

    def test_fails_when_model_absent(self):
        class R:
            def raise_for_status(self): return None
            def json(self): return {"models": [{"name": "llama3:8b"}]}

        with patch("requests.get", return_value=R()):
            ok, msg = llm.health_check("qwen2.5:7b")
        self.assertFalse(ok)
        self.assertIn("not on host", msg)

    def test_fails_when_unreachable(self):
        with patch("requests.get", side_effect=requests.ConnectionError("no route")):
            ok, msg = llm.health_check()
        self.assertFalse(ok)
        self.assertIn("cannot reach", msg)


if __name__ == "__main__":
    unittest.main()
