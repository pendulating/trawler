"""Tests for ``classify_response_line`` (dagspaces/common/batch_api.py).

The smoke-run inspection (2026-04-27) showed that 470/493 PrivacyLens
helpfulness rows received HTTP 404 from the live judge server, but the
finalize stage parsed those JSON-encoded error strings as if they were
real responses (helpfulness_score=0, helpfulness_judged=True). That
silent corruption is exactly what classify_response_line exists to
prevent.

Coverage:

- success line (real assistant content) → ok=True, content populated
- top-level ``error`` (sidecar exhausted retries) → ok=False, kind=judge_api_error
- response with status_code >= 400 → ok=False, kind=http_error
- response 200 but no choices → ok=False, kind=empty_choices
"""

from __future__ import annotations

from dagspaces.common.batch_api import classify_response_line, extract_content


class TestClassifyResponseLine:
    def test_success(self):
        line = {
            "custom_id": "x:1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {"message": {"content": "Answer: Yes."}}
                    ]
                },
            },
        }
        c = classify_response_line(line)
        assert c["ok"] is True
        assert c["content"] == "Answer: Yes."
        assert c["error_kind"] is None

    def test_sidecar_error(self):
        line = {
            "custom_id": "x:2",
            "error": "[other] HTTP 404: The model 'gpt-5.2' does not exist",
        }
        c = classify_response_line(line)
        assert c["ok"] is False
        assert c["error_kind"] == "judge_api_error"
        assert "404" in c["error_preview"]
        assert c["content"] == ""

    def test_http_error_with_status(self):
        line = {
            "custom_id": "x:3",
            "response": {
                "status_code": 500,
                "body": {"error": {"message": "internal"}},
            },
        }
        c = classify_response_line(line)
        assert c["ok"] is False
        assert c["error_kind"] == "http_error"
        assert "HTTP 500" in c["error_preview"]

    def test_empty_choices(self):
        line = {
            "custom_id": "x:4",
            "response": {
                "status_code": 200,
                "body": {"choices": []},
            },
        }
        c = classify_response_line(line)
        assert c["ok"] is False
        assert c["error_kind"] == "empty_choices"

    def test_extract_content_back_compat(self):
        """``extract_content`` still works the way the old script did."""
        success = {
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": "hi"}}]},
            }
        }
        assert extract_content(success) == "hi"

        err = {"error": "boom"}
        # Returns a JSON-encoded error string (legacy behavior).
        assert "error" in extract_content(err)
