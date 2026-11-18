"""Gemini Decision 1 client for the Decisions API.

POST https://generativelanguage.googleapis.com/v1beta/decisions
    x-goog-api-key: <GEMINI_API_KEY>
    {"model": "gemini-decision-1", "input": ..., "questions": {qid: {type: "noul", instructions: ...}}}

Answers come back as {"answers": {qid: {"noul": p, "confidence": c}}} where noul
is the probability that the answer is yes, clamped to [0.01, 0.99] by the API.
Input is billed once per request no matter how many questions ride along; each
question adds only its own instruction tokens. Up to 1,000 questions per call.

This is the concrete scorer backend behind the JevClient interface (Jev was
the working name for the batched decision engine in the plan). Computed
questions are resolved locally and never sent - exact answers, zero tokens.

Stdlib only (urllib), transport injectable so tests run with no network.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

from sdt.jev_client import DEFAULT_RESOLVERS, JevClient
from sdt.schemas import Question, QuestionType, SdtError, ScoreVector

DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
DEFAULT_MODEL = "gemini-decision-1"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

Transport = Callable[[str, Dict[str, str], bytes], Tuple[int, bytes]]


def _urllib_transport(url: str, headers: Dict[str, str], body: bytes) -> Tuple[int, bytes]:
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        # network blip: map to a retryable status instead of crashing the run
        return 503, ("network error: " + str(e)).encode("utf-8")


class GeminiDecisionClient(JevClient):
    scorer_name = "gemini-decision-1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        transport: Optional[Transport] = None,
        max_retries: int = 3,
        resolvers: Optional[Dict] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("GEMINI_DECISION_MODEL", DEFAULT_MODEL)
        self.endpoint = endpoint or os.environ.get("GEMINI_DECISIONS_ENDPOINT", DEFAULT_ENDPOINT)
        self.transport = transport or _urllib_transport
        self.max_retries = max_retries
        self.resolvers = dict(DEFAULT_RESOLVERS)
        self.resolvers.update(resolvers or {})
        self.last_response: Optional[Dict] = None

    def build_request(self, text: str, context: Dict, questions: List[Question]) -> Dict:
        semantic = [q for q in questions if q.qtype is QuestionType.SEMANTIC]
        return {
            "model": self.model,
            # input is an array of typed content blocks (Interactions API format);
            # each block carries its own "type" - confirmed by the API's own
            # 400 message: "The 'type' parameter is required at 'input[0]'".
            "input": [
                {"type": "text", "text": "Customer ticket:" + chr(10) + text},
                {"type": "text", "text": "Account and order context JSON:" + chr(10) + json.dumps(context, ensure_ascii=False)},
            ],
            "questions": {q.qid: self._serialize_question(q) for q in semantic},
        }

    def _serialize_question(self, q: Question) -> Dict:
        dt = getattr(q, "decision_type", None)
        if dt is not None and dt.value == "choice":
            return {"type": "choice", "instructions": q.text, "criteria": dict(q.options or {})}
        if dt is not None and dt.value == "score":
            return {"type": "score", "instructions": q.text, "criteria": list(q.rubric or [])}
        if q.criteria:
            return {"type": "noul", "instructions": q.text, "criteria": dict(q.criteria)}
        return {"type": "noul", "instructions": q.text}

    def score_batch(self, text: str, context: Dict, questions: List[Question]) -> ScoreVector:
        start = time.perf_counter()
        scores: Dict[str, float] = {}
        semantic: List[Question] = []
        for q in questions:
            if q.qtype is QuestionType.COMPUTED:
                fn = self.resolvers.get(q.resolver or "")
                if fn is None:
                    raise SdtError("no resolver registered for %r" % q.resolver)
                scores[q.qid] = 1.0 if fn(context) else 0.0
            else:
                semantic.append(q)

        if semantic:
            if not self.api_key:
                raise SdtError(
                    "GEMINI_API_KEY is not set. Create a key in Google AI Studio "
                    "(paid tier recommended: the free tier trains on your data) "
                    "and export it, or pass api_key= to GeminiDecisionClient."
                )
            request = self.build_request(text, context, semantic)
            body = json.dumps(request).encode("utf-8")
            headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
            payload = self._post_with_retry(headers, body)
            answers = payload.get("answers") or {}
            for q in semantic:
                ans = answers.get(q.qid)
                if ans is None:
                    raise SdtError("Decisions API answer missing for question %s" % q.qid)
                dt = getattr(q, "decision_type", None)
                probs = None
                if ans.get("type") == "choice" or (dt is not None and dt.value == "choice"):
                    probs = ans.get("probabilities")
                    if not probs:
                        raise SdtError("choice answer missing probabilities for %s" % q.qid)
                    for name, pv in probs.items():
                        scores[q.qid + "::" + str(name)] = float(pv)
                elif ans.get("type") == "score" or (dt is not None and dt.value == "score"):
                    probs = ans.get("probabilities")
                    if not probs:
                        raise SdtError("score answer missing probabilities for %s" % q.qid)
                    for lvl, pv in probs.items():
                        scores[q.qid + "::" + str(lvl)] = float(pv)
                else:
                    if "noul" not in ans:
                        raise SdtError("noul answer missing for question %s" % q.qid)
                    pv = float(ans["noul"])
                    if not 0.0 <= pv <= 1.0:
                        raise SdtError("noul probability out of range for %s: %r" % (q.qid, pv))
                    scores[q.qid] = pv

        elapsed = (time.perf_counter() - start) * 1000.0
        return ScoreVector(example_id="", scores=scores, latency_ms=round(elapsed, 3))

    def _post_with_retry(self, headers: Dict[str, str], body: bytes) -> Dict:
        attempts = 0
        while True:
            attempts += 1
            status, raw = self.transport(self.endpoint, headers, body)
            if status == 200:
                self.last_response = json.loads(raw.decode("utf-8"))
                return self.last_response
            if status in RETRYABLE_STATUSES and attempts <= self.max_retries:
                delay = 0.5 * (2 ** (attempts - 1)) + random.random() * 0.25
                time.sleep(delay)
                continue
            raise SdtError(
                "Decisions API HTTP %d after %d attempt(s): %s"
                % (status, attempts, raw[:300].decode("utf-8", "replace"))
            )
