from types import SimpleNamespace

from app.stage1 import draft_policy_py


class DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in token_ids)


class DummyOutputs:
    def __init__(self, token_ids, scores=None):
        self.sequences = token_ids
        self.scores = scores or []


def test_compare_generations_reports_divergence_metrics():
    tokenizer = DummyTokenizer()
    baseline = DummyOutputs([[97, 98, 99]], scores=[SimpleNamespace(max=lambda: 1.0)])
    assisted = DummyOutputs([[97, 100, 99]], scores=[SimpleNamespace(max=lambda: 2.0)])

    report = draft_policy_py.compare_generations(baseline, assisted, tokenizer)

    assert report["sequence_match"] is False
    assert report["decoded_text_match"] is False
    assert report["token_match_rate"] == 2 / 3
    assert report["first_divergence_index"] == 1
    assert report["score_diff_summary"]["score_steps"] == 1
