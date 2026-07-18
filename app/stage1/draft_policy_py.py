from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - exercised when transformers is unavailable
    AutoModelForCausalLM = None
    AutoTokenizer = None

TARGET_MODEL = "HuggingFaceTB/SmolLM-1.7B"
DRAFT_MODEL = "HuggingFaceTB/SmolLM-135M"
DEFAULT_PROMPT = "Hugging Face is an open-source company"


def load_models(target_model_name: str = TARGET_MODEL, draft_model_name: str = DRAFT_MODEL):
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("torch and transformers are required to run model generation checks")

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, dtype="auto")
    assistant_model = AutoModelForCausalLM.from_pretrained(draft_model_name, dtype="auto")

    target_model.eval()
    assistant_model.eval()
    return tokenizer, target_model, assistant_model


def prepare_inputs(tokenizer, prompt: str = DEFAULT_PROMPT):
    return tokenizer(prompt, return_tensors="pt")


def generate_baseline(target_model, inputs, max_new_tokens: int):
    if torch is None:
        raise ImportError("torch is required to run model generation checks")

    with torch.inference_mode():
        return target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )


def generate_assisted(target_model, assistant_model, inputs, max_new_tokens: int):
    if torch is None:
        raise ImportError("torch is required to run model generation checks")

    with torch.inference_mode():
        return target_model.generate(
            **inputs,
            assistant_model=assistant_model,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )


def _as_token_list(sequence: Any):
    if torch is not None and hasattr(sequence, "tolist"):
        sequence = sequence.tolist()

    if isinstance(sequence, list):
        if sequence and isinstance(sequence[0], list):
            return sequence[0]
        return sequence

    return list(sequence)


def compare_generations(baseline_outputs, assisted_outputs, tokenizer=None):
    baseline_tokens = _as_token_list(baseline_outputs.sequences)
    assisted_tokens = _as_token_list(assisted_outputs.sequences)

    if tokenizer is not None:
        baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
        assisted_text = tokenizer.decode(assisted_tokens, skip_special_tokens=True)
    else:
        baseline_text = "".join(str(token) for token in baseline_tokens)
        assisted_text = "".join(str(token) for token in assisted_tokens)

    max_length = max(len(baseline_tokens), len(assisted_tokens))
    matched_positions = sum(
        1 for index in range(max_length) if index < len(baseline_tokens) and index < len(assisted_tokens) and baseline_tokens[index] == assisted_tokens[index]
    )
    token_match_rate = matched_positions / max_length if max_length else 1.0

    first_divergence_index = None
    for index in range(max_length):
        if index >= len(baseline_tokens) or index >= len(assisted_tokens):
            first_divergence_index = index
            break
        if baseline_tokens[index] != assisted_tokens[index]:
            first_divergence_index = index
            break

    score_diff_summary = {"score_steps": 0, "max_abs_diff": None, "mean_abs_diff": None}
    if getattr(baseline_outputs, "scores", None) and getattr(assisted_outputs, "scores", None):
        score_diffs = []
        for baseline_score, assisted_score in zip(baseline_outputs.scores, assisted_outputs.scores):
            try:
                if torch is not None and hasattr(baseline_score, "shape") and hasattr(assisted_score, "shape"):
                    diff = (assisted_score - baseline_score).abs()
                    if hasattr(diff, "mean"):
                        score_diffs.append(float(diff.mean().item()))
                    else:
                        score_diffs.append(float(diff))
                else:
                    baseline_value = baseline_score.max() if hasattr(baseline_score, "max") else baseline_score
                    assisted_value = assisted_score.max() if hasattr(assisted_score, "max") else assisted_score
                    baseline_scalar = baseline_value.item() if hasattr(baseline_value, "item") else baseline_value
                    assisted_scalar = assisted_value.item() if hasattr(assisted_value, "item") else assisted_value
                    score_diffs.append(abs(float(assisted_scalar) - float(baseline_scalar)))
            except Exception:
                continue
        score_diff_summary = {
            "score_steps": len(score_diffs),
            "max_abs_diff": max(score_diffs) if score_diffs else None,
            "mean_abs_diff": (sum(score_diffs) / len(score_diffs)) if score_diffs else None,
        }

    return {
        "sequence_match": baseline_tokens == assisted_tokens,
        "decoded_text_match": baseline_text == assisted_text,
        "baseline_length": len(baseline_tokens),
        "assisted_length": len(assisted_tokens),
        "token_match_rate": token_match_rate,
        "first_divergence_index": first_divergence_index,
        "score_diff_summary": score_diff_summary,
    }


def check_baseline(tokenizer=None, target_model=None, assistant_model=None, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 20):
    if tokenizer is None or target_model is None or assistant_model is None:
        tokenizer, target_model, assistant_model = load_models()

    inputs = prepare_inputs(tokenizer, prompt=prompt)
    baseline = generate_baseline(target_model, inputs, max_new_tokens=max_new_tokens)
    assisted = generate_assisted(target_model, assistant_model, inputs, max_new_tokens=max_new_tokens)
    report = compare_generations(baseline, assisted, tokenizer=tokenizer)
    print(report)
    return report


def main():
    try:
        tokenizer, target_model, assistant_model = load_models()
        report = check_baseline(
            tokenizer=tokenizer,
            target_model=target_model,
            assistant_model=assistant_model,
        )
        print("Generation comparison completed.")
        return report
    except Exception as exc:  # pragma: no cover - exercised in environments without deps
        print(f"Model check failed: {exc}")
        return None


if __name__ == "__main__":
    main()