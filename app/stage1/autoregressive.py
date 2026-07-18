from app.stage1.api import get_data_from_endpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    print("Starting autoregressive check...")

    try:
        data = get_data_from_endpoint()
        print("Fetched data:", data)
    except Exception as exc:
        print(f"Data fetch failed: {exc}")

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-2.7B",
            quantization_config=quantization_config,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        inputs = tokenizer("Hello, I'm a language model", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("Autoregressive check completed successfully.")
    except Exception as exc:
        print(f"Model check failed: {exc}")


if __name__ == "__main__":
    main()
