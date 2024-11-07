from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("neody/nemma-100m")
model = AutoModelForCausalLM.from_pretrained("neody/nemma-100m")

while True:
    gen = model.generate(
        **tokenizer(
            input("Message: "),
            return_tensors="pt",
        ),
        generation_config=GenerationConfig(max_new_tokens=200, repetition_penalty=1.2),
    )
    print("Output: " + tokenizer.decode(gen[0], skip_special_tokens=True))
