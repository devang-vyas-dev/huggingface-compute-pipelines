from transformers import pipeline
pipe = pipeline("text-generation",model = "openai-community/gpt2")
prompt = "How is the Weather Today?"
output = pipe(prompt)
print(output)
