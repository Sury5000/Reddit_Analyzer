from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "sshleifer/distilbart-cnn-12-6"
model_path = "./models/summarizer_model"

# Download and save model and tokenizer locally
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"Model saved at {model_path}")
