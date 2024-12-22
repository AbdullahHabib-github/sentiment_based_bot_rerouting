from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
    # Specify the directory where you want to save the model
    local_directory = "./model"

    # Download and save the model and tokenizer in the specified directory
    model = AutoModelForSequenceClassification.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base", 
        cache_dir=local_directory
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base", 
        cache_dir=local_directory
    )
