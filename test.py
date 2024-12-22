from transformers import pipeline

def get_most_probable_emotion(text: str):
    # Initialize emotion detection pipeline
    # emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    emotion_classifier = pipeline(
    "text-classification",
    model="./model/models--j-hartmann--emotion-english-distilroberta-base/snapshots/0e1cd914e3d46199ed785853e12b57304e04178b",
    )
    # Get emotions predictions
    emotions = emotion_classifier(text)
    print(emotions)
    # Extract the emotion with the highest probability
    most_probable_emotion = max(emotions, key=lambda x: x['score'])
    return most_probable_emotion['label']

if __name__ == "__main__":
    text_corpus = "I am feeling so sad and happy"
    emotion = get_most_probable_emotion(text_corpus)
    print(f"The most probable emotion is: {emotion}")
