from dotenv import load_dotenv
from transformers import pipeline
import os

# Load environment variables from .env file
load_dotenv()


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


# def extract_text(chat_history):
#     text = ""
#     for message in chat_history:
#         if message["role"] == "user":
#             text = text+"\n"+message["content"]
#     return text


def select_emotion(query):
    # if len(chat_history)==0:
    #     return os.getenv("NORMAL_BOT")
    
    # simple_text = extract_text(chat_history)
    emotion = get_most_probable_emotion(query)

    match emotion:
        case "joy":
            return {"Assistant ID":os.getenv("JOY_BOT"),"Emotion":"Joy"}
        case "disgust":
            return {"Assistant ID":os.getenv("DISGUST_BOT"),"Emotion":"Disgust"}
        case "surprise":
            return {"Assistant ID":os.getenv("SURPRISE_BOT"),"Emotion":"Surprise"}
        case "sadness":
            return {"Assistant ID":os.getenv("SAD_BOT"),"Emotion":"Sadness"}
        case "fear":
            return {"Assistant ID":os.getenv("FEAR_BOT"),"Emotion":"Fear"}
        case "anger":
            return {"Assistant ID":os.getenv("ANGER_BOT"),"Emotion":"Anger"}
        case default:
            return {"Assistant ID":os.getenv("NORMAL_BOT"),"Emotion":"Neutral"}