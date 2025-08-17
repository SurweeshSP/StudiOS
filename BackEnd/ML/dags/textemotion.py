from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",  
    top_k=None,
    framework="pt"
)

def predict_textemotion(text: str):
    return classifier(text)
