import torch
from model import FastTextTopicClassifier
from utils import clean_text, collate_batch

def predict_topic(text: str):
    checkpoint = torch.load("final_models/final_model.pt", map_location='cpu')
    vocab, label_map = checkpoint['vocab'], checkpoint['label_map']
    inv_label_map = {v: k for k, v in label_map.items()}

    model = FastTextTopicClassifier(len(vocab), 64, len(label_map))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    text_tensor, offsets = collate_batch([clean_text(text)], vocab)
    with torch.no_grad():
        pred_idx = torch.argmax(model(text_tensor, offsets), dim=1).item()
    return inv_label_map[pred_idx]

if __name__ == "__main__":
    test_text = "I love watching the new football team"
    print(f"Input: {test_text}")
    print(f"Prediction: {predict_topic(test_text)}")