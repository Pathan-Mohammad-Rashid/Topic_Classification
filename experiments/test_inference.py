import torch
import torch.nn.functional as F
from model import FastTextTopicClassifier
from utils import clean_text, collate_batch

def rigorous_test(model_path: str = "/content/drive/MyDrive/final_model_backup.pt"):
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print("Error: Could not find the model backup in your Drive. Ensure your Drive is mounted!")
        return

    vocab = checkpoint['vocab']
    label_map = checkpoint['label_map']
    inv_label_map = {v: k for k, v in label_map.items()}

    # Initialize the custom architecture
    model = FastTextTopicClassifier(len(vocab), 64, len(label_map))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Diverse test cases to challenge the model
    test_cases = [
        "The new smartphone features a 120Hz OLED display and a faster multi-core processor.",
        "The striker scored a stunning hat-trick in the second half of the championship finals.",
        "Congress passed the new infrastructure bill after months of intense political debate.",
        "Researchers discovered a new exoplanet orbiting a red dwarf star in the habitable zone.",
        "The recipe calls for two cups of flour, granulated sugar, and a generous pinch of salt."
    ]

    print("\n--- Rigorous Inference Test Results ---")
    for text in test_cases:
        cleaned = clean_text(text)
        text_tensor, offsets = collate_batch([cleaned], vocab)

        with torch.no_grad():
            outputs = model(text_tensor, offsets)
            # Apply softmax to convert raw logits into readable probabilities
            probabilities = F.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)

            predicted_topic = inv_label_map[pred_idx.item()]
            print(f"Input: '{text}'")
            print(f"Prediction: {predicted_topic} (Confidence: {confidence.item():.2%})\n")

if __name__ == "__main__":
    rigorous_test()