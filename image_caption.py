import io
import os
import requests
from PIL import Image
from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
encoder = models.resnet50(pretrained=True)
encoder.fc = nn.Linear(in_features=2048, out_features=512)
encoder = encoder.to(device)
encoder.eval()

# Load the captioning model
decoder = torch.load("captioning_model.pt")
decoder = decoder.to(device)
decoder.eval()

# Load the object detection model
object_detector = FasterRCNN(
    models.resnet50(pretrained=True),
    num_classes=91
)
object_detector.to(device)
object_detector.eval()


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = F.to_tensor(image)
    image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transform(image).unsqueeze(0)


def encode_image(image):
    image = image.to(device)
    features = encoder(image)
    return features


def caption_image(features):
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word IDs to tokens
    sampled_caption = []
    for word_id in sampled_ids:
        word = decoder.vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break

    # Remove start and end tokens, and join the words into a sentence
    caption = " ".join(sampled_caption[1:-1])

    return caption


def detect_objects(image):
    image = image.to(device)
    with torch.no_grad():
        outputs = object_detector(image)
    return outputs


@app.route("/caption", methods=["POST"])
def generate_caption():
    # Get the image from the request
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    # Read and preprocess the image
    image = Image.open(io.BytesIO(file.read()))
    preprocessed_image = preprocess_image(image)

    # Encode the image features
    encoded_image = encode_image(preprocessed_image)

    # Generate the caption
    caption = caption_image(encoded_image)

    # Detect objects in the image
    object_detections = detect_objects(preprocessed_image)

    # Get the top 3 detected objects
    top_3_objects = object_detections[0]["labels"][:3]
    object_labels = [decoder.vocab.idx2word[obj_id] for obj_id in top_3_objects]

    return jsonify({"caption": caption, "objects": object_labels})


if __name__ == "__main__":
    app.run(debug=True)
