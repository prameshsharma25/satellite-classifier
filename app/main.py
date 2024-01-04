import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torch import nn
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch Normalization
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch Normalization
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.bn4 = nn.BatchNorm1d(512)  # Batch Normalization
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = x.view(-1, 256 * 28 * 28)
        x = self.dropout(self.relu4(self.bn4(self.fc1(x))))
        x = self.fc2(x)

        return x

model = NeuralNetwork()
model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

classes = ["cloudy", "desert", "green_area", "water"]

@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

@app.route("/predict", methods=["POST"])
def predict():
    print("/predict request")
    req_json = request.get_json()
    json_instances = req_json["instances"]
    X_list = [np.array(j["image"], dtype="uint8") for j in json_instances]
    X_transformed = torch.cat([transform(x).unsqueeze(dim=0) for x in X_list]).to(device)
    preds = model(X_transformed)
    preds_classes = [classes[i_max] for i_max in preds.argmax(1).tolist()]
    print(preds_classes)
    return jsonify({
        "predictions": preds_classes
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)