import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import mlflow.pytorch

from src.training.model import build_model

def load_model():
    deploy_env = os.getenv("DEPLOY_ENV", "local")
    print(f"Running in environment: {deploy_env}")

    model = build_model()

    if deploy_env == "aws":
        print("Loading model from MLflow artifact (AWS mode)")
        model_uri = os.getenv("MODEL_URI")
        model = mlflow.pytorch.load_model(model_uri)

    else:
        print("Loading local model.pt (LOCAL mode)")
        model.load_state_dict(torch.load("model.pt", map_location="cpu"))
        model.eval()

    return model

# ---- Load model ----
# def load_model(model_path="model.pt"):
#     model = build_model()
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     return model

# ---- Load model from MLflow ----
# def load_model(model_uri="mlruns/0/models/m-dac28dcfe177444fa6422ba52b1d2c03/artifacts"):
#     model = mlflow.pytorch.load_model(model_uri)
#     model.eval()
#     return model

# ---- Preprocess image ----
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension because model was trained with 4 dimenssions
    return img


# ---- Predict ----
def predict(given_model, image_path):
    classes = ["Cat", "Dog"]

    img_tensor = preprocess_image(image_path)

# torch.no_grad: disables gradient calculation for all operations within its scope.
# This is primarily used during model inference or validation 
# to reduce memory consumption and speed up computations, as gradients 
# are not needed for updating model weights in these phases
    with torch.no_grad():
        outputs = given_model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]

if __name__ == "__main__":
    sample = "data/raw/PetImages/Cat/0.jpg"
    model = load_model()
    result = predict(model, sample)
    print("Prediction:", result)