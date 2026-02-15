import torch
from PIL import Image
import torchvision.transforms as transforms

from src.training.model import build_model


# ---- Load model ----
def load_model(model_path="model.pt"):
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


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
def predict(image_path):
    classes = ["Cat", "Dog"]

    model = load_model()
    img_tensor = preprocess_image(image_path)

# torch.no_grad: disables gradient calculation for all operations within its scope.
# This is primarily used during model inference or validation 
# to reduce memory consumption and speed up computations, as gradients 
# are not needed for updating model weights in these phases
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]


if __name__ == "__main__":
    sample = "data/raw/PetImages/Cat/0.jpg"
    result = predict(sample)
    print("Prediction:", result)
