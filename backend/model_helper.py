# ============================================================================================
# FILE: backend/model_helper.py
# ============================================================================================
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from PIL import Image

class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

# =====================================================================
# SYSTEM HARDWARE DETECTOR (AUTO-FALLBACK ENGINE)
# =====================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # High-speed Apple Silicon acceleration
else:
    device = torch.device("cpu")


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.2):
        super().__init__()
        self.model = models.resnet50(weights=None)
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# =====================================================================
# DETERMINISTIC ABSOLUTE PATH ENGINE
# =====================================================================
CURRENT_SCRIPT_DIR = Path(__file__).parent.resolve()

# Look for 'model' directly inside the backend folder
SHARED_MODEL_PATH = CURRENT_SCRIPT_DIR / "model" / "saved_model.pth"
model_path_str = str(SHARED_MODEL_PATH)

if not os.path.exists(model_path_str):
    raise FileNotFoundError(f"Backend Engine weights file missing at: {model_path_str}")

# =====================================================================
# GLOBAL STATE MEMORY CACHING (WARMUP ON TARGET DEVICE)
# =====================================================================
trained_model = CarClassifierResNet()
# Load the model states mapping accurately onto your system's best device
trained_model.load_state_dict(torch.load(model_path_str, map_location=device))
trained_model.to(device)
trained_model.eval()

# Shared deterministic pre-processing transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =====================================================================
# API PREDICTION ROUTINE
# =====================================================================
def predict(image_path: str) -> tuple:
    """
    Accepts an absolute or relative string file path pointing to an image,
    processes it through the PyTorch tensor matrix transformation layer,
    and references the globally cached model for rapid response delivery.
    """
    # Safe system file handling to prevent locking or leaks
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        # Move the image tensor to the same hardware device as your model
        image_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Execute high-speed prediction without calculating weight gradients
    with torch.no_grad():
        output = trained_model(image_tensor)

        # Calculate real probability distribution
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        # Format the float to 2 decimal places using an f-string
        formatted_confidence = f"{confidence.item():.2f}"

        # Return both the class string and the confidence decimal string
        return class_names[predicted_class.item()], formatted_confidence
