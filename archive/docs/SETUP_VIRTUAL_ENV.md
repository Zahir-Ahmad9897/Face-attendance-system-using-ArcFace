# Setup Virtual Environment & Download Model from Colab

## 🎯 Complete Setup Guide

Follow these steps to set up a virtual environment and download your trained model from Colab.

---

## STEP 1: Create Virtual Environment on Your PC

### On Windows (PowerShell):

```powershell
# Navigate to your project folder
cd d:\face_det\ROLLCALL.AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### On Linux/Mac:

```bash
# Navigate to your project folder
cd /path/to/face_det/ROLLCALL.AI

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**You should see `(venv)` in your terminal prompt**

---

## STEP 2: Install Dependencies in Virtual Environment

```bash
# Make sure virtual environment is activated (you see (venv))

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install opencv-python pillow matplotlib numpy scipy mtcnn

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## STEP 3: Download Model from Colab

### Method A: Direct Download (Recommended)

In your Colab notebook, run this cell:

```python
# ============================================================================
# Download Model Files to Your PC
# ============================================================================
from google.colab import files
import os

# Create a folder for downloads
os.makedirs('/content/downloads', exist_ok=True)

# Copy files to download folder
import shutil
shutil.copy('/content/face_models/best_model.pth', '/content/downloads/')
shutil.copy('/content/face_models/embeddings_db.npz', '/content/downloads/')
shutil.copy('/content/face_models/class_mapping.json', '/content/downloads/')

print("✅ Files ready for download!")
print("\nDownloading files...")

# Download each file
files.download('/content/downloads/best_model.pth')
files.download('/content/downloads/embeddings_db.npz')
files.download('/content/downloads/class_mapping.json')

print("\n✅ All files downloaded!")
print("\nSave these files to: d:\\face_det\\ROLLCALL.AI\\face_models\\")
```

**Files will download to your browser's download folder**

### Method B: Mount Google Drive

If direct download is slow:

```python
# In Colab
from google.colab import drive
drive.mount('/content/drive')

# Copy to Google Drive
import shutil
shutil.copytree('/content/face_models', '/content/drive/MyDrive/face_models')

print("✅ Files copied to Google Drive!")
print("Access them at: Google Drive > face_models/")
```

Then download from Google Drive to your PC.

---

## STEP 4: Organize Files Locally

Create this folder structure:

```
d:\face_det\ROLLCALL.AI\
├── venv\                      # Virtual environment (created)
├── face_models\               # Create this folder
│   ├── best_model.pth        # Downloaded from Colab
│   ├── embeddings_db.npz     # Downloaded from Colab
│   └── class_mapping.json    # Downloaded from Colab
└── inference.py               # Will create next
```

**Create the face_models folder:**

```powershell
# In PowerShell (make sure you're in ROLLCALL.AI folder)
mkdir face_models

# Move downloaded files here
# (Manually move from Downloads folder to d:\face_det\ROLLCALL.AI\face_models\)
```

---

## STEP 5: Create Inference Script

Create `inference.py` in `d:\face_det\ROLLCALL.AI\`:

```python
"""
Local Inference Script
Run in virtual environment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from mtcnn import MTCNN
import os
import json
from scipy.spatial import distance as dist

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# Model Architecture (SAME AS TRAINING)
# ============================================================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.25):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        # Training logic (not used in inference)
        return cosine * self.s

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        self.arcface = ArcMarginProduct(embedding_size, num_classes, s=30.0, m=0.25)

    def forward(self, x, labels=None):
        features = self.features(x)
        features = torch.flatten(features, 1)
        embedding = self.embedding(features)
        if labels is not None:
            output = self.arcface(embedding, labels)
            return output, embedding
        return F.normalize(embedding, p=2, dim=1)

# ============================================================================
# Complete System
# ============================================================================
class FaceRecognitionSystem:
    def __init__(self, model_path='./face_models', threshold=0.65):
        self.device = device
        self.threshold = threshold
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(
            os.path.join(model_path, 'best_model.pth'),
            map_location=device
        )
        
        mapping = checkpoint['class_mapping']
        num_classes = len(mapping['idx_to_class'])
        
        self.model = ArcFaceModel(num_classes, embedding_size=512).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load embeddings
        embeddings_data = np.load(os.path.join(model_path, 'embeddings_db.npz'))
        self.person_names = embeddings_data['names'].tolist()
        self.person_embeddings = embeddings_data['embeddings']
        
        # Face detector
        self.face_detector = MTCNN()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        print(f"✅ System loaded!")
        print(f"   Registered people: {self.person_names}")
        print(f"   Threshold: {threshold}")
    
    def recognize(self, image_path):
        """Recognize person from image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Detect face
        detections = self.face_detector.detect_faces(image_np)
        
        if not detections:
            return None, "No face detected"
        
        # Get first face
        detection = detections[0]
        box = detection['box']
        x, y, w, h = box
        
        # Crop face
        face_crop = image_np[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_crop)
        
        # Extract embedding
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(face_tensor)
            embedding_np = embedding.cpu().numpy()[0]
        
        # Calculate similarities
        similarities = {}
        for i, person_name in enumerate(self.person_names):
            person_emb = self.person_embeddings[i]
            similarity = np.dot(embedding_np, person_emb)
            similarities[person_name] = float(similarity)
        
        # Get best match
        best_person = max(similarities, key=similarities.get)
        best_similarity = similarities[best_person]
        
        if best_similarity < self.threshold:
            return "Unknown", best_similarity
        
        return best_person, best_similarity

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Initialize system
    system = FaceRecognitionSystem(
        model_path='./face_models',
        threshold=0.65
    )
    
    # Test with an image
    print("\n" + "="*70)
    print("Testing Recognition")
    print("="*70)
    
    # Replace with your test image path
    test_image = "test.jpg"  # Put a test image here
    
    if os.path.exists(test_image):
        person, confidence = system.recognize(test_image)
        print(f"\nResult: {person}")
        print(f"Confidence: {confidence:.2%}")
    else:
        print(f"\n⚠️  Test image not found: {test_image}")
        print("Place a test image and update the path above")
```

---

## STEP 6: Test the System

```powershell
# Make sure virtual environment is activated
# You should see (venv) in prompt

# Run inference
python inference.py
```

**Expected Output:**
```
Device: cpu
Loading model...
✅ System loaded!
   Registered people: ['mehran', 'yousaf', 'zahir']
   Threshold: 0.65

======================================================================
Testing Recognition
======================================================================

Result: zahir
Confidence: 79.11%
```

---

## STEP 7: Deactivate Virtual Environment (When Done)

```powershell
# Deactivate virtual environment
deactivate
```

---

## 🔧 Troubleshooting

### "Module not found"
```bash
# Make sure venv is activated
# Reinstall packages
pip install torch torchvision opencv-python pillow matplotlib numpy scipy mtcnn
```

### "CUDA not available"
- Normal if you don't have GPU
- Model will run on CPU (slower but works)

### "File not found"
- Check file paths are correct
- Ensure files are in `face_models/` folder

---

## 📁 Final Structure

```
d:\face_det\ROLLCALL.AI\
├── venv\                      # Virtual environment
│   ├── Scripts\
│   ├── Lib\
│   └── ...
├── face_models\               # Model files
│   ├── best_model.pth        # ~45 MB
│   ├── embeddings_db.npz     # ~5 KB
│   └── class_mapping.json    # ~1 KB
├── inference.py               # Your inference script
└── test.jpg                   # Test image
```

---

## ✅ You're Done!

Your virtual environment is set up and your model is ready to use locally! 🎉
