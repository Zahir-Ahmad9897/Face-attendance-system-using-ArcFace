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
    def __init__(self, model_path='../models', threshold=0.65):
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
        model_path='../models',
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