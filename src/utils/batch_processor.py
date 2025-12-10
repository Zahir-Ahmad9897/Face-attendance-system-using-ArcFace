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
from datetime import datetime
import pandas as pd
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Model Architecture
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
# Batch Image Processor
# ============================================================================
class BatchImageProcessor:
    def __init__(self, model_path='./face_models', threshold=0.65):
        self.device = device
        self.threshold = threshold
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(
            os.path.join(model_path, 'best_model.pth'),
            map_location=device,
            weights_only=False
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
        print(f"   Device: {device}")
        print(f"   Registered people: {self.person_names}")
        print(f"   Threshold: {threshold}\n")
    
    def recognize_image(self, image_path):
        """Recognize faces in a single image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            # Detect faces
            detections = self.face_detector.detect_faces(image_np)
            
            if not detections:
                return {
                    'image': os.path.basename(image_path),
                    'status': 'No face detected',
                    'person': None,
                    'confidence': 0.0
                }
            
            # Process first face
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
                best_person = "Unknown"
            
            return {
                'image': os.path.basename(image_path),
                'status': 'Success',
                'person': best_person.title(),
                'confidence': best_similarity,
                'all_scores': {k.title(): v for k, v in similarities.items()}
            }
            
        except Exception as e:
            return {
                'image': os.path.basename(image_path),
                'status': f'Error: {str(e)}',
                'person': None,
                'confidence': 0.0
            }
    
    def process_folder(self, folder_path, output_csv='batch_results.csv'):
        """Process all images in a folder"""
        folder = Path(folder_path)
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No images found in {folder_path}")
            return
        
        print(f"📁 Found {len(image_files)} images")
        print("="*70)
        
        results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing [{i}/{len(image_files)}]: {image_file.name}...", end=' ')
            
            result = self.recognize_image(str(image_file))
            results.append(result)
            
            if result['status'] == 'Success':
                print(f"✅ {result['person']} ({result['confidence']:.2%})")
            else:
                print(f"⚠️  {result['status']}")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to {output_csv}")
        
        # Print summary
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"Total images processed: {len(results)}")
        
        successful = sum(1 for r in results if r['status'] == 'Success')
        print(f"Successfully recognized: {successful}")
        
        no_face = sum(1 for r in results if 'No face' in r['status'])
        print(f"No face detected: {no_face}")
        
        errors = sum(1 for r in results if 'Error' in r['status'])
        print(f"Errors: {errors}")
        
        # Count by person
        print("\n👥 Recognition breakdown:")
        person_counts = {}
        for r in results:
            if r['person']:
                person_counts[r['person']] = person_counts.get(r['person'], 0) + 1
        
        for person, count in sorted(person_counts.items()):
            print(f"   {person}: {count}")
        
        print("="*70)
        
        return df

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import sys
    
    processor = BatchImageProcessor(
        model_path='./face_models',
        threshold=0.65
    )
    
    # Check if folder path provided
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path containing images: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
    else:
        # Process folder
        output_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        processor.process_folder(folder_path, output_csv=output_file)
