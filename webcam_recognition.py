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
import json

# Import database module
from database import (
    add_attendance, 
    get_today_attendance, 
    get_all_attendance,
    is_student_present_today,
    get_statistics as get_db_statistics
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# Webcam Face Recognition System
# ============================================================================
class WebcamFaceRecognition:
    def __init__(self, model_path='./face_models', threshold=0.65):
        self.device = device
        self.threshold = threshold
        # No longer need in-memory log, using database
        self.marked_today = set()
        
        # Load already marked students from database
        today_records = get_today_attendance()
        for record in today_records:
            self.marked_today.add(record['Name'])
        
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
        print(f"   Threshold: {threshold}")
    
    def recognize_face(self, face_crop):
        """Recognize a face from cropped image"""
        try:
            # Convert to PIL
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            
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
                return "Unknown", best_similarity, similarities
            
            return best_person, best_similarity, similarities
        except Exception as e:
            print(f"Error in recognition: {e}")
            return "Error", 0.0, {}
    
    def mark_attendance(self, person_name):
        """Mark attendance for a person using database"""
        if person_name not in self.marked_today and person_name != "Unknown":
            timestamp = datetime.now()
            
            # Save to database instead of JSON
            add_attendance(
                name=person_name.title(),
                date=timestamp.strftime('%Y-%m-%d'),
                time=timestamp.strftime('%H:%M:%S'),
                status='Present'
            )
            
            self.marked_today.add(person_name)
            print(f"\n✅ Attendance marked for {person_name.title()} at {timestamp.strftime('%H:%M:%S')}")
            return True
        return False
    
    def save_attendance(self, filename='attendance.json', format='json'):
        """Export attendance from database to JSON or CSV"""
        # Get all attendance from database
        all_records = get_all_attendance()
        
        if all_records:
            if format == 'json':
                # Save as JSON
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(all_records, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Attendance exported to {filename}")
            else:
                # Save as CSV
                df = pd.DataFrame(all_records)
                df.to_csv(filename, index=False)
                print(f"\n💾 Attendance exported to {filename}")
            return True
        else:
            print("\n⚠️ No attendance records to export")
            return False
    
    def run(self):
        """Run webcam face recognition"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n" + "="*70)
        print("🎥 WEBCAM FACE RECOGNITION STARTED")
        print("="*70)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'j' to save as JSON (default)")
        print("  - Press 'c' to save as CSV")
        print("  - Press 'r' to reset today's attendance")
        print("="*70 + "\n")
        
        frame_count = 0
        process_every_n_frames = 5  # Process every 5th frame for performance
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Process face detection every N frames
            if frame_count % process_every_n_frames == 0:
                # Detect faces
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.face_detector.detect_faces(rgb_frame)
                
                for detection in detections:
                    box = detection['box']
                    x, y, w, h = box
                    
                    # Handle negative coordinates
                    x, y = max(0, x), max(0, y)
                    
                    # Crop face
                    face_crop = frame[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:
                        # Recognize
                        person, confidence, similarities = self.recognize_face(face_crop)
                        
                        # Draw box and label
                        color = (0, 255, 0) if person != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Label
                        label = f"{person.title()}: {confidence:.2%}"
                        cv2.putText(display_frame, label, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Mark attendance if high confidence
                        if confidence > self.threshold:
                            self.mark_attendance(person)
            
            # Display info from database
            today_records = get_today_attendance()
            all_records = get_all_attendance()
            info_text = f"Marked Today: {len(self.marked_today)} | Total Logs: {len(all_records)}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show marked people
            y_offset = 60
            for person in self.marked_today:
                cv2.putText(display_frame, f"✓ {person.title()}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
            
            cv2.imshow('Face Recognition - Attendance System', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('j'):
                self.save_attendance('attendance.json', format='json')
            elif key == ord('c'):
                self.save_attendance('attendance.csv', format='csv')
            elif key == ord('r'):
                self.marked_today.clear()
                print("\n🔄 Today's attendance reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Get final statistics from database
        today_records = get_today_attendance()
        all_records = get_all_attendance()
        
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("="*70)
        print(f"Total people marked today: {len(self.marked_today)}")
        print(f"Total attendance records in database: {len(all_records)}")
        if self.marked_today:
            print(f"People present: {', '.join([p.title() for p in self.marked_today])}")
        print(f"\n💾 All data saved to database (attendance.db)")
        print("="*70)

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    system = WebcamFaceRecognition(
        model_path='./face_models',
        threshold=0.65
    )
    system.run()
