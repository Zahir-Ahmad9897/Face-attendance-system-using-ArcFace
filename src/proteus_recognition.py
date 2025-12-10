"""
Proteus Face Recognition with Serial Communication
===================================================

This script continuously runs face recognition and sends predictions 
to a virtual COM port for Proteus simulation.

Recognition Results Sent:
- "zahir" when Zahir is detected
- "mehran" when Mehran is detected
- "yousaf" when Yousaf is detected
- "unknown" when face is unrecognized or no face detected

Default Settings:
- COM Port: COM3
- Baud Rate: 9600
- Continuous operation with real-time updates

Usage:
    python proteus_recognition.py
    
    Or with custom COM port:
    python proteus_recognition.py --port COM5
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
import time
import argparse

# Serial communication
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    print("WARNING: pyserial not installed. Serial communication disabled.")
    print("Install with: pip install pyserial")
    SERIAL_AVAILABLE = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Model Architecture (Same as training)
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
# Serial Communication Handler
# ============================================================================
class SerialCommunicator:
    def __init__(self, port='COM3', baud_rate=9600, timeout=1):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        self.connected = False
        
    def connect(self):
        """Connect to serial port"""
        if not SERIAL_AVAILABLE:
            print("[SERIAL] pyserial not available - running in demo mode")
            return False
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.connected = True
            print(f"[SERIAL] Connected to {self.port} at {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"[SERIAL] Failed to connect to {self.port}: {e}")
            print(f"[SERIAL] Make sure virtual COM port is set up")
            self.connected = False
            return False
    
    def send(self, message):
        """Send message to serial port"""
        if not SERIAL_AVAILABLE:
            # Demo mode - just print
            print(f"[SERIAL DEMO] Would send: {message}")
            return True
            
        if not self.connected:
            # Try to reconnect
            if not self.connect():
                return False
        
        try:
            # Send message as bytes with newline
            data = f"{message}\n".encode('utf-8')
            self.serial_conn.write(data)
            self.serial_conn.flush()
            return True
        except serial.SerialException as e:
            print(f"[SERIAL] Error sending data: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn and self.connected:
            self.serial_conn.close()
            self.connected = False
            print("[SERIAL] Connection closed")

# ============================================================================
# Proteus Face Recognition System
# ============================================================================
class ProteusFaceRecognition:
    def __init__(self, model_path='./models', threshold=0.65, com_port='COM3', baud_rate=9600):
        self.device = device
        self.threshold = threshold
        
        # Initialize serial communication
        self.serial = SerialCommunicator(port=com_port, baud_rate=baud_rate)
        
        # Load model
        print("\n" + "="*70)
        print("PROTEUS FACE RECOGNITION - SERIAL OUTPUT")
        print("="*70)
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
        
        print("[OK] Model loaded successfully")
        print(f"   Device: {device}")
        print(f"   Registered people: {self.person_names}")
        print(f"   Threshold: {threshold}")
        print("="*70)
        
        # Connect to serial port
        print(f"\nConnecting to {com_port}...")
        self.serial.connect()
        print("="*70)
    
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
                return "unknown", best_similarity
            
            return best_person, best_similarity
        except Exception as e:
            print(f"[ERROR] Recognition error: {e}")
            return "unknown", 0.0
    
    def run(self):
        """Run continuous face recognition with serial output"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return
        
        print("\n" + "="*70)
        print("RUNNING - Press 'q' to quit")
        print("="*70)
        print("Serial Output: Recognition results sent to " + self.serial.port)
        print("Format: <name>\\n (e.g., 'zahir\\n', 'mehran\\n', 'unknown\\n')")
        print("="*70 + "\n")
        
        last_sent_name = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            current_name = "unknown"
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                # Detect faces
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = self.face_detector.detect_faces(rgb_frame)
                
                if detections:
                    detection = detections[0]  # Use first detected face
                    box = detection['box']
                    x, y, w, h = box
                    
                    # Handle negative coordinates
                    x, y = max(0, x), max(0, y)
                    
                    # Crop face
                    face_crop = frame[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:
                        # Recognize
                        person, confidence = self.recognize_face(face_crop)
                        current_name = person.lower()
                        
                        # Draw box and label
                        color = (0, 255, 0) if person != "unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Label
                        label = f"{person.upper()}: {confidence:.2%}"
                        cv2.putText(display_frame, label, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Send to serial if name changed
                if current_name != last_sent_name:
                    success = self.serial.send(current_name)
                    if success:
                        print(f"[SENT] {current_name}")
                    last_sent_name = current_name
            
            # Display current recognition
            info_text = f"Current: {last_sent_name if last_sent_name else 'unknown'}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Proteus Face Recognition - Press Q to quit', display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Quitting...")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.serial.close()
        
        print("\n" + "="*70)
        print("SESSION ENDED")
        print("="*70)

# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Proteus Face Recognition with Serial Output')
    parser.add_argument('--port', type=str, default='COM3', 
                      help='COM port for serial communication (default: COM3)')
    parser.add_argument('--baud', type=int, default=9600,
                      help='Baud rate (default: 9600)')
    parser.add_argument('--threshold', type=float, default=0.65,
                      help='Recognition threshold (default: 0.65)')
    parser.add_argument('--model', type=str, default='./models',
                      help='Path to model directory (default: ./models)')
    
    args = parser.parse_args()
    
    try:
        system = ProteusFaceRecognition(
            model_path=args.model,
            threshold=args.threshold,
            com_port=args.port,
            baud_rate=args.baud
        )
        system.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
