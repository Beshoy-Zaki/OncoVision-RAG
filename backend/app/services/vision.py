import os
import cv2
import numpy as np
import onnxruntime as ort
from app.core.config import settings
from app.utils.logging_config import logger

class VisionService:
    _instance = None
    
    CLASSES = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
    
    def __init__(self):
        self.session = None
        
    def initialize(self):
        model_path = settings.YOLO_MODEL_PATH
        if not os.path.exists(model_path):
            alt_path = os.path.join("..", model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
                
        logger.info(f"Loading YOLO ONNX vision model from: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        logger.info("YOLO ONNX model loaded successfully.")

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        if self.session is None:
            raise RuntimeError("VisionService is not initialized.")
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file: could not decode.")
            
        return self._predict_image(img)

    def predict_from_path(self, image_path: str) -> dict:
        if self.session is None:
            raise RuntimeError("VisionService is not initialized.")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        return self._predict_image(img)

    def _predict_image(self, img_bgr: np.ndarray) -> dict:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (224, 224))
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        
        output = self.session.run(None, {'images': tensor})[0][0]
        
        # Softmax probabilities
        exp_out = np.exp(output - np.max(output))
        probs = exp_out / np.sum(exp_out)
        
        top_idx = int(np.argmax(probs))
        pred_class = self.CLASSES[top_idx]
        confidence = float(probs[top_idx])
        
        return {
            "detected_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {self.CLASSES[i]: round(float(probs[i]), 4) for i in range(len(self.CLASSES))}
        }

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

vision_service = VisionService()
