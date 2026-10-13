import os
import cv2
import numpy as np
import onnxruntime as ort
from app.core.config import settings
from app.utils.logging_config import logger


class VisionService:
    CLASSES = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}

    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self.input_name: str = "images"
        self.input_hw: tuple[int, int] = (224, 224)

    def initialize(self):
        model_path = settings.YOLO_MODEL_PATH
        if not os.path.exists(model_path):
            alt_path = os.path.abspath(os.path.join("..", model_path))
            if os.path.exists(alt_path):
                model_path = alt_path

        if not os.path.exists(model_path):
            logger.warning(f"YOLO ONNX model not found at {model_path}. Vision service disabled.")
            return

        logger.info(f"Loading YOLO ONNX vision model from: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
            self.input_hw = (inp.shape[2], inp.shape[3])

        logger.info(
            f"YOLO ONNX model loaded (input='{self.input_name}', resolution={self.input_hw})."
        )

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        if self.session is None:
            raise RuntimeError("VisionService is not initialized.")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file: could not decode image bytes.")

        return self._predict_image(img)

    def predict_from_path(self, image_path: str) -> dict:
        if self.session is None:
            raise RuntimeError("VisionService is not initialized.")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path}")

        return self._predict_image(img)

    def _predict_image(self, img_bgr: np.ndarray) -> dict:
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, self.input_hw, interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        raw_out = self.session.run(None, {self.input_name: tensor})[0]
        output = raw_out[0]

        # Case 1: Object Detection ONNX export -> shape (N, 6) [x1, y1, x2, y2, conf, cls_id]
        if output.ndim == 2 and output.shape[1] == 6:
            return self._parse_detection_output(output, orig_w, orig_h)

        # Case 2: Classification ONNX export -> shape (4,)
        output = output.flatten()
        # Check if output is already probability distribution (Ultralytics Classify head applies softmax)
        is_already_prob = (
            np.all((output >= 0.0) & (output <= 1.0))
            and np.isclose(np.sum(output), 1.0, atol=1e-2)
        )
        if is_already_prob:
            probs = output / np.sum(output)
        else:
            exp_out = np.exp(output - np.max(output))
            probs = exp_out / np.sum(exp_out)

        top_idx = int(np.argmax(probs))
        pred_class = self.CLASSES.get(top_idx, "unknown")
        confidence = float(probs[top_idx])

        return {
            "detected_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.CLASSES[i]: round(float(probs[i]), 4) for i in range(len(self.CLASSES))
            },
            "bbox": None,
        }

    def _parse_detection_output(self, detections: np.ndarray, orig_w: int, orig_h: int) -> dict:
        det_classes = {0: "glioma", 1: "meningioma", 2: "pituitary"}
        valid = detections[detections[:, 4] >= settings.CONF_THRESHOLD]

        probs = {cls_name: 0.0 for cls_name in self.CLASSES.values()}
        if len(valid) == 0:
            max_tumor_conf = float(np.max(detections[:, 4])) if len(detections) > 0 else 0.0
            notumor_conf = round(max(0.0, 1.0 - max_tumor_conf), 4)
            probs["notumor"] = notumor_conf
            return {
                "detected_class": "notumor",
                "confidence": notumor_conf,
                "probabilities": probs,
                "bbox": None,
            }

        best_idx = int(np.argmax(valid[:, 4]))
        x1, y1, x2, y2, conf, cls_id = valid[best_idx]
        pred_class = det_classes.get(int(cls_id), "glioma")

        scale_x = orig_w / float(self.input_hw[0])
        scale_y = orig_h / float(self.input_hw[1])
        bbox = [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        ]

        probs[pred_class] = round(float(conf), 4)
        probs["notumor"] = round(max(0.0, 1.0 - float(conf)), 4)

        return {
            "detected_class": pred_class,
            "confidence": round(float(conf), 4),
            "probabilities": probs,
            "bbox": bbox,
        }

    @property
    def is_loaded(self) -> bool:
        return self.session is not None


vision_service = VisionService()
