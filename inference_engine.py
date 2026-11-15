"""
Inference Engine for Cotton Leaf Disease Prediction & Verification.
Supports two-stage inference:
  Stage 1: Cotton Leaf Gatekeeper (cotton_detector.tflite)
  Stage 2: 9-Class Disease Specialist (cotton_leaf_disease.tflite)
"""

import os
import json
import threading
import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter


class CottonInferenceEngine:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self._lock = threading.RLock()

        self.cotton_model_path = os.path.join(base_dir, "cotton_detector.tflite")
        self.disease_model_path = os.path.join(base_dir, "cotton_leaf_disease.tflite")
        self.class_names_path = os.path.join(base_dir, "class_names_9class.json")
        self.disease_info_path = os.path.join(base_dir, "disease_info.json")

        self.class_names = self._load_class_names()
        self.disease_info = self._load_disease_info()

        self.cotton_interpreter = None
        self.disease_interpreter = None

        self._init_models()

    def _load_class_names(self):
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return [
            "Alternaria Leaf Spot",
            "Bacterial Blight",
            "Curl Virus",
            "Fusarium Wilt",
            "Healthy Leaf",
            "Herbicide Growth Damage",
            "Leaf Redding",
            "Leaf Variegation",
            "Verticillium Wilt"
        ]

    def _load_disease_info(self):
        if os.path.exists(self.disease_info_path):
            with open(self.disease_info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _init_models(self):
        for path in (self.cotton_model_path, self.disease_model_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required model is missing: {path}")
        if os.path.exists(self.cotton_model_path):
            self.cotton_interpreter = Interpreter(model_path=self.cotton_model_path)
            self.cotton_interpreter.allocate_tensors()
            self.cotton_in_idx = self.cotton_interpreter.get_input_details()[0]["index"]
            self.cotton_out_idx = self.cotton_interpreter.get_output_details()[0]["index"]

        if os.path.exists(self.disease_model_path):
            self.disease_interpreter = Interpreter(model_path=self.disease_model_path)
            self.disease_interpreter.allocate_tensors()
            self.disease_in_idx = self.disease_interpreter.get_input_details()[0]["index"]
            self.disease_out_idx = self.disease_interpreter.get_output_details()[0]["index"]

    def preprocess_image(self, image):
        """
        Resize image to 224x224 RGB float32 array in range [0..255].
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        image = ImageOps.exif_transpose(image).convert("RGB")
        resized = image.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.array(resized, dtype=np.float32)
        batch_array = np.expand_dims(img_array, axis=0)
        return resized, batch_array

    def check_is_cotton_leaf(self, batch_array):
        """
        Stage 1: Verify if the input is a genuine cotton leaf.
        Model output:
           < 0.5 => Cotton Leaf
          >= 0.5 => Not a Cotton Leaf
        """
        if self.cotton_interpreter is None:
            raise RuntimeError("Cotton verification model is unavailable.")

        with self._lock:
            self.cotton_interpreter.set_tensor(self.cotton_in_idx, batch_array)
            self.cotton_interpreter.invoke()
            output = self.cotton_interpreter.get_tensor(self.cotton_out_idx)
        raw_val = float(output[0][0])
        if not np.isfinite(raw_val) or not 0 <= raw_val <= 1:
            raise ValueError("Cotton model returned an invalid probability.")

        is_cotton = (raw_val < 0.5)
        # Cotton confidence: closer to 0 is higher confidence
        cotton_confidence = max(0.0, min(100.0, (1.0 - raw_val) * 100))
        return is_cotton, cotton_confidence, raw_val

    def predict_disease(self, batch_array):
        """
        Stage 2: 9-class deep disease classification.
        """
        if self.disease_interpreter is None:
            raise RuntimeError("Disease interpreter not initialized.")

        with self._lock:
            self.disease_interpreter.set_tensor(self.disease_in_idx, batch_array)
            self.disease_interpreter.invoke()
            output = self.disease_interpreter.get_tensor(self.disease_out_idx)[0]
        if (len(output) != len(self.class_names) or not np.all(np.isfinite(output))
                or np.any(output < 0) or np.any(output > 1)
                or not np.isclose(np.sum(output), 1.0, atol=0.01)):
            raise ValueError("Disease model output does not match the class probabilities.")

        probs = [float(p) for p in output]
        ranked_indices = np.argsort(probs)[::-1]

        top_idx = int(ranked_indices[0])
        second_idx = int(ranked_indices[1])

        top_class = self.class_names[top_idx]
        top_confidence = probs[top_idx] * 100.0

        second_class = self.class_names[second_idx]
        second_confidence = probs[second_idx] * 100.0

        margin = top_confidence - second_confidence
        is_low_confidence = (top_confidence < 60.0)
        is_ambiguous = (margin < 12.0)

        all_preds = [
            {
                "class_name": self.class_names[i],
                "probability": float(probs[i] * 100.0),
                "index": int(i)
            }
            for i in ranked_indices
        ]

        top_3 = all_preds[:3]

        return {
            "predicted_class": top_class,
            "confidence": top_confidence,
            "second_class": second_class,
            "second_confidence": second_confidence,
            "margin": margin,
            "is_low_confidence": is_low_confidence,
            "is_ambiguous": is_ambiguous,
            "top_3": top_3,
            "all_predictions": all_preds
        }

    def generate_saliency_map(self, original_image, intensity=0.65):
        """
        Compute visual lesion focus map emphasizing color anomalies,
        necrotic discoloration, and high-frequency lesion boundaries.
        Returns a PIL Image heatmap overlay.
        """
        img_rgb = original_image.convert("RGB").resize((224, 224))
        arr = np.array(img_rgb, dtype=np.float32)

        # Calculate deviation from healthy leaf green reference
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        green_excess = 2.0 * g - r - b
        chlorosis_necrosis = (r + 0.5 * b) / (g + 1e-5)

        # Edge & gradient enhancement
        gray = img_rgb.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(gray, dtype=np.float32)

        # Combined anomaly score
        score = (chlorosis_necrosis * 1.5) + (edge_arr / 255.0 * 2.0) - (green_excess / 255.0)
        score = np.clip(score, 0, None)
        min_v, max_v = score.min(), score.max()
        if max_v > min_v:
            norm_score = (score - min_v) / (max_v - min_v)
        else:
            norm_score = np.zeros_like(score)

        # Apply colormap (smooth heatmap: blue -> cyan -> yellow -> red)
        h_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        # Red channel
        h_arr[:, :, 0] = np.clip(norm_score * 255 * 1.6, 0, 255).astype(np.uint8)
        # Green channel
        h_arr[:, :, 1] = np.clip((1.0 - np.abs(norm_score - 0.5) * 2) * 230, 0, 255).astype(np.uint8)
        # Blue channel
        h_arr[:, :, 2] = np.clip((1.0 - norm_score) * 255 * 0.8, 0, 255).astype(np.uint8)

        heatmap_pil = Image.fromarray(h_arr).filter(ImageFilter.GaussianBlur(radius=3))
        blended = Image.blend(img_rgb, heatmap_pil, alpha=intensity)
        return blended

    def analyze(self, image_input, enforce_cotton_check=True):
        """
        Complete end-to-end diagnosis pipeline.
        """
        resized_img, batch_array = self.preprocess_image(image_input)

        is_cotton, cotton_conf, raw_gate_score = self.check_is_cotton_leaf(batch_array)

        # If gatekeeper rejects and enforcement is enabled
        if enforce_cotton_check and not is_cotton:
            return {
                "success": True,
                "is_cotton": False,
                "cotton_confidence": cotton_conf,
                "gate_score": raw_gate_score,
                "predicted_class": "Not a Cotton Leaf",
                "message": (
                    "The uploaded image does not appear to be a cotton leaf. "
                    "Please upload a clear, focused image of a cotton plant leaf."
                ),
                "disease_info": None,
                "top_3": [],
                "all_predictions": [],
                "heatmap": None
            }

        # Stage 2 Disease prediction
        disease_results = self.predict_disease(batch_array)
        predicted_class = disease_results["predicted_class"]
        info = self.disease_info.get(predicted_class, {})

        # Generate visual saliency heatmap
        heatmap_img = self.generate_saliency_map(resized_img)

        return {
            "success": True,
            "is_cotton": bool(is_cotton),
            "cotton_confidence": cotton_conf,
            "gate_score": raw_gate_score,
            "predicted_class": predicted_class,
            "confidence": disease_results["confidence"],
            "second_class": disease_results["second_class"],
            "second_confidence": disease_results["second_confidence"],
            "margin": disease_results["margin"],
            "is_low_confidence": disease_results["is_low_confidence"],
            "is_ambiguous": disease_results["is_ambiguous"],
            "top_3": disease_results["top_3"],
            "all_predictions": disease_results["all_predictions"],
            "disease_info": info,
            "heatmap": heatmap_img
        }
