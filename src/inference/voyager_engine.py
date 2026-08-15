"""
VoyagerEngine: Inference Abstraction Layer for Axelera Metis 111C NPU and Fallbacks.
"""

import os
import time
import numpy as np
from typing import List, Dict, Union, Optional, Tuple

class VoyagerEngine:
    """
    Axelera Metis AIPU Inference Engine using Voyager SDK.
    Supports seamless fallback to ONNXRuntime / PyTorch for CPU testing.
    """

    def __init__(self, axm_path: Optional[str] = None, onnx_path: Optional[str] = None, chip_id: int = 0):
        self.axm_path = axm_path
        self.onnx_path = onnx_path
        self.chip_id = chip_id
        
        self.backend = "unknown"
        self.session = None
        self.input_name = None
        self.output_names = []
        
        self._initialize_backend()

    def _initialize_backend(self):
        """Attempts to load Axelera Voyager SDK engine, falling back to ONNXRuntime if unavailable."""
        # 1. Attempt Axelera Voyager SDK load
        if self.axm_path and os.path.exists(self.axm_path):
            try:
                # Try importing voyager Python SDK (Axelera runtime)
                import voyager
                print(f"[VOYAGER SDK] Loading model {self.axm_path} on Axelera Metis chip {self.chip_id}...")
                self.session = voyager.Engine(self.axm_path, chip_id=self.chip_id)
                self.backend = "axelera_voyager"
                print(f"[VOYAGER SDK SUCCESS] Model loaded on Axelera Metis 111C AIPU.")
                return
            except ImportError:
                print(f"[VOYAGER SDK NOTICE] 'voyager' module not installed in current Python environment.")
            except Exception as e:
                print(f"[VOYAGER SDK WARNING] Could not load .axm model on Metis AIPU: {e}")

        # 2. Fallback to ONNXRuntime
        if self.onnx_path and os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]
                self.session = ort.InferenceSession(self.onnx_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                self.backend = "onnxruntime"
                print(f"[ENGINE FALLBACK SUCCESS] Model loaded via ONNXRuntime.")
                return
            except Exception as e:
                print(f"[ENGINE WARNING] Failed to load ONNX model: {e}")

        # 3. Virtual / Mock backend for testing without files
        print(f"[ENGINE NOTICE] Initializing Virtual Engine mode (Simulation).")
        self.backend = "virtual"

    def run(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Executes model inference on the given input tensor.
        :param input_tensor: Preprocessed numpy array (e.g., shape [1, 3, H, W] float32 or uint8)
        :return: List of output numpy tensors from the model.
        """
        t0 = time.time()
        
        if self.backend == "axelera_voyager":
            # Voyager SDK inference call
            outputs = self.session.run(input_tensor)
            if not isinstance(outputs, list):
                outputs = [outputs]
            return outputs

        elif self.backend == "onnxruntime":
            # Ensure float32 format
            if input_tensor.dtype != np.float32:
                input_tensor = input_tensor.astype(np.float32)
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            return outputs

        elif self.backend == "virtual":
            # Return synthetic dummy output tensor for testing flow
            # e.g., YOLO output shape [1, 84, 8400]
            batch_size = input_tensor.shape[0] if len(input_tensor.shape) == 4 else 1
            dummy_output = np.zeros((batch_size, 84, 8400), dtype=np.float32)
            return [dummy_output]

        else:
            raise RuntimeError(f"Engine backend '{self.backend}' is not properly initialized.")

    def get_backend(self) -> str:
        return self.backend
