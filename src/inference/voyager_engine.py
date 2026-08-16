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

    def __init__(self, axm_path: Optional[str] = None, onnx_path: Optional[str] = None, chip_id: int = 0, num_cores: int = 4):
        self.axm_path = axm_path
        self.onnx_path = onnx_path
        self.chip_id = chip_id
        self.num_cores = num_cores
        
        self.backend = "unknown"
        self.session = None
        self.input_name = None
        self.output_names = []
        
        self._initialize_backend()

    def _initialize_backend(self):
        """Attempts to load Axelera Voyager SDK engine, falling back to ONNXRuntime if unavailable."""
        # 1. Attempt Axelera Voyager / Metis SDK load
        if self.axm_path and os.path.exists(self.axm_path):
            ax_module = None
            module_name_used = None

            # Try importing official Axelera Python SDK packages first
            for mod_name in ["axelera", "axelera_voyager", "voyager_sdk", "metis", "voyager"]:
                try:
                    mod = __import__(mod_name)
                    # Skip Spotify's unrelated 'voyager' ANN vector library (which exposes 'Float8Index' / 'Space')
                    if mod_name == "voyager" and hasattr(mod, "Float8Index") and not hasattr(mod, "Engine") and not hasattr(mod, "Model"):
                        continue
                    ax_module = mod
                    module_name_used = mod_name
                    break
                except ImportError:
                    continue

            if ax_module is not None:
                try:
                    print(f"[AXELERA SDK] Loading model {self.axm_path} via '{module_name_used}' (Chip {self.chip_id}, {self.num_cores} cores)...")
                    
                    engine_cls = None
                    for attr_name in ["Model", "Session", "InferenceSession", "Engine", "Pipeline", "Runtime", "load_model", "load"]:
                        if hasattr(ax_module, attr_name):
                            engine_cls = getattr(ax_module, attr_name)
                            break

                    if engine_cls is None:
                        for submod_name in ["api", "runtime", "npu", "engine", "inference"]:
                            if hasattr(ax_module, submod_name):
                                submod = getattr(ax_module, submod_name)
                                for attr_name in ["Model", "Session", "InferenceSession", "Engine", "Pipeline", "Runtime"]:
                                    if hasattr(submod, attr_name):
                                        engine_cls = getattr(submod, attr_name)
                                        break
                            if engine_cls:
                                break

                    if engine_cls is not None:
                        try:
                            self.session = engine_cls(self.axm_path, chip_id=self.chip_id, num_cores=self.num_cores)
                        except TypeError:
                            try:
                                self.session = engine_cls(self.axm_path, chip_id=self.chip_id)
                            except TypeError:
                                self.session = engine_cls(self.axm_path)

                        self.backend = "axelera_voyager"
                        print(f"[AXELERA SDK SUCCESS] Model loaded on Metis AIPU via '{module_name_used}.{getattr(engine_cls, '__name__', 'Engine')}'.")
                        return
                    else:
                        exposed_attrs = [a for a in dir(ax_module) if not a.startswith('_')]
                        print(f"[AXELERA SDK WARNING] Could not find Engine/Model class in '{module_name_used}'. Exposed attributes: {exposed_attrs}")
                except Exception as e:
                    print(f"[AXELERA SDK WARNING] Could not load .axm model on Metis AIPU: {e}")
            else:
                print(f"[AXELERA SDK NOTICE] Axelera NPU Python SDK ('axelera' / 'axelera_voyager') not found in environment.")

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
            if hasattr(self.session, "run"):
                outputs = self.session.run(input_tensor)
            elif hasattr(self.session, "predict"):
                outputs = self.session.predict(input_tensor)
            elif hasattr(self.session, "forward"):
                outputs = self.session.forward(input_tensor)
            elif callable(self.session):
                outputs = self.session(input_tensor)
            else:
                raise RuntimeError(f"Voyager session object '{type(self.session).__name__}' has no run/predict method")

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
            batch_size = input_tensor.shape[0] if len(input_tensor.shape) == 4 else 1
            dummy_output = np.zeros((batch_size, 84, 8400), dtype=np.float32)
            return [dummy_output]

        else:
            raise RuntimeError(f"Engine backend '{self.backend}' is not properly initialized.")

    def get_backend(self) -> str:
        return self.backend
