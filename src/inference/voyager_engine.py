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
        # 1. Attempt Axelera Voyager SDK load
        if self.axm_path and os.path.exists(self.axm_path):
            try:
                import voyager
                print(f"[VOYAGER SDK] Loading model {self.axm_path} on Axelera Metis chip {self.chip_id} ({self.num_cores} AIPU cores)...")

                # Auto-discover Voyager SDK model loader class/factory
                engine_cls = None
                for attr_name in ["Model", "Session", "InferenceSession", "Engine", "Pipeline", "Runtime", "load_model", "load"]:
                    if hasattr(voyager, attr_name):
                        engine_cls = getattr(voyager, attr_name)
                        break

                if engine_cls is None:
                    # Check submodules (e.g. voyager.api, voyager.runtime)
                    for submod_name in ["api", "runtime", "npu", "engine"]:
                        if hasattr(voyager, submod_name):
                            submod = getattr(voyager, submod_name)
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
                    print(f"[VOYAGER SDK SUCCESS] Model loaded via Voyager SDK ({type(self.session).__name__}).")
                    return
                else:
                    exposed_attrs = [a for a in dir(voyager) if not a.startswith('_')]
                    print(f"[VOYAGER SDK WARNING] Could not find Engine/Model class in 'voyager'. Available attributes: {exposed_attrs}")

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
