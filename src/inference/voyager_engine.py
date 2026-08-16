"""
VoyagerEngine: Inference Abstraction Layer for Axelera Metis 111C NPU and Fallbacks.
"""

import os
import time
import importlib
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

    def _find_axelera_engine_class(self) -> Tuple[Optional[type], Optional[str]]:
        """Discovers Axelera NPU model loader class across all module variations and submodules."""
        possible_imports = [
            "axelera.voyager",
            "axelera.runtime",
            "axelera.inference",
            "axelera.pipeline",
            "axelera.npu",
            "axelera_voyager",
            "voyager_sdk",
            "metis",
            "axelera"
        ]

        target_attrs = [
            "Model", "Pipeline", "Session", "InferenceSession", "Engine", 
            "Runtime", "AxeleraModel", "AxeleraPipeline", "Runner", "load_model", "load"
        ]

        for mod_path in possible_imports:
            try:
                mod = importlib.import_module(mod_path)
                # Skip Spotify's unrelated 'voyager' ANN library
                if hasattr(mod, "Float8Index") and not any(hasattr(mod, a) for a in target_attrs):
                    continue

                for attr_name in target_attrs:
                    if hasattr(mod, attr_name):
                        cls_obj = getattr(mod, attr_name)
                        return cls_obj, mod_path

                # Check sub-attributes of the imported module
                for sub_attr in dir(mod):
                    if not sub_attr.startswith('_'):
                        try:
                            sub_obj = getattr(mod, sub_attr)
                            for attr_name in target_attrs:
                                if hasattr(sub_obj, attr_name):
                                    cls_obj = getattr(sub_obj, attr_name)
                                    return cls_obj, f"{mod_path}.{sub_attr}"
                        except Exception:
                            pass
            except (ImportError, AttributeError):
                continue
            except Exception:
                continue

        return None, None

    def _initialize_backend(self):
        """Attempts to load Axelera Voyager SDK engine, falling back to ONNXRuntime if unavailable."""
        # 1. Attempt Axelera Voyager / Metis SDK load
        if self.axm_path and os.path.exists(self.axm_path):
            engine_cls, mod_path = self._find_axelera_engine_class()

            if engine_cls is not None:
                try:
                    print(f"[AXELERA SDK] Loading model {self.axm_path} via '{mod_path}.{getattr(engine_cls, '__name__', 'Engine')}' (Chip {self.chip_id}, {self.num_cores} cores)...")
                    try:
                        self.session = engine_cls(self.axm_path, chip_id=self.chip_id, num_cores=self.num_cores)
                    except TypeError:
                        try:
                            self.session = engine_cls(self.axm_path, chip_id=self.chip_id)
                        except TypeError:
                            self.session = engine_cls(self.axm_path)

                    self.backend = "axelera_voyager"
                    print(f"[AXELERA SDK SUCCESS] Model loaded on Metis AIPU via '{mod_path}'.")
                    return
                except Exception as e:
                    print(f"[AXELERA SDK WARNING] Could not load .axm model on Metis AIPU: {e}")
            else:
                print(f"[AXELERA SDK NOTICE] Axelera NPU model loader class not found in imported module paths.")

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
