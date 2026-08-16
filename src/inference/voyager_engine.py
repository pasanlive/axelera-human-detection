"""
VoyagerEngine: Inference Abstraction Layer for Axelera Metis 111C NPU and Fallbacks.
"""

import os
import time
import importlib
from pathlib import Path
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any

class VoyagerEngine:
    """
    Axelera Metis AIPU Inference Engine using Voyager SDK / axelera.runtime.
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

    def _try_load_axelera_runtime(self) -> bool:
        """Dedicated loader for official axelera.runtime package."""
        try:
            import axelera.runtime as axr
            print(f"[AXELERA RUNTIME] Initializing Axelera Metis NPU via axelera.runtime...")

            # 1. Discover Context / Devices
            ctx = None
            if hasattr(axr, "select_devices"):
                try:
                    devices = axr.select_devices()
                    if devices:
                        for ctx_fn in [
                            lambda: axr.Context(devices),
                            lambda: axr.Context(devices[0]),
                            lambda: axr.Context(devices=devices),
                        ]:
                            try:
                                ctx = ctx_fn()
                                if ctx is not None:
                                    break
                            except Exception:
                                continue
                except Exception as e:
                    print(f"[AXELERA RUNTIME NOTICE] select_devices: {e}")

            if ctx is None:
                for ctx_fn in [
                    lambda: axr.Context(),
                    lambda: axr.Context(self.chip_id),
                    lambda: axr.Context(device_id=self.chip_id)
                ]:
                    try:
                        ctx = ctx_fn()
                        if ctx is not None:
                            break
                    except Exception:
                        continue

            if ctx is None:
                print("[AXELERA RUNTIME WARNING] Could not instantiate axelera.runtime.Context")
                return False

            print("[AXELERA RUNTIME SUCCESS] Created axelera.runtime.Context handle.")

            # 2. Load Model
            path_str = str(self.axm_path)
            model_obj = None

            for load_fn in [
                lambda: axr.Model(path_str, ctx),
                lambda: axr.Model(ctx, path_str),
                lambda: axr.axelera_load_model(path_str, ctx),
                lambda: axr.graph_exec_load_model(path_str, ctx),
                lambda: axr.Model(path_str, context=ctx),
            ]:
                try:
                    model_obj = load_fn()
                    if model_obj is not None:
                        break
                except Exception as e:
                    last_e = e
                    continue

            if model_obj is None:
                print(f"[AXELERA RUNTIME WARNING] Could not load model '{path_str}' into axelera.runtime.Model")
                return False

            # 3. Create ModelInstance if required
            if hasattr(model_obj, "create_instance"):
                try:
                    self.session = model_obj.create_instance()
                except Exception:
                    self.session = model_obj
            elif hasattr(model_obj, "create_model_instance"):
                try:
                    self.session = model_obj.create_model_instance()
                except Exception:
                    self.session = model_obj
            else:
                self.session = model_obj

            self.backend = "axelera_voyager"
            print(f"[AXELERA RUNTIME SUCCESS] Model '{Path(self.axm_path).name}' successfully loaded on Metis AIPU ({self.num_cores} cores).")
            return True

        except ImportError:
            return False
        except Exception as e:
            print(f"[AXELERA RUNTIME WARNING] Load attempt failed: {e}")
            return False

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

    def _get_or_create_context(self, mod_path: Optional[str]) -> Optional[Any]:
        """Searches across Axelera modules to instantiate an NPU Context / Device handle."""
        search_modules = []
        if mod_path:
            search_modules.append(mod_path)
            root_mod = mod_path.split('.')[0]
            if root_mod not in search_modules:
                search_modules.append(root_mod)

        search_modules.extend([
            "axelera",
            "axelera.voyager",
            "axelera.runtime",
            "axelera.npu",
            "axelera.api",
            "axelera.inference",
            "axelera_voyager",
            "voyager_sdk",
            "metis"
        ])

        target_ctx_names = [
            "Context", "Device", "RuntimeContext", "EngineContext", 
            "create_context", "context", "get_context", "npu_context"
        ]

        for m_name in search_modules:
            try:
                mod = importlib.import_module(m_name)

                # If select_devices is available
                devices = None
                if hasattr(mod, "select_devices"):
                    try:
                        devices = mod.select_devices()
                    except Exception:
                        pass

                for ctx_name in target_ctx_names:
                    if hasattr(mod, ctx_name):
                        factory = getattr(mod, ctx_name)
                        ctx_attempts = []
                        if devices:
                            ctx_attempts.extend([
                                lambda f=factory, d=devices: f(d),
                                lambda f=factory, d=devices: f(d[0]),
                                lambda f=factory, d=devices: f(devices=d),
                            ])

                        ctx_attempts.extend([
                            lambda f=factory: f(chip_id=self.chip_id),
                            lambda f=factory: f(self.chip_id),
                            lambda f=factory: f(device=self.chip_id),
                            lambda f=factory: f(chip=self.chip_id),
                            lambda f=factory: f(num_cores=self.num_cores),
                            lambda f=factory: f("metis-111c"),
                            lambda f=factory: f(),
                        ])

                        for fn in ctx_attempts:
                            try:
                                ctx = fn()
                                if ctx is not None:
                                    print(f"[AXELERA SDK] Created NPU Context via '{m_name}.{ctx_name}'")
                                    return ctx
                            except Exception:
                                continue
            except Exception:
                continue

        return None

    def _initialize_backend(self):
        """Attempts to load Axelera Voyager SDK engine, falling back to ONNXRuntime if unavailable."""
        # 1. Attempt Axelera Voyager / Metis SDK load
        if self.axm_path and os.path.exists(self.axm_path):
            # First try dedicated axelera.runtime loader
            if self._try_load_axelera_runtime():
                return

            # Generic fallback loader
            engine_cls, mod_path = self._find_axelera_engine_class()
            if engine_cls is not None:
                try:
                    print(f"[AXELERA SDK] Loading model {self.axm_path} via '{mod_path}.{getattr(engine_cls, '__name__', 'Engine')}' (Chip {self.chip_id}, {self.num_cores} cores)...")
                    
                    context_obj = self._get_or_create_context(mod_path)

                    path_str = str(self.axm_path)
                    path_bytes = str(self.axm_path).encode('utf-8')
                    path_obj = Path(self.axm_path)

                    attempts = []
                    if context_obj is not None:
                        attempts.extend([
                            lambda: engine_cls(path_str, context_obj),
                            lambda: engine_cls(path_str, context=context_obj),
                            lambda: engine_cls(context_obj, path_str),
                            lambda: engine_cls(context_obj, path_bytes),
                            lambda: engine_cls(context_obj, path_obj),
                            lambda: engine_cls(path_bytes, context=context_obj),
                            lambda: engine_cls(path_bytes, context_obj),
                        ])

                    attempts.extend([
                        lambda: engine_cls(path_str, chip_id=self.chip_id, num_cores=self.num_cores),
                        lambda: engine_cls(path_bytes, chip_id=self.chip_id, num_cores=self.num_cores),
                        lambda: engine_cls(path_str, chip_id=self.chip_id),
                        lambda: engine_cls(path_bytes, chip_id=self.chip_id),
                        lambda: engine_cls(path_str),
                        lambda: engine_cls(path_bytes),
                        lambda: engine_cls(path_obj)
                    ])

                    last_err = None
                    for attempt_fn in attempts:
                        try:
                            self.session = attempt_fn()
                            if self.session is not None:
                                break
                        except Exception as err:
                            last_err = err
                            continue

                    if self.session is not None:
                        self.backend = "axelera_voyager"
                        print(f"[AXELERA SDK SUCCESS] Model loaded on Metis AIPU via '{mod_path}'.")
                        return
                    else:
                        raise last_err if last_err else RuntimeError("Model initialization failed across all parameter signatures")

                except Exception as e:
                    print(f"[AXELERA SDK WARNING] Could not load .axm model on Metis AIPU: {e}")

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
            # Voyager / Axelera Runtime SDK inference call
            if hasattr(self.session, "run"):
                outputs = self.session.run(input_tensor)
            elif hasattr(self.session, "forward"):
                outputs = self.session.forward(input_tensor)
            elif hasattr(self.session, "predict"):
                outputs = self.session.predict(input_tensor)
            elif hasattr(self.session, "execute"):
                outputs = self.session.execute(input_tensor)
            elif callable(self.session):
                outputs = self.session(input_tensor)
            else:
                raise RuntimeError(f"Axelera session object '{type(self.session).__name__}' has no run/forward method")

            if not isinstance(outputs, list):
                outputs = [outputs]
            return outputs

        elif self.backend == "onnxruntime":
            if input_tensor.dtype != np.float32:
                input_tensor = input_tensor.astype(np.float32)
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            return outputs

        elif self.backend == "virtual":
            batch_size = input_tensor.shape[0] if len(input_tensor.shape) == 4 else 1
            dummy_output = np.zeros((batch_size, 84, 8400), dtype=np.float32)
            return [dummy_output]

        else:
            raise RuntimeError(f"Engine backend '{self.backend}' is not properly initialized.")

    def get_backend(self) -> str:
        return self.backend
