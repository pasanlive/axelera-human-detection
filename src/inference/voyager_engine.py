"""
VoyagerEngine: Inference Abstraction Layer for Axelera Metis 111C NPU and Fallbacks.
"""

import os
import time
import ctypes
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

            # 1. Discover Context
            ctx = None
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
                return False

            path_str = str(self.axm_path)
            path_abs = os.path.abspath(path_str)
            model_obj = None

            # 2. Check for official Axelera Metis compiled_model/model.json structure first
            model_json_path = os.path.join(path_abs, "compiled_model", "model.json")
            if os.path.exists(model_json_path):
                comp_dir = os.path.dirname(model_json_path)
                for loader_fn in [
                    lambda: axr.axelera_load_model(comp_dir, "model.json"),
                    lambda: axr.graph_exec_load_model(comp_dir, "model.json"),
                ]:
                    try:
                        model_obj = loader_fn()
                        if model_obj is not None:
                            print(f"[AXELERA RUNTIME SUCCESS] Model '{Path(self.axm_path).name}' loaded on Metis AIPU ({self.num_cores} cores).")
                            self.session = model_obj
                            self.backend = "axelera_voyager"
                            return True
                    except Exception:
                        continue

            # 3. Fallback candidate search
            search_targets = []
            if os.path.exists(path_abs):
                if os.path.isdir(path_abs):
                    for f in os.listdir(path_abs):
                        search_targets.append((path_abs, f))
                else:
                    search_targets.append((os.path.dirname(path_abs), os.path.basename(path_abs)))
            else:
                parent_dir = os.path.dirname(path_abs)
                if os.path.exists(parent_dir):
                    for f in os.listdir(parent_dir):
                        search_targets.append((parent_dir, f))

            for search_dir, search_file in search_targets:
                target_full = os.path.join(search_dir, search_file)
                if not os.path.exists(target_full):
                    continue

                if os.path.isfile(target_full):
                    if not search_file.endswith('.json'):
                        continue
                    f_size = os.path.getsize(target_full)
                    if f_size == 0:
                        continue
                elif os.path.isdir(target_full):
                    for sub_f in os.listdir(target_full):
                        sub_target = (target_full, sub_f)
                        if sub_target not in search_targets:
                            search_targets.append(sub_target)
                    continue

                for loader_fn in [
                    lambda d=search_dir, f=search_file: axr.axelera_load_model(d, f),
                    lambda d=search_dir, f=search_file: axr.graph_exec_load_model(d, f),
                ]:
                    try:
                        model_obj = loader_fn()
                        if model_obj is not None:
                            print(f"[AXELERA RUNTIME SUCCESS] Model '{search_file}' loaded & instantiated on Metis AIPU ({self.num_cores} cores).")
                            self.session = model_obj
                            self.backend = "axelera_voyager"
                            return True
                    except Exception:
                        continue

            print(f"[AXELERA RUNTIME WARNING] Could not load .axm NPU model '{path_str}'. Fallback to ONNXRuntime will be engaged.")
            return False

        except ImportError:
            return False
        except Exception:
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
                for ctx_name in target_ctx_names:
                    if hasattr(mod, ctx_name):
                        factory = getattr(mod, ctx_name)
                        ctx_attempts = [
                            lambda f=factory: f(chip_id=self.chip_id),
                            lambda f=factory: f(self.chip_id),
                            lambda f=factory: f(device=self.chip_id),
                            lambda f=factory: f(chip=self.chip_id),
                            lambda f=factory: f(num_cores=self.num_cores),
                            lambda f=factory: f("metis-111c"),
                            lambda f=factory: f(),
                        ]
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
                    path_bytes = path_str.encode('utf-8')
                    c_path_p = ctypes.c_char_p(path_bytes)
                    c_path_buf = ctypes.create_string_buffer(path_bytes)
                    path_obj = Path(self.axm_path)

                    attempts = []
                    if context_obj is not None:
                        attempts.extend([
                            lambda: engine_cls(context_obj, c_path_p),
                            lambda: engine_cls(c_path_p, context_obj),
                            lambda: engine_cls(context_obj, c_path_buf),
                            lambda: engine_cls(c_path_buf, context_obj),
                            lambda: engine_cls(path_str, context_obj),
                            lambda: engine_cls(context_obj, path_str),
                        ])
                    else:
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
    def _prepare_input_tensor(self, input_tensor: np.ndarray) -> np.ndarray:
        """Adapts input tensor layout (NCHW->NHWC), dtype (float32->int8/uint8), and shape padding for Metis AIPU."""
        target_shape = None
        target_dtype = None

        if hasattr(self.session, "_input_infos"):
            try:
                infos = getattr(self.session, "_input_infos")
                info = infos[0] if isinstance(infos, (list, tuple)) else (infos.get(0) if isinstance(infos, dict) else None)
                if info is not None:
                    target_shape = getattr(info, "shape", None)
                    target_dtype = getattr(info, "dtype", None)
            except Exception:
                pass

        curr_tensor = input_tensor

        # 1. Adapt layout (NCHW [1, 3, H, W] -> NHWC [1, H, W, 3]) for Axelera Metis NPU
        if len(curr_tensor.shape) == 4 and curr_tensor.shape[1] in [1, 3, 4]:
            if target_shape is None or (len(target_shape) == 4 and target_shape[-1] in [1, 3, 4]):
                curr_tensor = np.transpose(curr_tensor, (0, 2, 3, 1))

        # 2. Adapt layout & padding if target_shape is known
        if target_shape is not None and tuple(curr_tensor.shape) != tuple(target_shape):
            # Adapt data type before padding
            if target_dtype is not None and curr_tensor.dtype != target_dtype:
                if target_dtype == np.int8:
                    if curr_tensor.max() <= 1.0:
                        curr_tensor = (curr_tensor * 255.0 - 128.0).clip(-128, 127).astype(np.int8)
                    else:
                        curr_tensor = (curr_tensor - 128.0).clip(-128, 127).astype(np.int8)
                elif target_dtype == np.uint8:
                    if curr_tensor.max() <= 1.0:
                        curr_tensor = (curr_tensor * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        curr_tensor = curr_tensor.clip(0, 255).astype(np.uint8)
                else:
                    curr_tensor = curr_tensor.astype(target_dtype)

            # Perform zero padding if dimensions differ
            if tuple(curr_tensor.shape) != tuple(target_shape):
                padded = np.zeros(target_shape, dtype=curr_tensor.dtype)
                if len(target_shape) == 4 and len(curr_tensor.shape) == 4:
                    n = min(curr_tensor.shape[0], target_shape[0])
                    d1 = min(curr_tensor.shape[1], target_shape[1])
                    d2 = min(curr_tensor.shape[2], target_shape[2])
                    d3 = min(curr_tensor.shape[3], target_shape[3])
                    padded[:n, :d1, :d2, :d3] = curr_tensor[:n, :d1, :d2, :d3]
                    curr_tensor = padded
                else:
                    curr_tensor = np.resize(curr_tensor, target_shape)
        else:
            # Adapt data type if shapes match
            if target_dtype is not None and curr_tensor.dtype != target_dtype:
                if target_dtype == np.int8:
                    if curr_tensor.max() <= 1.0:
                        curr_tensor = (curr_tensor * 255.0 - 128.0).clip(-128, 127).astype(np.int8)
                    else:
                        curr_tensor = (curr_tensor - 128.0).clip(-128, 127).astype(np.int8)
                elif target_dtype == np.uint8:
                    if curr_tensor.max() <= 1.0:
                        curr_tensor = (curr_tensor * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        curr_tensor = curr_tensor.clip(0, 255).astype(np.uint8)
                else:
                    curr_tensor = curr_tensor.astype(target_dtype)

        return curr_tensor

    def run(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Executes model inference on the given input tensor.
        :param input_tensor: Preprocessed numpy array (e.g., shape [1, 3, H, W] float32 or uint8)
        :return: List of output numpy tensors from the model.
        """
        t0 = time.time()
        
        if self.backend == "axelera_voyager":
            # Axelera Metis NPU GraphExecutor / Voyager SDK inference execution
            if hasattr(self.session, "set_input") and hasattr(self.session, "run"):
                input_tensor = self._prepare_input_tensor(input_tensor)

                try:
                    self.session.set_input(0, input_tensor)
                except (AssertionError, TypeError) as ae:
                    err_str = str(ae)
                    if "Expected input shape to be" in err_str:
                        import re
                        m = re.search(r"Expected input shape to be \(([^)]+)\)", err_str)
                        if m:
                            shape_dims = tuple(int(x.strip()) for x in m.group(1).split(','))
                            if len(input_tensor.shape) == 4 and input_tensor.shape[1] in [1, 3, 4] and shape_dims[-1] in [1, 3, 4]:
                                input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))
                            padded = np.zeros(shape_dims, dtype=input_tensor.dtype)
                            n = min(input_tensor.shape[0], shape_dims[0])
                            d1 = min(input_tensor.shape[1], shape_dims[1])
                            d2 = min(input_tensor.shape[2], shape_dims[2])
                            d3 = min(input_tensor.shape[3], shape_dims[3])
                            padded[:n, :d1, :d2, :d3] = input_tensor[:n, :d1, :d2, :d3]
                            input_tensor = padded
                            self.session.set_input(0, input_tensor)
                        else:
                            raise ae
                    elif "int8" in err_str.lower():
                        input_tensor = (input_tensor * 255.0 - 128.0).clip(-128, 127).astype(np.int8) if input_tensor.max() <= 1.0 else (input_tensor - 128.0).clip(-128, 127).astype(np.int8)
                        self.session.set_input(0, input_tensor)
                    elif "uint8" in err_str.lower():
                        input_tensor = (input_tensor * 255.0).clip(0, 255).astype(np.uint8) if input_tensor.max() <= 1.0 else input_tensor.clip(0, 255).astype(np.uint8)
                        self.session.set_input(0, input_tensor)
                    else:
                        raise ae

                self.session.run()

                num_outputs = 1
                if hasattr(self.session, "get_num_outputs"):
                    try:
                        num_outputs = self.session.get_num_outputs()
                    except Exception:
                        num_outputs = 1

                outputs = []
                for i in range(num_outputs):
                    try:
                        out = self.session.get_output(i)
                        if hasattr(out, "asnumpy"):
                            out = out.asnumpy()
                        elif hasattr(out, "numpy"):
                            out = out.numpy()
                        outputs.append(np.array(out))
                    except Exception:
                        pass
                if outputs:
                    return outputs
            elif hasattr(self.session, "run"):
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
                raise RuntimeError(f"Axelera session object '{type(self.session).__name__}' has no recognized execution method")

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
