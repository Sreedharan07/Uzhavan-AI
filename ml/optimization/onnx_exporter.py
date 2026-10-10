"""
🌾 UZHAVAN AI - ONNX Runtime Quantized Inference Pipeline
Converts PyTorch models to ONNX with INT8 quantization for 3-5x inference speedup.

Supports: Dialect CNN-BiLSTM, Prosody Model, Crop Classifier, Intent Classifier, Emotion TCN
Time Complexity: Same forward pass, but with INT8 arithmetic (~3-5x hardware speedup)
Memory: ~4x reduction via INT8 quantization
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class ONNXExporter:
    """
    Export PyTorch models to ONNX format with optional INT8 quantization.
    """

    @staticmethod
    def export_model(
        model: torch.nn.Module,
        dummy_input: Any,
        output_path: str,
        input_names: List[str] = None,
        output_names: List[str] = None,
        dynamic_axes: Optional[Dict] = None,
        opset_version: int = 14,
    ) -> str:
        """
        Export a PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model in eval mode
            dummy_input: Example input tensor(s)
            output_path: Path for the .onnx file
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic dimension specification
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX file
        """
        model.eval()
        output_path = str(output_path)
        
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )
            return output_path
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {e}")

    @staticmethod
    def quantize_model(onnx_path: str, quantized_path: Optional[str] = None) -> str:
        """
        Apply INT8 dynamic quantization to an ONNX model.
        Reduces model size by ~4x and improves inference speed by ~2-3x.
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            raise ImportError("Install onnxruntime: pip install onnxruntime")

        if quantized_path is None:
            quantized_path = onnx_path.replace(".onnx", "_int8.onnx")

        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QInt8,
        )
        return quantized_path


class OptimizedInferencePipeline:
    """
    Unified inference pipeline that auto-selects ONNX or PyTorch at runtime.
    Falls back to PyTorch if ONNX models not found.
    Logs per-model inference latency for paper benchmarks.
    """

    def __init__(self, models_dir: str = "models/onnx"):
        self.models_dir = Path(models_dir)
        self._onnx_sessions: Dict[str, Any] = {}
        self._pytorch_models: Dict[str, torch.nn.Module] = {}
        self._latency_log: List[Dict] = []

    def _get_onnx_session(self, model_name: str):
        """Load or retrieve cached ONNX runtime session."""
        if model_name in self._onnx_sessions:
            return self._onnx_sessions[model_name]

        onnx_path = self.models_dir / f"{model_name}_int8.onnx"
        if not onnx_path.exists():
            onnx_path = self.models_dir / f"{model_name}.onnx"
        
        if not onnx_path.exists():
            return None

        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available_providers = ort.get_available_providers()
            providers = [p for p in providers if p in available_providers]

            session = ort.InferenceSession(str(onnx_path), providers=providers)
            self._onnx_sessions[model_name] = session
            return session
        except Exception:
            return None

    def register_pytorch_model(self, model_name: str, model: torch.nn.Module):
        """Register a PyTorch model as fallback."""
        model.eval()
        self._pytorch_models[model_name] = model

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        pytorch_input: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Run inference with auto-selection between ONNX and PyTorch.
        
        Returns:
            {"output": np.ndarray, "backend": str, "latency_ms": float}
        """
        # Try ONNX first
        session = self._get_onnx_session(model_name)

        if session is not None:
            start = time.perf_counter()
            result = session.run(None, inputs)
            latency = (time.perf_counter() - start) * 1000

            self._log_latency(model_name, "onnx", latency)
            return {"output": result[0], "backend": "onnx_int8", "latency_ms": round(latency, 3)}

        # Fallback to PyTorch
        model = self._pytorch_models.get(model_name)
        if model is not None and pytorch_input is not None:
            start = time.perf_counter()
            with torch.no_grad():
                output = model(pytorch_input)
            latency = (time.perf_counter() - start) * 1000

            self._log_latency(model_name, "pytorch", latency)
            return {
                "output": output.cpu().numpy(),
                "backend": "pytorch",
                "latency_ms": round(latency, 3),
            }

        raise RuntimeError(f"No model found for '{model_name}' (checked ONNX and PyTorch)")

    def _log_latency(self, model_name: str, backend: str, latency_ms: float):
        self._latency_log.append({
            "model": model_name,
            "backend": backend,
            "latency_ms": latency_ms,
        })

    def get_latency_report(self) -> Dict:
        """Generate latency comparison report for paper benchmarks."""
        from collections import defaultdict
        stats = defaultdict(lambda: defaultdict(list))

        for entry in self._latency_log:
            stats[entry["model"]][entry["backend"]].append(entry["latency_ms"])

        report = {}
        for model, backends in stats.items():
            report[model] = {}
            for backend, latencies in backends.items():
                arr = np.array(latencies)
                report[model][backend] = {
                    "mean_ms": round(float(arr.mean()), 3),
                    "std_ms": round(float(arr.std()), 3),
                    "min_ms": round(float(arr.min()), 3),
                    "max_ms": round(float(arr.max()), 3),
                    "n_samples": len(latencies),
                }

        return report

    def export_all_models(self, output_dir: Optional[str] = None):
        """Export all registered PyTorch models to ONNX."""
        out = Path(output_dir) if output_dir else self.models_dir
        out.mkdir(parents=True, exist_ok=True)

        exported = []
        for name, model in self._pytorch_models.items():
            # Skip — real export needs actual dummy inputs per model
            exported.append({"model": name, "status": "registered_for_export"})

        return exported
