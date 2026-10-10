"""
🌾 UZHAVAN AI - Comprehensive Evaluation Framework & Experiment Tracking
Generates dialect-wise WER, CER, classification metrics, and experiment comparison reports.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def generate_dialect_evaluation_table(asr_metrics: Dict, dialect_metrics: Dict) -> str:
    """
    Generates markdown table for dialect-wise performance.
    """
    lines = [
        "| Dialect | WER | CER | Classification Acc |",
        "| :--- | :---: | :---: | :---: |"
    ]
    
    dialects = [
        ("Standard Tamil", "standard"),
        ("Kongu Tamil", "kongu"),
        ("Madurai Tamil", "madurai"),
        ("Tirunelveli Tamil", "tirunelveli"),
        ("Delta Tamil", "thanjavur"),
        ("Chennai Tamil", "chennai"),
        ("Code-Mixed", "code_mixed")
    ]
    
    dialect_asr = asr_metrics.get("dialect_wise", {})
    dialect_cls = dialect_metrics.get("per_class", {})
    
    for name, code in dialects:
        wer_val = dialect_asr.get(code, {}).get("wer", None)
        cer_val = dialect_asr.get(code, {}).get("cer", None)
        acc_val = dialect_cls.get(code, {}).get("accuracy", None)
        
        wer_str = f"{wer_val * 100:.2f}%" if wer_val is not None else "XX%"
        cer_str = f"{cer_val * 100:.2f}%" if cer_val is not None else "XX%"
        acc_str = f"{acc_val * 100:.2f}%" if acc_val is not None else "XX%"
        
        lines.append(f"| {name} | {wer_str} | {cer_str} | {acc_str} |")
        
    return "\n".join(lines)


def generate_experiment_report(experiments: List[Dict]) -> str:
    """
    Compares baseline vs dialect-trained vs agri-trained vs full model.
    """
    lines = [
        "# 🧪 UZHAVAN AI - Experiment Evaluation Report",
        "",
        "| Model | WER | CER | F1-Score | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    
    for exp in experiments:
        name = exp.get("name", "Model")
        wer_val = exp.get("wer", "XX")
        cer_val = exp.get("cer", "XX")
        f1_val = exp.get("f1", "XX")
        lat_val = exp.get("latency_ms", "XX")
        lines.append(f"| {name} | {wer_val} | {cer_val} | {f1_val} | {lat_val} |")
        
    return "\n".join(lines)
