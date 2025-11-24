"""
Compatibility patches that are applied automatically when Python starts.

This module ensures that older libraries which still pass integer GPU indices
to `torch.nn.parallel._functions._get_stream` keep working with newer
PyTorch releases (>=2.1) where `_get_stream` expects a `torch.device`.
"""

from __future__ import annotations

def _patch_torch_get_stream():
    try:
        import torch  # type: ignore[import]
        from torch.nn.parallel import _functions as torch_parallel_functions  # type: ignore[attr-defined]
    except Exception:
        return

    _original_get_stream = getattr(torch_parallel_functions, "_get_stream", None)
    if not callable(_original_get_stream):
        return

    def _patched_get_stream(device):
        """Accept both integers and torch.device instances."""
        if not isinstance(device, torch.device):
            if device == -1 or (isinstance(device, str) and device.lower() == "cpu"):
                device = torch.device("cpu")
            else:
                device = torch.device("cuda", device)
        return _original_get_stream(device)

    try:
        _original_get_stream(torch.device("cuda", 0))
    except Exception:
        return
    else:
        torch_parallel_functions._get_stream = _patched_get_stream


_patch_torch_get_stream()
