"""fl_core/knowledge_compression — Compression utilities."""
from hfl_core.knowledge_compression.topk_sparsification import TopKCompressor
from hfl_core.knowledge_compression.int8_quantization import (
    quantize_tensor, dequantize_tensor, compute_payload_bits, SparseINT8Payload,
)
