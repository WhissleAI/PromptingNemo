"""SVD-based weight transfer from teacher to student for model distillation."""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def svd_compress_linear(weight: torch.Tensor, target_rows: int, target_cols: int) -> torch.Tensor:
    """Compress a weight matrix via truncated SVD.

    For a weight of shape [out, in, ...], compress to [target_rows, target_cols, ...].
    Extra dimensions (e.g. Conv1d kernel dim) are preserved.
    """
    orig_shape = weight.shape
    extra_dims = orig_shape[2:]

    if extra_dims:
        w2d = weight.reshape(orig_shape[0], -1)
    else:
        w2d = weight

    U, S, Vt = torch.linalg.svd(w2d, full_matrices=False)

    k = min(target_rows, target_cols, S.shape[0])
    U_k = U[:target_rows, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :target_cols]

    compressed = U_k @ torch.diag(S_k) @ Vt_k

    if extra_dims:
        in_times_extra = target_cols * int(torch.tensor(extra_dims).prod().item()) if extra_dims else target_cols
        if compressed.shape[1] < in_times_extra:
            padded = torch.zeros(target_rows, in_times_extra, device=weight.device, dtype=weight.dtype)
            padded[:, :compressed.shape[1]] = compressed
            compressed = padded
        compressed = compressed[:, :in_times_extra].reshape(target_rows, target_cols, *extra_dims)

    return compressed


def _transfer_conformer_layer(student_layer: nn.Module, teacher_layer: nn.Module,
                               student_dim: int, teacher_dim: int):
    """Transfer weights from one conformer layer, compressing dimensions via SVD."""
    student_sd = dict(student_layer.named_parameters())
    teacher_sd = dict(teacher_layer.named_parameters())

    for name, student_param in student_sd.items():
        if name not in teacher_sd:
            continue
        teacher_param = teacher_sd[name]
        sp = student_param.shape
        tp = teacher_param.shape

        if sp == tp:
            student_param.data.copy_(teacher_param.data)
        elif len(sp) == 1:
            if sp[0] <= tp[0]:
                student_param.data.copy_(teacher_param.data[:sp[0]])
        elif len(sp) >= 2 and sp[0] <= tp[0] and sp[1] <= tp[1]:
            compressed = svd_compress_linear(teacher_param.data, sp[0], sp[1])
            if compressed.shape == sp:
                student_param.data.copy_(compressed)
            elif all(s <= t for s, t in zip(sp, tp)):
                student_param.data.copy_(teacher_param.data[tuple(slice(0, s) for s in sp)])


def _get_encoder_layers(model) -> list:
    """Extract the list of conformer/convolution layers from a NeMo encoder."""
    encoder = model.encoder
    for attr in ['layers', 'conformer_layers']:
        candidate = getattr(encoder, attr, None)
        if candidate is not None and hasattr(candidate, '__len__'):
            return list(candidate)

    if hasattr(encoder, '_modules'):
        for key, module in encoder._modules.items():
            if hasattr(module, '__len__') and len(module) > 4:
                return list(module)

    raise RuntimeError("Could not locate encoder layers for weight transfer")


def build_layer_mapping(teacher_layers: int, student_layers: int) -> List[Tuple[int, int]]:
    """Build student→teacher layer index mapping (evenly spaced)."""
    mapping = []
    for s_idx in range(student_layers):
        t_idx = int(round(s_idx * (teacher_layers - 1) / max(student_layers - 1, 1)))
        mapping.append((s_idx, t_idx))
    return mapping


def build_first_n_layer_mapping(teacher_layers: int, student_layers: int) -> List[Tuple[int, int]]:
    """Map student layer i to teacher layer i (take first N consecutive layers)."""
    n = min(student_layers, teacher_layers)
    return [(i, i) for i in range(n)]


def init_student_from_teacher(student, teacher, layer_mapping: List[Tuple[int, int]] = None,
                              strategy: str = 'evenly_spaced'):
    """Initialize student encoder weights from teacher via SVD compression.

    Args:
        strategy: 'evenly_spaced' (sample uniformly across depth) or
                  'first_n' (take first N consecutive teacher layers).
    """
    student_layers = _get_encoder_layers(student)
    teacher_layers = _get_encoder_layers(teacher)

    student_dim = student.encoder._feat_out
    teacher_dim = teacher.encoder._feat_out

    if layer_mapping is None:
        if strategy == 'first_n':
            layer_mapping = build_first_n_layer_mapping(len(teacher_layers), len(student_layers))
        else:
            layer_mapping = build_layer_mapping(len(teacher_layers), len(student_layers))

    transferred = 0
    for s_idx, t_idx in layer_mapping:
        if s_idx < len(student_layers) and t_idx < len(teacher_layers):
            _transfer_conformer_layer(
                student_layers[s_idx], teacher_layers[t_idx],
                student_dim, teacher_dim,
            )
            transferred += 1

    # Transfer preprocessor weights (feature extraction) — same architecture, direct copy
    if hasattr(student, 'preprocessor') and hasattr(teacher, 'preprocessor'):
        student_pre_sd = student.preprocessor.state_dict()
        teacher_pre_sd = teacher.preprocessor.state_dict()
        compatible = {}
        for k in student_pre_sd:
            if k in teacher_pre_sd and student_pre_sd[k].shape == teacher_pre_sd[k].shape:
                compatible[k] = teacher_pre_sd[k]
        if compatible:
            student.preprocessor.load_state_dict(compatible, strict=False)

    logging.info(
        "Initialized %d/%d student encoder layers from teacher (d=%d→%d via SVD)",
        transferred, len(student_layers), teacher_dim, student_dim,
    )


def copy_decoder_weights(student, teacher):
    """Copy decoder weights from teacher to student (same vocabulary)."""
    student_dec = student.decoder.state_dict()
    teacher_dec = teacher.decoder.state_dict()

    compatible = {}
    for k in student_dec:
        if k in teacher_dec and student_dec[k].shape == teacher_dec[k].shape:
            compatible[k] = teacher_dec[k]

    if compatible:
        student.decoder.load_state_dict(compatible, strict=False)
        logging.info("Copied %d/%d decoder weight tensors from teacher", len(compatible), len(student_dec))
    else:
        logging.warning(
            "No compatible decoder weights found. Student decoder shape: %s, Teacher: %s",
            {k: v.shape for k, v in student_dec.items()},
            {k: v.shape for k, v in teacher_dec.items()},
        )
