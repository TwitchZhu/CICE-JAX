"""Defines variable precision for all common data types.
Code originally based on kinds_mod.F in POP.

author for JAX version: Xinyuan Zhu
XZ: converted from F90 to Python

2020: TC added support for NO_I8 and NO_R16
2006: ECH converted to free source form (F90)
author: Elizabeth C. Hunke and William H. Lipscomb, LANL
"""

import numpy as np

char_len = 80
char_len_long = 256
log_kind = np.bool_

# NumPy's int8/16/32/64 and IEEE float32/float64 have the same widths on every
# platform, so these kinds do not change with the compiler. The commented
# block below is kept only for reference: _require_int_kind corresponds to
# Fortran selected_int_kind, and _require_real_kind to selected_real_kind.
int_kind = np.int32
int8_kind = np.int64
real_kind = np.float32
dbl_kind = np.float64
# NO_I8 and NO_R16 are not supported. r16_kind is omitted; see README.md.

# def selected_int_kind(r):
#     """Smallest integer dtype that represents integers of r decimal digits.
#
#     Returns -1 when this build has no such integer, as Fortran does.
#     """
#     chosen = None
#     chosen_digits = None
#     for bits in (8, 16, 32, 64):
#         dtype = getattr(np, f"int{bits}")
#         digits = int(math.floor(math.log10(np.iinfo(dtype).max)))
#         if digits >= r and (chosen_digits is None or digits < chosen_digits):
#             chosen = dtype
#             chosen_digits = digits
#     if chosen is None:
#         return -1
#     return chosen
#
#
# def selected_real_kind(p, r=None):
#     """Smallest real dtype with decimal precision p and exponent range r.
#
#     Fortran result codes: -1 precision unavailable, -2 range unavailable,
#     -3 neither available.
#     """
#     candidates = [np.float16, np.float32, np.float64]
#     if np.dtype(np.longdouble) != np.dtype(np.float64):
#         candidates.append(np.longdouble)
#     float128 = getattr(np, "float128", None)
#     if float128 is not None and all(np.dtype(float128) != np.dtype(item) for item in candidates):
#         candidates.append(float128)
#
#     chosen = None
#     chosen_precision = None
#     precision_ok_somewhere = False
#     range_ok_somewhere = r is None
#     for dtype in candidates:
#         info = np.finfo(dtype)
#         precision_ok = info.precision >= p
#         decimal_range = int(math.floor(math.log10(info.max)))
#         range_ok = r is None or decimal_range >= r
#         precision_ok_somewhere = precision_ok_somewhere or precision_ok
#         range_ok_somewhere = range_ok_somewhere or range_ok
#         if precision_ok and range_ok:
#             if chosen_precision is None or info.precision < chosen_precision:
#                 chosen = dtype
#                 chosen_precision = info.precision
#     if chosen is not None:
#         return chosen
#     if not precision_ok_somewhere and not range_ok_somewhere:
#         return -3
#     if not precision_ok_somewhere:
#         return -1
#     return -2
#
#
# def _require_int_kind(r):
#     """Corresponds to Fortran selected_int_kind. Kept for reference only."""
#     kind = selected_int_kind(r)
#     if kind == -1:
#         raise RuntimeError(f"selected_int_kind({r}) returned -1")
#     return kind
#
#
# def _require_real_kind(p, r=None):
#     """Corresponds to Fortran selected_real_kind. Kept for reference only."""
#     kind = selected_real_kind(p, r)
#     if kind in (-1, -2, -3):
#         raise RuntimeError(f"selected_real_kind({p}, {r}) returned {kind}")
#     return kind
#
#
# int_kind = _require_int_kind(6)  # np.int32
# int8_kind = _require_int_kind(13)  # np.int64
# real_kind = _require_real_kind(6)  # np.float32
# dbl_kind = _require_real_kind(13)  # np.float64
