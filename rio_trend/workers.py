"""Color functions for use with rio-mucho."""
import sys

from .operations import parse_operations, simple_atmo
import numpy as np
import pymannkendall as mk


# from .utils import to_math_type, scale_dtype

# Rio workers

@np.errstate(invalid='ignore')
def color_worker(srcs, window, ij, args):
    """A user function."""
    src = srcs[0]

    # mask = srcs[1].read(window=window) if srcs[1] else None

    arr = src.read(window=window)
    # arr = to_math_type(arr)
    i, j = ij
    h, w = src.shape
    if j == 0 and i % 2 == 0:
        print(f"{(window.row_off * 100 / h):.2f}%...", end='', flush=True)
    # for func in parse_operations(args["ops_string"]):
    #     arr = func(arr)

    out = np.full((3, *arr.shape[1:]), np.finfo(np.float32).min)

    for i, j in np.ndindex(arr.shape[1:]):
        pixel = arr[:, i, j]

        if np.array_equal(pixel, args['nodatavals']):
            continue

        # result = np.count_nonzero(pixel == args['nodata'])
        # out[2, i, j] = result
        # continue

        count_nodata = np.count_nonzero(pixel == args['nodata'])

        if count_nodata <= 2:
            pixel = pixel[pixel != args['nodata']]
        else:
            continue

        # if mask is None or mask[0, i, j] == 1:
        try:
            trend = mk.yue_wang_modification_test(pixel)
        except:
            pass
        else:
            out[0, i, j] = trend.slope
            out[1, i, j] = trend.p
            is_significant = trend.p <= 0.05
            # out[2, i, j] = np.digitize(trend.p, [0, 0.05]) * np.sign(trend.slope)
            out[2, i, j] = np.sign(trend.slope) if is_significant else 2

    return out.astype(np.float32)  # scale_dtype(arr, args["out_dtype"])
