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
            out[2, i, j] = -9999
        else:
            out[0, i, j] = trend.slope
            out[1, i, j] = trend.p

            is_significant = bool(trend.p <= 0.05)
            sign = int(np.sign(trend.slope))

            if sign == 0:
                out[2, i, j] = 0
            elif is_significant:
                out[2, i, j] = sign
            elif not is_significant:
                out[2, i, j] = sign * 2
            else:
                out[2, i, j] = -9999

            # out[2, i, j] = np.sign(trend.slope) if is_significant else 2
            # result = classify(is_significant, sign)
            # if result == -9999:
                print(-9999, is_significant, sign)

            # out[2, i, j] = np.digitize(trend.p, [0, 0.05]) * np.sign(trend.slope)


    return out.astype(np.float32)  # scale_dtype(arr, args["out_dtype"])

'''
def classify(is_significant: bool, sign: int) -> int:
    # print(type(is_significant), type(sign))
    match [is_significant, sign]:
        case [True, 1]:
            return 1
        case [True, -1]:
            return -1
        case [False, 1]:
            return 2
        case [False, -1]:
            return -2
        case [_, 0]:
            return 0
        case _:
            return -9999
'''