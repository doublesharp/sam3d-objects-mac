"""Exact lexicographic integer row uniqueness for CPU mesh indices."""
import torch


def unique_rows(rows, return_inverse=False, return_counts=False):
    fallback = lambda: torch.unique(rows, dim=0, return_inverse=return_inverse,
                                   return_counts=return_counts)
    if (rows.device.type != 'cpu' or rows.dtype not in (torch.int32, torch.int64)
            or rows.ndim != 2 or not rows.numel()):
        return fallback()
    minima = rows.amin(0).tolist()
    maxima = rows.amax(0).tolist()
    widths = [hi-lo+1 for lo, hi in zip(minima, maxima)]
    product = 1
    for width in widths:
        product *= width
        if product > torch.iinfo(torch.int64).max:
            return fallback()
    keys = torch.zeros(len(rows), dtype=torch.int64)
    # Python range checks precede subtraction so extrema cannot overflow.
    for column, (low, width) in enumerate(zip(minima, widths)):
        keys = keys * width + (rows[:, column].to(torch.int64) - low)
    unique, inverse, counts = torch.unique(keys, sorted=True, return_inverse=True,
                                         return_counts=True)
    decoded = torch.empty((len(unique), rows.shape[1]), dtype=rows.dtype)
    remainder = unique
    for column in range(rows.shape[1]-1, -1, -1):
        decoded[:, column] = remainder.remainder(widths[column]) + minima[column]
        remainder = remainder.div(widths[column], rounding_mode='floor')
    result = [decoded]
    if return_inverse: result.append(inverse)
    if return_counts: result.append(counts)
    return tuple(result) if len(result) > 1 else decoded
