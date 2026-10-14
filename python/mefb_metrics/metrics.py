"""MEFB evaluation metrics ported from MATLAB to NumPy/SciPy.

The functions in this module follow the MATLAB implementations from
https://github.com/xingchenzhang/MEFB as closely as practical, including the
channel-dispatch behavior used by the original ``metrics*.m`` wrappers.
Inputs are expected to be image-like NumPy arrays in the 0-255 range.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft
from scipy import ndimage, signal


METRIC_NAMES = (
    "Cross_entropy",
    "Entropy",
    "FMI",
    "NMI",
    "Psnr",
    "Qncie",
    "TE",
    "Avg_gradient",
    "Edge_intensity",
    "Qabf",
    "Qp",
    "SD",
    "Spatial_frequency",
    "Qc",
    "Qw",
    "Qy",
    "MEF_SSIM",
    "Qcb",
    "Qcv",
    "VIF",
)


def compute_metrics(
    img1: np.ndarray,
    img2: np.ndarray,
    fused: np.ndarray,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute multiple MEFB metrics."""
    names = METRIC_NAMES if metrics is None else tuple(metrics)
    return {name: compute_metric(name, img1, img2, fused) for name in names}


def compute_metric(name: str, img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    """Compute one MEFB metric by name."""
    try:
        func = _METRICS[name.lower()]
    except KeyError as exc:
        valid = ", ".join(METRIC_NAMES)
        raise ValueError(f"unknown metric {name!r}; valid metrics: {valid}") from exc
    return float(func(img1, img2, fused))


def _as_float(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def _as_numeric(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
    return np.asarray(arr, dtype=dtype)


def _as_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return arr[..., :3] if arr.shape[2] != 1 else arr[..., 0]
    raise ValueError(f"expected an HxW or HxWxC image, got shape {arr.shape}")


def _channels(arr: np.ndarray) -> int:
    arr = _as_image(arr)
    return 1 if arr.ndim == 2 else arr.shape[2]


def _rgb2gray(arr: np.ndarray) -> np.ndarray:
    arr = _as_image(arr)
    if arr.ndim == 2:
        return arr
    out = (
        _as_float(arr[..., 0]) * 0.298936
        + _as_float(arr[..., 1]) * 0.587043
        + _as_float(arr[..., 2]) * 0.114021
    )
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = np.floor(out + 0.5).clip(info.min, info.max).astype(arr.dtype)
    return out


def _maybe_gray(arr: np.ndarray) -> np.ndarray:
    return _rgb2gray(arr) if _as_image(arr).ndim == 3 else _as_image(arr)


def _dispatch_channels(
    img1: np.ndarray,
    img2: np.ndarray,
    fused: np.ndarray,
    func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> float:
    img1 = _as_image(img1)
    img2 = _as_image(img2)
    fused = _as_image(fused)
    b1 = _channels(img1)
    b2 = _channels(img2)
    b = _channels(fused)

    if b1 == 1 and b2 == 1 and b == 3:
        fused = fused[..., 0]
        b = 1

    if b == 1:
        return float(func(img1, img2, fused))

    vals = []
    for k in range(b):
        if b1 == 1:
            vals.append(func(img1, img2 if b2 == 1 else img2[..., k], fused[..., k]))
        else:
            vals.append(func(img1[..., k], img2[..., k], fused[..., k]))
    return float(np.mean(vals))


def _floor_u8_indices(arr: np.ndarray) -> np.ndarray:
    return np.floor(_as_float(arr)).astype(np.int64).clip(0, 255)


def _hist256(arr: np.ndarray) -> np.ndarray:
    return np.bincount(_floor_u8_indices(arr).ravel(), minlength=256).astype(np.float64)[:256]


def _normalize1(data: np.ndarray) -> np.ndarray:
    data = _as_float(data)
    da = np.max(data)
    xiao = np.min(data)
    if da == 0 and xiao == 0:
        return data.copy()
    return np.floor((data - xiao) / (da - xiao) * 255.0 + 0.5)


def _fspecial_gaussian(size: int | tuple[int, int], sigma: float) -> np.ndarray:
    if isinstance(size, int):
        rows = cols = size
    else:
        rows, cols = size
    y, x = np.mgrid[-(rows - 1) / 2 : (rows - 1) / 2 + 1, -(cols - 1) / 2 : (cols - 1) / 2 + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    tiny = np.finfo(np.float64).eps * h.max()
    h[h < tiny] = 0
    total = h.sum()
    return h / total if total else h


def _fspecial_sobel() -> np.ndarray:
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)


def _conv2_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return ndimage.convolve(_as_numeric(image), _as_numeric(kernel), mode="constant", cval=0.0)


def _filter2(image: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    if mode == "same":
        return ndimage.correlate(_as_numeric(image), _as_numeric(kernel), mode="constant", cval=0.0)
    return signal.correlate2d(_as_numeric(image), _as_numeric(kernel), mode=mode, boundary="fill", fillvalue=0)


def _filter2_replicate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return ndimage.correlate(_as_numeric(image), _as_numeric(kernel), mode="nearest")


def _imfilter_symmetric_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = _as_numeric(image)
    kernel = _as_numeric(kernel)
    pad_rows = kernel.shape[0] // 2
    pad_cols = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="symmetric")
    if kernel.shape[0] % 2 == 0:
        padded = padded[1:, :]
    if kernel.shape[1] % 2 == 0:
        padded = padded[:, 1:]
    return signal.correlate2d(padded, kernel, mode="valid", boundary="fill", fillvalue=0)


def _mean2(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def _entropy(values: np.ndarray) -> float:
    hist = _hist256(values)
    p = hist / values.size
    nz = p != 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def metrics_entropy(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    return _dispatch_channels(img1, img2, fused, lambda _a, _b, f: _entropy(_maybe_gray(f)))


def _cross_entropy_pair(img: np.ndarray, fused: np.ndarray) -> float:
    f1 = _maybe_gray(img)
    f2 = _maybe_gray(fused)
    h1 = _hist256(f1)
    h2 = _hist256(f2)
    p1 = h1 / f1.size
    p2 = h2 / f1.size
    idx = (p1 != 0) & (p2 != 0)
    return float(np.sum(p1[idx] * np.log2(p1[idx] / p2[idx])))


def metrics_cross_entropy(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        return (_cross_entropy_pair(a, f) + _cross_entropy_pair(b, f)) / 2.0

    return _dispatch_channels(img1, img2, fused, one)


def _mse_mefb(a: np.ndarray, b: np.ndarray) -> float:
    a = _as_float(_maybe_gray(a))
    b = _as_float(_maybe_gray(b))
    m, n = a.shape
    return float(np.sqrt(np.sum((a - b) ** 2)) / (m * n))


def metrics_psnr(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        mes = (_mse_mefb(a, f) + _mse_mefb(b, f)) / 2.0
        return 20.0 * np.log10(255.0 / np.sqrt(mes))

    return _dispatch_channels(img1, img2, fused, one)


def _mutual_info(im1: np.ndarray, im2: np.ndarray) -> tuple[float, float, float, float]:
    x = _floor_u8_indices(im1).ravel()
    y = _floor_u8_indices(im2).ravel()
    h = np.bincount(x * 256 + y, minlength=256 * 256).reshape(256, 256).astype(np.float64)
    h /= np.sum(h)
    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)
    hx = -np.sum(im1_marg * np.log2(im1_marg + (im1_marg == 0)))
    hy = -np.sum(im2_marg * np.log2(im2_marg + (im2_marg == 0)))
    hxy = -np.sum(h * np.log2(h + (h == 0)))
    mi = hx + hy - hxy
    return float(mi), float(hxy), float(hx), float(hy)


def metrics_nmi(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        a = _normalize1(a)
        b = _normalize1(b)
        f = _normalize1(f)
        i_fx, _h_xf, h_x, h_f1 = _mutual_info(a, f)
        i_fy, _h_yf, h_y, h_f2 = _mutual_info(b, f)
        return 2.0 * (i_fx / (h_f1 + h_x) + i_fy / (h_f2 + h_y))

    return _dispatch_channels(img1, img2, fused, one)


def _tsallis(im1: np.ndarray, im2: np.ndarray, q: float) -> float:
    x = _floor_u8_indices(im1).ravel()
    y = _floor_u8_indices(im2).ravel()
    h = np.bincount(x * 256 + y, minlength=256 * 256).reshape(256, 256).astype(np.float64)
    h /= np.sum(h)
    # MEFB's MATLAB code names these marginal sums in a swapped way and then
    # indexes them as im1_marg(i) * im2_marg(j) for h(i,j). Keep that exact
    # ordering so TE matches the reference implementation.
    p1 = np.sum(h, axis=0)
    p2 = np.sum(h, axis=1)
    denom = np.outer(p1, p2) ** (q - 1.0)
    mask = denom != 0
    result = np.sum((h[mask] ** q) / denom[mask])
    return float((1.0 - result) / (1.0 - q))


def metrics_te(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        a = _normalize1(a)
        b = _normalize1(b)
        f = _normalize1(f)
        return _tsallis(a, f, 1.85) + _tsallis(b, f, 1.85)

    return _dispatch_channels(img1, img2, fused, one)


def metrics_qncie(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def ncc(a: np.ndarray, b: np.ndarray) -> float:
        mi, hxy, hx, hy = _mutual_info(a, b)
        return hx / 8.0 + hy / 8.0 - hxy / 8.0

    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        a = _normalize1(a)
        b = _normalize1(b)
        f = _normalize1(f)
        r = np.array(
            [[1.0, ncc(a, b), ncc(a, f)], [ncc(a, b), 1.0, ncc(b, f)], [ncc(a, f), ncc(b, f), 1.0]]
        )
        eig = np.linalg.eigvals(r).real
        hr = -np.sum(eig * np.log2(eig / 3.0) / 3.0) / 8.0
        return 1.0 - hr

    return _dispatch_channels(img1, img2, fused, one)


def metrics_avg_gradient(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    fused = _as_float(_as_image(fused))
    if _channels(img1) == 1 and _channels(img2) == 1 and _channels(fused) == 3:
        fused = _rgb2gray(fused)
    if fused.ndim == 2:
        fused = fused[..., None]
    r, c, b = fused.shape
    vals = []
    for k in range(b):
        gy, gx = np.gradient(fused[..., k], 1.0, 1.0)
        vals.append(np.sum(np.sqrt((gx * gx + gy * gy) / 2.0)) / ((r - 1) * (c - 1)))
    return float(np.mean(vals))


def metrics_edge_intensity(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    fused = _as_float(_as_image(fused))
    if _channels(img1) == 1 and _channels(img2) == 1 and _channels(fused) == 3:
        fused = fused[..., 0]
    if fused.ndim == 2:
        fused = fused[..., None]
    w = _fspecial_sobel()
    gx = np.stack([_filter2_replicate(fused[..., k], w) for k in range(fused.shape[2])], axis=2)
    gy = np.stack([_filter2_replicate(fused[..., k], w.T) for k in range(fused.shape[2])], axis=2)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def metrics_sd(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(_a: np.ndarray, _b: np.ndarray, f: np.ndarray) -> float:
        f = _as_float(_maybe_gray(f))
        mu = np.mean(f)
        return float(np.sqrt(np.sum((f - mu) ** 2) / f.size))

    return _dispatch_channels(img1, img2, fused, one)


def metrics_spatial_frequency(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(_a: np.ndarray, _b: np.ndarray, f: np.ndarray) -> float:
        f = _as_float(f)
        m, n = f.shape
        rf = np.sum((f[:, 1:] - f[:, :-1]) ** 2) / (m * n)
        cf = np.sum((f[1:, :] - f[:-1, :]) ** 2) / (m * n)
        return float(np.sqrt(rf + cf))

    return _dispatch_channels(img1, img2, fused, one)


def metrics_qabf(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    return _dispatch_channels(img1, img2, fused, _qabf)


def _qabf(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    lval, tg, kg, dg, ta, ka, da = 1.0, 0.9994, -15.0, 0.5, 0.9879, -22.0, 0.8
    del lval
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    pa = 255.0 * _as_float(_maybe_gray(_as_float(a)))
    pb = 255.0 * _as_float(_maybe_gray(_as_float(b)))
    pf = 255.0 * _as_float(_maybe_gray(_as_float(f)))

    sax = _conv2_same(pa, h3)
    say = _conv2_same(pa, h1)
    sbx = _conv2_same(pb, h3)
    sby = _conv2_same(pb, h1)
    sfx = _conv2_same(pf, h3)
    sfy = _conv2_same(pf, h1)

    ga = np.sqrt(sax * sax + say * say)
    gb = np.sqrt(sbx * sbx + sby * sby)
    gf = np.sqrt(sfx * sfx + sfy * sfy)
    aa = np.full_like(sax, np.pi / 2.0)
    ab = np.full_like(sbx, np.pi / 2.0)
    af = np.full_like(sfx, np.pi / 2.0)
    np.divide(say, sax, out=aa, where=sax != 0)
    np.divide(sby, sbx, out=ab, where=sbx != 0)
    np.divide(sfy, sfx, out=af, where=sfx != 0)
    aa = np.where(sax == 0, np.pi / 2.0, np.arctan(aa))
    ab = np.where(sbx == 0, np.pi / 2.0, np.arctan(ab))
    af = np.where(sfx == 0, np.pi / 2.0, np.arctan(af))

    gaf = np.empty_like(ga)
    gbf = np.empty_like(gb)
    np.divide(gf, ga, out=gaf, where=ga > gf)
    np.divide(ga, gf, out=gaf, where=(ga < gf) & (gf != 0))
    gaf[ga == gf] = gf[ga == gf]
    np.divide(gf, gb, out=gbf, where=gb > gf)
    np.divide(gb, gf, out=gbf, where=(gb < gf) & (gf != 0))
    gbf[gb == gf] = gf[gb == gf]
    aaf = 1.0 - np.abs(aa - af) / (np.pi / 2.0)
    abf = 1.0 - np.abs(ab - af) / (np.pi / 2.0)
    qaf = tg / (1.0 + np.exp(kg * (gaf - dg))) * ta / (1.0 + np.exp(ka * (aaf - da)))
    qbf = tg / (1.0 + np.exp(kg * (gbf - dg))) * ta / (1.0 + np.exp(ka * (abf - da)))
    return float(np.sum(qaf * ga + qbf * gb) / np.sum(ga + gb))


def _ssim_index(
    img1: np.ndarray,
    img2: np.ndarray,
    k: tuple[float, float] = (0.01, 0.03),
    window: np.ndarray | None = None,
    dynamic_range: float = 255.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    img1 = _as_float(img1)
    img2 = _as_float(img2)
    if img1.shape != img2.shape:
        return -np.inf, np.array(-np.inf), np.array(-np.inf), np.array(-np.inf)
    if window is None:
        if img1.shape[0] < 11 or img1.shape[1] < 11:
            return -np.inf, np.array(-np.inf), np.array(-np.inf), np.array(-np.inf)
        window = _fspecial_gaussian(11, 1.5)
    c1 = (k[0] * dynamic_range) ** 2
    c2 = (k[1] * dynamic_range) ** 2
    window = window / np.sum(window)
    mu1 = _filter2(img1, window, "valid")
    mu2 = _filter2(img2, window, "valid")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _filter2(img1 * img1, window, "valid") - mu1_sq
    sigma2_sq = _filter2(img2 * img2, window, "valid") - mu2_sq
    sigma12 = _filter2(img1 * img2, window, "valid") - mu1_mu2
    if c1 > 0 and c2 > 0:
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
    else:
        numerator1 = 2 * mu1_mu2 + c1
        numerator2 = 2 * sigma12 + c2
        denominator1 = mu1_sq + mu2_sq + c1
        denominator2 = sigma1_sq + sigma2_sq + c2
        ssim_map = np.ones_like(mu1)
        idx = denominator1 * denominator2 > 0
        ssim_map[idx] = (numerator1[idx] * numerator2[idx]) / (denominator1[idx] * denominator2[idx])
        idx = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[idx] = numerator1[idx] / denominator1[idx]
    return _mean2(ssim_map), ssim_map, sigma1_sq, sigma2_sq


def _ssim_yang(
    img1: np.ndarray, img2: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img1 = _as_float(img1)
    img2 = _as_float(img2)
    if img1.shape[0] < 11 or img1.shape[1] < 11:
        return -np.inf, np.array(-np.inf), np.array(-np.inf), np.array(-np.inf), np.array(-np.inf)
    window = _fspecial_gaussian(7, 1.5)
    mu1 = _filter2(img1, window, "valid")
    mu2 = _filter2(img2, window, "valid")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _filter2(img1 * img1, window, "valid") - mu1_sq
    sigma2_sq = _filter2(img2 * img2, window, "valid") - mu2_sq
    sigma12 = _filter2(img1 * img2, window, "valid") - mu1_mu2
    c1 = 2e-16
    c2 = 2e-16
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return _mean2(ssim_map), ssim_map, sigma1_sq, sigma2_sq, sigma12


def metrics_qc(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        a = _as_float(_maybe_gray(a))
        b = _as_float(_maybe_gray(b))
        f = _as_float(_maybe_gray(f))
        _, ssim_map2, *_rest2, sigma_xf = _ssim_yang(a, f)
        _, ssim_map3, *_rest3, sigma_yf = _ssim_yang(b, f)
        sim_xyf = sigma_xf / (sigma_xf + sigma_yf)
        sim = sim_xyf * ssim_map2 + (1.0 - sim_xyf) * ssim_map3
        sim = np.where(sim_xyf < 0, 0, sim)
        sim = np.where(sim_xyf > 1, 1, sim)
        return float(np.mean(sim[~np.isnan(sim)]))

    return _dispatch_channels(img1, img2, fused, one)


def metrics_qw(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        a = _as_float(_maybe_gray(a))
        b = _as_float(_maybe_gray(b))
        f = _as_float(_maybe_gray(f))
        flt1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
        flt2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)
        ff = np.sqrt(_filter2(f, flt1, "same") ** 2 + _filter2(f, flt2, "same") ** 2)
        af = np.sqrt(_filter2(a, flt1, "same") ** 2 + _filter2(a, flt2, "same") ** 2)
        bf = np.sqrt(_filter2(b, flt1, "same") ** 2 + _filter2(b, flt2, "same") ** 2)

        _, _, sigma1_sq, sigma2_sq = _ssim_index(af, bf)
        buffer = sigma1_sq + sigma2_sq
        sigma1_sq = sigma1_sq + (buffer == 0) * 0.5
        sigma2_sq = sigma2_sq + (buffer == 0) * 0.5
        ramda = sigma1_sq / (sigma1_sq + sigma2_sq)
        _, ssim_map1, _, _ = _ssim_index(ff, af)
        _, ssim_map2, _, _ = _ssim_index(ff, bf)
        cw = np.maximum(sigma1_sq, sigma2_sq)
        cw = cw / np.sum(cw)
        return float(np.sum(cw * (ramda * ssim_map1 + (1.0 - ramda) * ssim_map2)))

    return _dispatch_channels(img1, img2, fused, one)


def metrics_qy(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    def one(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
        a = _as_float(_maybe_gray(a))
        b = _as_float(_maybe_gray(b))
        f = _as_float(_maybe_gray(f))
        _, ssim_map1, sigma1_sq1, sigma2_sq1, _ = _ssim_yang(a, b)
        _, ssim_map2, _, _, _ = _ssim_yang(a, f)
        _, ssim_map3, _, _, _ = _ssim_yang(b, f)
        bin_map = ssim_map1 >= 0.75
        ramda = sigma1_sq1 / (sigma1_sq1 + sigma2_sq1)
        q1 = (ramda * ssim_map2 + (1.0 - ramda) * ssim_map3) * bin_map
        q2 = np.maximum(ssim_map2, ssim_map3) * (~bin_map)
        vals = q1 + q2
        return float(np.mean(vals[~np.isnan(vals)]))

    return _dispatch_channels(img1, img2, fused, one)


def metrics_qcb(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    return _dispatch_channels(img1, img2, fused, _qcb)


def _freqspace_mesh(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    if cols % 2 == 0:
        u = np.arange(-cols, cols, 2, dtype=np.float64) / cols
    else:
        u = np.arange(-(cols - 1), cols, 2, dtype=np.float64) / cols
    if rows % 2 == 0:
        v = np.arange(-rows, rows, 2, dtype=np.float64) / rows
    else:
        v = np.arange(-(rows - 1), rows, 2, dtype=np.float64) / rows
    return np.meshgrid(u, v)


def _gaussian2d(_n1: int, _n2: int, sigma: float) -> np.ndarray:
    x, y = np.meshgrid(np.arange(-15, 16), np.arange(-15, 16))
    return np.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)


def _contrast(g1: np.ndarray, g2: np.ndarray, image: np.ndarray) -> np.ndarray:
    buff = signal.fftconvolve(_as_numeric(image), _as_numeric(g1)[::-1, ::-1], mode="same")
    buff1 = signal.fftconvolve(_as_numeric(image), _as_numeric(g2)[::-1, ::-1], mode="same")
    return buff / buff1 - 1.0


def _qcb(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    im1 = _normalize1(_as_float(a) / 255.0 if np.issubdtype(np.asarray(a).dtype, np.integer) else _as_float(a))
    im2 = _normalize1(_as_float(b) / 255.0 if np.issubdtype(np.asarray(b).dtype, np.integer) else _as_float(b))
    fused = _normalize1(_as_float(f) / 255.0 if np.issubdtype(np.asarray(f).dtype, np.integer) else _as_float(f))
    f0, f1, alpha = 15.3870, 1.3456, 0.7622
    k, h, p, q, z = 1.0, 1.0, 3.0, 2.0, 0.0001
    rows, cols = im1.shape
    u, v = _freqspace_mesh((rows, cols))
    u = (cols / 30.0) * u
    v = (rows / 30.0) * v
    r = np.sqrt(u * u + v * v)
    sd = np.exp(-((r / f0) ** 2)) - alpha * np.exp(-((r / f1) ** 2))
    fused1 = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(im1)) * sd))
    fused2 = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(im2)) * sd))
    ffused = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(fused)) * sd))
    g1 = _gaussian2d(rows, cols, 2)
    g2 = _gaussian2d(rows, cols, 4)
    c1 = np.abs(_contrast(g1, g2, fused1))
    c2 = np.abs(_contrast(g1, g2, fused2))
    cf = np.abs(_contrast(g1, g2, ffused))
    c1p = (k * (c1**p)) / (h * (c1**q) + z)
    c2p = (k * (c2**p)) / (h * (c2**q) + z)
    cfp = (k * (cf**p)) / (h * (cf**q) + z)
    q1f = np.where(c1p < cfp, c1p / cfp, cfp / c1p)
    q2f = np.where(c2p < cfp, c2p / cfp, cfp / c2p)
    denom = c1p * c1p + c2p * c2p
    ramda1 = (c1p * c1p) / denom
    ramda2 = (c2p * c2p) / denom
    quality = ramda1 * q1f + ramda2 * q2f
    return float(np.mean(quality).real)


def metrics_qcv(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    return _dispatch_channels(img1, img2, fused, _qcv)


def _block_reduce_valid(image: np.ndarray, block: int, func: Callable[[np.ndarray], float]) -> np.ndarray:
    rows = image.shape[0] // block
    cols = image.shape[1] // block
    out = np.empty((rows, cols), dtype=np.complex128 if np.iscomplexobj(image) else np.float64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = func(image[i * block : (i + 1) * block, j * block : (j + 1) * block])
    return out


def _block_sum_power_valid(image: np.ndarray, block: int, power: float) -> np.ndarray:
    rows = image.shape[0] // block
    cols = image.shape[1] // block
    cropped = image[: rows * block, : cols * block]
    blocks = cropped.reshape(rows, block, cols, block).swapaxes(1, 2)
    return np.sum(blocks**power, axis=(2, 3))


def _block_mean_square_valid(image: np.ndarray, block: int) -> np.ndarray:
    rows = image.shape[0] // block
    cols = image.shape[1] // block
    cropped = image[: rows * block, : cols * block]
    blocks = cropped.reshape(rows, block, cols, block).swapaxes(1, 2)
    return np.mean(blocks**2, axis=(2, 3))


def _qcv(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    alpha_c, alpha_s, f_c, f_s = 1.0, 0.685, 97.3227, 12.1653
    window_size = 16
    alpha = 5.0
    im1 = _normalize1(a)
    im2 = _normalize1(b)
    fused = _normalize1(f)
    flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    im1_g = np.sqrt(_filter2(im1, flt1, "same") ** 2 + _filter2(im1, flt2, "same") ** 2)
    im2_g = np.sqrt(_filter2(im2, flt1, "same") ** 2 + _filter2(im2, flt2, "same") ** 2)
    ramda1 = _block_sum_power_valid(im1_g, window_size, alpha)
    ramda2 = _block_sum_power_valid(im2_g, window_size, alpha)

    f1 = im1 - fused
    f2 = im2 - fused
    rows, cols = im1.shape
    u, v = _freqspace_mesh((rows, cols))
    u = (cols / 8.0) * u
    v = (rows / 8.0) * v
    r = np.sqrt(u * u + v * v)
    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-((0.144 * r) ** 1.1))
    # Daly and Ahumada filters are built in the MATLAB code but theta_m is used.
    theta_a = alpha_c * np.exp(-((r / f_c) ** 2)) - alpha_s * np.exp(-((r / f_s) ** 2))
    del theta_a
    df1 = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(f1)) * theta_m))
    df2 = fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(f2)) * theta_m))
    d1 = _block_mean_square_valid(df1, window_size)
    d2 = _block_mean_square_valid(df2, window_size)
    q = np.sum(ramda1 * d1 + ramda2 * d2) / np.sum(ramda1 + ramda2)
    return float(np.real(q))


def _matlab_edge_sobel(image: np.ndarray) -> np.ndarray:
    image = _as_float(image)
    if np.nanmax(image) > 1.0 or np.nanmin(image) < 0.0:
        image = image / 255.0
    sobel = _fspecial_sobel()
    sobel = sobel / np.sum(np.abs(sobel))
    strength2 = _filter2_replicate(image, sobel) ** 2 + _filter2_replicate(image, sobel.T) ** 2
    thresh2 = 4.0 * np.mean(strength2)
    strength2 = strength2.copy()
    strength2[strength2 <= thresh2] = 0.0
    rows, cols = strength2.shape
    left = np.concatenate([np.zeros((rows, 1)), strength2[:, :-1]], axis=1)
    right = np.concatenate([strength2[:, 1:], np.zeros((rows, 1))], axis=1)
    up = np.concatenate([np.zeros((1, cols)), strength2[:-1, :]], axis=0)
    down = np.concatenate([strength2[1:, :], np.zeros((1, cols))], axis=0)
    return ((strength2 > left) & (strength2 > right) | ((strength2 > up) & (strength2 > down))).astype(np.float64)


def metrics_fmi(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    return _dispatch_channels(img1, img2, fused, lambda a, b, f: _fmi(a, b, f, feature="edge", w=3))


def _feature_image(image: np.ndarray, feature: str) -> np.ndarray:
    image = _as_float(image)
    if feature == "none":
        return image
    if feature == "gradient":
        return np.gradient(image)[0]
    if feature == "edge":
        return _matlab_edge_sobel(image)
    if feature == "dct":
        return fft.dctn(image, norm="ortho")
    raise ValueError("feature must be one of 'none', 'gradient', 'edge', or 'dct'")


def _window_fmi(a_sub: np.ndarray, f_sub: np.ndarray) -> float:
    if np.array_equal(a_sub, f_sub):
        return 1.0
    side = a_sub.shape[0]
    a_max, a_min = np.max(a_sub), np.min(a_sub)
    f_max, f_min = np.max(f_sub), np.min(f_sub)
    a_sub = np.ones((side, side)) if a_max == a_min else (a_sub - a_min) / (a_max - a_min)
    f_sub = np.ones((side, side)) if f_max == f_min else (f_sub - f_min) / (f_max - f_min)
    a_sum = np.sum(a_sub)
    f_sum = np.sum(f_sub)
    a_pdf = a_sub.ravel(order="F") / a_sum if a_sum != 0 else np.full(side * side, np.nan)
    f_pdf = f_sub.ravel(order="F") / f_sum if f_sum != 0 else np.full(side * side, np.nan)
    a_cdf = np.cumsum(a_pdf)
    f_cdf = np.cumsum(f_pdf)
    a_temp = a_pdf - np.mean(a_pdf)
    f_temp = f_pdf - np.mean(f_pdf)
    numerator = np.sum(a_temp * f_temp)
    c = 0.0 if numerator == 0 else numerator / np.sqrt(np.sum(a_temp * a_temp) * np.sum(f_temp * f_temp))
    idx = np.arange(1, len(a_pdf) + 1, dtype=np.float64)
    e_a = np.sum(idx * a_pdf)
    e2_a = np.sum((idx**2) * a_pdf)
    e_f = np.sum(idx * f_pdf)
    e2_f = np.sum((idx**2) * f_pdf)
    a_sd = np.sqrt(e2_a - e_a**2)
    f_sd = np.sqrt(e2_f - e_f**2)
    joint_entropy = 0.0
    if c >= 0:
        if c == 0 or a_sd == 0 or f_sd == 0:
            phi = 0.0
        else:
            up = 0.5 * (f_cdf[:, None] + a_cdf[None, :] - np.abs(f_cdf[:, None] - a_cdf[None, :]))
            cov_up = np.sum(up - f_cdf[:, None] * a_cdf[None, :])
            phi = c / (cov_up / (f_sd * a_sd))
        up_cdf = 0.5 * (f_cdf[:, None] + a_cdf[None, :] - np.abs(f_cdf[:, None] - a_cdf[None, :]))
        up = np.empty_like(up_cdf)
        up[0, 0] = up_cdf[0, 0]
        up[1:, 0] = up_cdf[1:, 0] - up_cdf[:-1, 0]
        up[0, 1:] = up_cdf[0, 1:] - up_cdf[0, :-1]
        up[1:, 1:] = up_cdf[1:, 1:] - up_cdf[:-1, 1:] - up_cdf[1:, :-1] + up_cdf[:-1, :-1]
        jpdf = phi * up + (1.0 - phi) * f_pdf[:, None] * a_pdf[None, :]
    else:
        if a_sd == 0 or f_sd == 0:
            theta = 0.0
        else:
            lo = 0.5 * (f_cdf[:, None] + a_cdf[None, :] - 1 + np.abs(f_cdf[:, None] + a_cdf[None, :] - 1))
            cov_lo = np.sum(lo - f_cdf[:, None] * a_cdf[None, :])
            theta = c / (cov_lo / (f_sd * a_sd))
        lo_cdf = 0.5 * (f_cdf[:, None] + a_cdf[None, :] - 1 + np.abs(f_cdf[:, None] + a_cdf[None, :] - 1))
        lo = np.empty_like(lo_cdf)
        lo[0, 0] = lo_cdf[0, 0]
        lo[1:, 0] = lo_cdf[1:, 0] - lo_cdf[:-1, 0]
        lo[0, 1:] = lo_cdf[0, 1:] - lo_cdf[0, :-1]
        lo[1:, 1:] = lo_cdf[1:, 1:] - lo_cdf[:-1, 1:] - lo_cdf[1:, :-1] + lo_cdf[:-1, :-1]
        jpdf = theta * lo + (1.0 - theta) * f_pdf[:, None] * a_pdf[None, :]
    nz = jpdf != 0
    joint_entropy = float(np.real(-np.sum(jpdf[nz].astype(np.complex128) * np.log2(jpdf[nz].astype(np.complex128)))))
    idx = a_pdf != 0
    a_entropy = float(np.sum(-a_pdf[idx] * np.log2(a_pdf[idx])))
    idx = f_pdf != 0
    f_entropy = float(np.sum(-f_pdf[idx] * np.log2(f_pdf[idx])))
    mi = a_entropy + f_entropy - joint_entropy
    return 0.0 if mi == 0 else float(2.0 * mi / (a_entropy + f_entropy))


def _fmi(a: np.ndarray, b: np.ndarray, f: np.ndarray, feature: str = "edge", w: int = 3) -> float:
    af = _feature_image(a, feature)
    bf = _feature_image(b, feature)
    ff = _feature_image(f, feature)
    half = int(np.floor(w / 2))
    rows, cols = af.shape
    out = np.ones((rows - 2 * half, cols - 2 * half), dtype=np.float64)
    for r in range(half, rows - half):
        for c in range(half, cols - half):
            aw = af[r - half : r + half + 1, c - half : c + half + 1]
            bw = bf[r - half : r + half + 1, c - half : c + half + 1]
            fw = ff[r - half : r + half + 1, c - half : c + half + 1]
            out[r - half, c - half] = (_window_fmi(aw, fw) + _window_fmi(bw, fw)) / 2.0
    return float(np.mean(out))


def metrics_mef_ssim(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    seq = np.stack([_as_float(_rgb2gray(img1)), _as_float(_rgb2gray(img2))], axis=2)
    fi = _as_float(_rgb2gray(fused))
    out, _q, _maps = _mef_ms_ssim(seq, fi)
    return float(out)


def _mef_ssim(
    img_seq: np.ndarray,
    fused: np.ndarray,
    k: float = 0.03,
    window: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if window is None:
        window = _fspecial_gaussian(11, 1.5)
    img_seq = _as_float(img_seq)
    fused = _as_float(fused)
    s1, s2, s3 = img_seq.shape
    w_size = window.shape[0]
    bd = w_size // 2
    s_window = np.ones((w_size, w_size), dtype=np.float64) / (w_size * w_size)
    mu = np.zeros((s1 - 2 * bd, s2 - 2 * bd, s3), dtype=np.float64)
    ed = np.zeros_like(mu)
    for i in range(s3):
        image = img_seq[:, :, i]
        mu[:, :, i] = _filter2(image, s_window, "valid")
        sigma_sq = _filter2(image * image, s_window, "valid") - mu[:, :, i] * mu[:, :, i]
        ed[:, :, i] = np.sqrt(np.maximum(w_size * w_size * sigma_sq, 0.0)) + 0.001

    src_windows = sliding_window_view(img_seq, (w_size, w_size), axis=(0, 1))
    src_windows = np.moveaxis(src_windows, 2, -1)
    seq_sum = np.sum(src_windows, axis=4)
    numerator = np.linalg.norm(seq_sum - np.mean(seq_sum, axis=(2, 3), keepdims=True), axis=(2, 3))
    denominator = np.zeros_like(numerator)
    for i in range(s3):
        denominator += np.linalg.norm(src_windows[..., i] - mu[:, :, i, None, None], axis=(2, 3))
    r_map = (numerator + np.finfo(np.float64).eps) / (denominator + np.finfo(np.float64).eps)
    r_map = np.where(r_map > 1, 1 - np.finfo(np.float64).eps, r_map)
    r_map = np.where(r_map < 0, np.finfo(np.float64).eps, r_map)
    p = np.tan(np.pi / 2.0 * r_map)
    p = np.where(p > 10, 10, p)
    w_map = (ed / w_size) ** p[..., None] + np.finfo(np.float64).eps
    w_map = w_map / np.sum(w_map, axis=2, keepdims=True)
    max_ed = np.max(ed, axis=2)

    weighted = np.sum(
        w_map[:, :, None, None, :] * (src_windows - mu[:, :, None, None, :]) / ed[:, :, None, None, :],
        axis=4,
    )
    norm = np.linalg.norm(weighted, axis=(2, 3))
    scale = np.where(norm > 0, max_ed / norm, 0.0)
    ref_blocks = weighted * scale[:, :, None, None]
    fused_windows = sliding_window_view(fused, (w_size, w_size))
    r_vec = ref_blocks.reshape(ref_blocks.shape[0], ref_blocks.shape[1], -1, order="F")
    f_vec = fused_windows.reshape(fused_windows.shape[0], fused_windows.shape[1], -1, order="F")
    weights = window.ravel(order="F")
    mu1 = np.sum(weights * r_vec, axis=2)
    mu2 = np.sum(weights * f_vec, axis=2)
    sigma1_sq = np.sum(weights * (r_vec - mu1[:, :, None]) ** 2, axis=2)
    sigma2_sq = np.sum(weights * (f_vec - mu2[:, :, None]) ** 2, axis=2)
    sigma12 = np.sum(weights * (r_vec - mu1[:, :, None]) * (f_vec - mu2[:, :, None]), axis=2)
    c = (k * 255.0) ** 2
    q_map = (2.0 * sigma12 + c) / (sigma1_sq + sigma2_sq + c)
    return _mean2(q_map), q_map


def _mef_ms_ssim(
    img_seq: np.ndarray,
    fused: np.ndarray,
    k: float = 0.03,
    window: np.ndarray | None = None,
    level: int = 3,
    weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray, list[np.ndarray]]:
    if window is None:
        window = _fspecial_gaussian(11, 1.5)
    if weight is None:
        weight = np.array([0.0448, 0.2856, 0.3001], dtype=np.float64)
        weight = weight / np.sum(weight)
    if level != len(weight):
        return -np.inf, np.array(-np.inf), [np.array(-np.inf)]
    min_width = min(img_seq.shape[0], img_seq.shape[1]) / (2 ** (level - 1))
    if min_width < max(window.shape):
        return -np.inf, np.array(-np.inf), [np.array(np.inf)]
    downsample = np.ones((2, 2), dtype=np.float64) / 4.0
    qs = np.zeros(level, dtype=np.float64)
    maps: list[np.ndarray] = []
    seq = _as_float(img_seq)
    fi = _as_float(fused)
    for lvl in range(level - 1):
        qs[lvl], qmap = _mef_ssim(seq, fi, k, window)
        maps.append(qmap)
        seq = np.stack([_imfilter_symmetric_same(seq[:, :, i], downsample)[::2, ::2] for i in range(seq.shape[2])], axis=2)
        fi = _imfilter_symmetric_same(fi, downsample)[::2, ::2]
    qs[level - 1], qmap = _mef_ssim(seq, fi, k, window)
    maps.append(qmap)
    return float(np.prod(qs**weight)), qs, maps


def _lowpassfilter(shape: tuple[int, int], cutoff: float, n: int) -> np.ndarray:
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError("cutoff frequency must be between 0 and 0.5")
    rows, cols = shape
    x = (np.arange(1, cols + 1) - (np.fix(cols / 2) + 1)) / cols
    y = (np.arange(1, rows + 1) - (np.fix(rows / 2) + 1)) / rows
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    return fft.fftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))


def _phase_congruency(
    image: np.ndarray,
    nscale: int = 4,
    norient: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
    dtheta_on_sigma: float = 1.2,
    k: float = 2.0,
    cutoff: float = 0.5,
    g: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    im = _as_float(image)
    epsilon = 0.0001
    theta_sigma = np.pi / norient / dtheta_on_sigma
    rows, cols = im.shape
    imagefft = fft.fft2(im)
    zero = np.zeros((rows, cols), dtype=np.float64)
    total_energy = zero.copy()
    total_sum_an = zero.copy()
    orientation = zero.copy()
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols / 2, cols / 2) / cols
    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows / 2, rows / 2) / rows
    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x * x + y * y)
    radius[int(np.floor(rows / 2)), int(np.floor(cols / 2))] = 1
    theta = np.arctan2(-y, x)
    radius = fft.ifftshift(radius)
    theta = fft.ifftshift(theta)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    lp = _lowpassfilter((rows, cols), 0.45, 15)
    log_gabor = []
    for scale in range(nscale):
        wavelength = min_wavelength * (mult**scale)
        fo = 1.0 / wavelength
        filt = np.exp((-(np.log(radius / fo) ** 2)) / (2 * (np.log(sigma_onf) ** 2)))
        filt *= lp
        filt[0, 0] = 0
        log_gabor.append(filt)
    spread = []
    for orient in range(norient):
        angl = orient * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread.append(np.exp((-(dtheta**2)) / (2 * theta_sigma**2)))

    max_energy = None
    for orient in range(norient):
        angl = orient * np.pi / norient
        sum_e = zero.copy()
        sum_o = zero.copy()
        sum_an = zero.copy()
        energy = zero.copy()
        eo = []
        ifft_filter_array = []
        em_n = 0.0
        max_an = None
        for scale in range(nscale):
            filt = log_gabor[scale] * spread[orient]
            ifft_filt = np.real(fft.ifft2(filt)) * np.sqrt(rows * cols)
            ifft_filter_array.append(ifft_filt)
            conv = fft.ifft2(imagefft * filt)
            eo.append(conv)
            an = np.abs(conv)
            sum_an += an
            sum_e += np.real(conv)
            sum_o += np.imag(conv)
            if scale == 0:
                em_n = np.sum(filt**2)
                max_an = an
            else:
                max_an = np.maximum(max_an, an)
        x_energy = np.sqrt(sum_e**2 + sum_o**2) + epsilon
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy
        for conv in eo:
            e = np.real(conv)
            o = np.imag(conv)
            energy += e * mean_e + o * mean_o - np.abs(e * mean_o - o * mean_e)
        median_e2n = np.median(np.reshape(np.abs(eo[0]) ** 2, rows * cols))
        mean_e2n = -median_e2n / np.log(0.5)
        noise_power = mean_e2n / em_n
        est_sum_an2 = zero.copy()
        for ifft_filt in ifft_filter_array:
            est_sum_an2 += ifft_filt**2
        est_sum_aiaj = zero.copy()
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                est_sum_aiaj += ifft_filter_array[si] * ifft_filter_array[sj]
        est_noise_energy2 = 2 * noise_power * np.sum(est_sum_an2) + 4 * noise_power * np.sum(est_sum_aiaj)
        tau = np.sqrt(est_noise_energy2 / 2.0)
        est_noise_energy = tau * np.sqrt(np.pi / 2.0)
        est_noise_energy_sigma = np.sqrt((2.0 - np.pi / 2.0) * tau * tau)
        threshold = (est_noise_energy + k * est_noise_energy_sigma) / 1.7
        energy = np.maximum(energy - threshold, zero)
        width = sum_an / (max_an + epsilon) / nscale
        weight = 1.0 / (1.0 + np.exp((cutoff - width) * g))
        energy_this_orient = weight * energy
        total_sum_an += sum_an
        total_energy += energy_this_orient
        if orient == 0:
            max_energy = energy_this_orient
        else:
            change = energy_this_orient > max_energy
            orientation = orient * change + orientation * (~change)
            max_energy = np.maximum(max_energy, energy_this_orient)
        pc = weight * energy / sum_an
        covx = pc * np.cos(angl)
        covy = pc * np.sin(angl)
        covx2 += covx**2
        covy2 += covy**2
        covxy += covx * covy
    phase_congruency = total_energy / (total_sum_an + epsilon)
    orientation = orientation * (180.0 / norient)
    covx2 = covx2 / (norient / 2.0)
    covy2 = covy2 / (norient / 2.0)
    covxy = covxy / norient
    denom = np.sqrt(covxy**2 + (covx2 - covy2) ** 2) + epsilon
    sin2theta = covxy / denom
    cos2theta = (covx2 - covy2) / denom
    orient = np.rint(np.arctan2(sin2theta, cos2theta) / 2.0 * 180.0 / np.pi)
    orient = np.where(orient < 0, orient + 180, orient)
    mmax = (covy2 + covx2 + denom) / 2.0
    mmin = (covy2 + covx2 - denom) / 2.0
    return phase_congruency, orient, mmax, mmin


def metrics_qp(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    return _dispatch_channels(img1, img2, fused, _qp)


def _correlation_coefficient(
    im1: np.ndarray,
    im2: np.ndarray,
    im_max: np.ndarray,
    imf: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    mask3: np.ndarray,
) -> float:
    window = _fspecial_gaussian(11, 1.5)
    c1 = c2 = c3 = 0.0001
    im1 = mask1 * im1
    im2 = mask2 * im2
    im_max = mask3 * im_max
    mu1 = _filter2(im1, window, "same")
    mu2 = _filter2(im2, window, "same")
    muf = _filter2(imf, window, "same")
    mu_max = _filter2(im_max, window, "same")
    sigma1_sq = _filter2(im1 * im1, window, "same") - mu1 * mu1
    sigma2_sq = _filter2(im2 * im2, window, "same") - mu2 * mu2
    sigma_max_sq = _filter2(im_max * im_max, window, "same") - mu_max * mu_max
    sigmaf_sq = _filter2(imf * imf, window, "same") - muf * muf
    sigma1f = _filter2(im1 * imf, window, "same") - mu1 * muf
    sigma2f = _filter2(im2 * imf, window, "same") - mu2 * muf
    sigma_maxf = _filter2(im_max * imf, window, "same") - mu_max * muf
    res1 = np.zeros_like(mu1)
    res2 = np.zeros_like(mu1)
    res3 = np.zeros_like(mu1)
    idx1 = mask1 == 1
    idx2 = mask2 == 1
    idx3 = mask3 == 1
    res1[idx1] = (sigma1f[idx1] + c1) / (np.sqrt(np.abs(sigma1_sq[idx1] * sigmaf_sq[idx1])) + c1)
    res2[idx2] = (sigma2f[idx2] + c2) / (np.sqrt(np.abs(sigma2_sq[idx2] * sigmaf_sq[idx2])) + c2)
    res3[idx3] = (sigma_maxf[idx3] + c3) / (np.sqrt(np.abs(sigma_max_sq[idx3] * sigmaf_sq[idx3])) + c3)
    result = np.maximum.reduce([res1, res2, res3])
    return float(np.sum(result) / np.sum(mask3))


def _qp(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    threshold = 0.1
    im1 = _as_float(_maybe_gray(a))
    im2 = _as_float(_maybe_gray(b))
    fused = _as_float(_maybe_gray(f))
    pc1, _or1, m1_big, m1_small = _phase_congruency(im1)
    pc2, _or2, m2_big, m2_small = _phase_congruency(im2)
    pcf, _orf, mf_big, mf_small = _phase_congruency(fused)
    mask = pc1 > pc2
    pc_max = mask * pc1 + (~mask) * pc2
    m_big_max = mask * m1_big + (~mask) * m2_big
    m_small_max = mask * m1_small + (~mask) * m2_small
    mask1 = pc1 > threshold
    mask2 = pc2 > threshold
    mask3 = pc_max > threshold
    result_pc = _correlation_coefficient(pc1, pc2, pc_max, pcf, mask1, mask2, mask3)
    result_m_big = _correlation_coefficient(m1_big, m2_big, m_big_max, mf_big, mask1, mask2, mask3)
    result_m_small = _correlation_coefficient(m1_small, m2_small, m_small_max, mf_small, mask1, mask2, mask3)
    return float(result_pc * result_m_big * result_m_small)


def metrics_vif(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    return _dispatch_channels(img1, img2, fused, _vif)


def _com_vid_vind_g(ref: np.ndarray, dist: np.ndarray, sq: float) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    sigma_nsq = sq
    nums: list[np.ndarray] = []
    dens: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    ref = _as_float(ref)
    dist = _as_float(dist)
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        win = _fspecial_gaussian(n, n / 5.0)
        if scale > 1:
            ref = _filter2(ref, win, "valid")[::2, ::2]
            dist = _filter2(dist, win, "valid")[::2, ::2]
        mu1 = _filter2(ref, win, "valid")
        mu2 = _filter2(dist, win, "valid")
        sigma1_sq = _filter2(ref * ref, win, "valid") - mu1 * mu1
        sigma2_sq = _filter2(dist * dist, win, "valid") - mu2 * mu2
        sigma12 = _filter2(ref * dist, win, "valid") - mu1 * mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0
        gain = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - gain * sigma12
        idx = sigma1_sq < 1e-10
        gain[idx] = 0
        sv_sq[idx] = sigma2_sq[idx]
        sigma1_sq[idx] = 0
        idx = sigma2_sq < 1e-10
        gain[idx] = 0
        sv_sq[idx] = 0
        idx = gain < 0
        sv_sq[idx] = sigma2_sq[idx]
        gain[idx] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10
        vid = np.log10(1 + gain * gain * sigma1_sq / (sv_sq + sigma_nsq))
        vind = np.log10(1 + sigma1_sq / sigma_nsq)
        nums.append(vid)
        dens.append(vind)
        gs.append(gain)
    return nums, dens, gs


def _vif(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    sq = 0.005 * 255.0 * 255.0
    c = 1e-7
    t1 = _as_float(_maybe_gray(a))
    t2 = _as_float(_maybe_gray(b))
    tf = _as_float(_maybe_gray(f))
    weights = np.array([1, 0, 0.15, 1], dtype=np.float64) / 2.15
    t1n, t1d, t1g = _com_vid_vind_g(t1, tf, sq)
    t2n, t2d, t2g = _com_vid_vind_g(t2, tf, sq)
    vals = np.empty(4, dtype=np.float64)
    for i in range(4):
        z1, z2 = t1n[i], t2n[i]
        m1, m2 = t1d[i], t2d[i]
        g1, g2 = t1g[i], t2g[i]
        choose_first = g1 < g2
        z12 = z2.copy()
        z12[choose_first] = z1[choose_first]
        m12 = m2.copy()
        m12[choose_first] = m1[choose_first]
        vals[i] = np.sum(z12 + c) / np.sum(m12 + c)
    return float(np.sum(vals * weights))


_METRICS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "cross_entropy": metrics_cross_entropy,
    "entropy": metrics_entropy,
    "fmi": metrics_fmi,
    "nmi": metrics_nmi,
    "psnr": metrics_psnr,
    "qncie": metrics_qncie,
    "te": metrics_te,
    "avg_gradient": metrics_avg_gradient,
    "edge_intensity": metrics_edge_intensity,
    "qabf": metrics_qabf,
    "qp": metrics_qp,
    "sd": metrics_sd,
    "spatial_frequency": metrics_spatial_frequency,
    "qc": metrics_qc,
    "qw": metrics_qw,
    "qy": metrics_qy,
    "mef_ssim": metrics_mef_ssim,
    "qcb": metrics_qcb,
    "qcv": metrics_qcv,
    "vif": metrics_vif,
}
