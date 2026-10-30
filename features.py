import cv2, numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
FR = 128

def segment(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = cv2.GaussianBlur(lab[..., 0], (5, 5), 0)
    t, _ = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (L < t).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, cc, st, cen = cv2.connectedComponentsWithStats(m)
    ok = False
    if n > 1:
        best, bs = None, -1
        for i in range(1, n):
            d = np.hypot(*(cen[i] - FR / 2)) / (FR / 2)
            sc = st[i, cv2.CC_STAT_AREA] * (1 - 0.7 * min(d, 1))
            if sc > bs: bs, best = sc, i
        m = (cc == best).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(m); cv2.drawContours(m, cnts, -1, 1, -1)
        ok = 0.02 < m.mean() < 0.85
    if not ok:
        m = np.zeros((FR, FR), np.uint8)
        cv2.ellipse(m, (FR // 2, FR // 2), (FR // 3, FR // 3), 0, 0, 360, 1, -1)
    return m, ok

FEAT_NAMES, FEAT_GROUP = [], []
def _reg(names, g):
    for n in names: FEAT_NAMES.append(n); FEAT_GROUP.append(g)
_reg(['area_frac', 'compact', 'solidity', 'extent', 'aspect', 'ecc', 'asym_h', 'asym_v', 'radial_cv', 'border_grad'], 'shape')
_reg([f'{c}_{s}' for c in ['R','G','B','H','S','V','L','a','b'] for s in ['mean','std']], 'color')
_reg(['contrast_L', 'contrast_a', 'contrast_b', 'n_colors'], 'color')
_reg(['glcm_contrast', 'glcm_dissim', 'glcm_homog', 'glcm_energy', 'glcm_corr'] + [f'lbp{i}' for i in range(10)], 'texture')
_reg(['seg_ok'], 'meta')
NF = len(FEAT_NAMES)

def hand_feats(img):
    m, ok = segment(img)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    P = cv2.arcLength(c, True) + 1e-6; Ac = cv2.contourArea(c) + 1e-6
    Ah = cv2.contourArea(cv2.convexHull(c)) + 1e-6
    x, y, w, h = cv2.boundingRect(c)
    compact = 4 * np.pi * Ac / P ** 2; solidity = Ac / Ah; extent = Ac / (w * h + 1e-6); aspect = w / (h + 1e-6)
    ecc = 0.0
    if len(c) >= 5:
        (_, (MA, ma), _) = cv2.fitEllipse(c)
        ecc = float(np.sqrt(max(0.0, 1 - (min(MA, ma) / max(MA, ma, 1e-6)) ** 2)))
    ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
    M = np.float32([[1, 0, FR / 2 - cx], [0, 1, FR / 2 - cy]])
    mc = cv2.warpAffine(m, M, (FR, FR), flags=cv2.INTER_NEAREST).astype(bool)
    iou = lambda a, b: (a & b).sum() / ((a | b).sum() + 1e-6)
    asym_h = 1 - iou(mc, mc[:, ::-1]); asym_v = 1 - iou(mc, mc[::-1, :])
    pts = c[:, 0, :].astype(np.float32); r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    radial_cv = r.std() / (r.mean() + 1e-6)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gm = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    border_grad = gm[edge].mean() if edge.any() else 0.0
    f = [m.mean(), compact, solidity, extent, aspect, ecc, asym_h, asym_v, radial_cv, border_grad]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    inside = m > 0
    ring = (cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0) & ~(cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0)
    if ring.sum() < 10: ring = ~inside
    if ring.sum() < 10: ring = inside
    for arr in (img, hsv, lab):
        px = arr[inside].astype(np.float32)
        for k in range(3): f += [px[:, k].mean(), px[:, k].std()]
    f += list(lab[inside].astype(np.float32).mean(0) - lab[ring].astype(np.float32).mean(0))
    hb = np.clip(hsv[..., 0][inside] // 30, 0, 5).astype(int); vb = np.clip(hsv[..., 2][inside] // 86, 0, 2).astype(int)
    hist = np.bincount(hb * 3 + vb, minlength=18) / max(inside.sum(), 1)
    f.append(float((hist > 0.05).sum()))
    crop = (gray[y:y+h, x:x+w] // 16).astype(np.uint8)
    gl = graycomatrix(crop, [1, 2], [0, np.pi / 2], levels=16, symmetric=True, normed=True)
    f += [float(np.nan_to_num(graycoprops(gl, p).mean())) for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']]
    lbp = local_binary_pattern(gray, 8, 1, 'uniform')[inside]
    f += list(np.bincount(lbp.astype(int), minlength=10)[:10] / max(len(lbp), 1))
    f.append(float(ok))
    f = np.nan_to_num(np.array(f, dtype=np.float32))
    assert len(f) == NF, (len(f), NF)
    return f

