import numpy as np
import scipy.ndimage as ndi


def fbm_noise(shape, octaves=5, lacunarity=2.0, gain=0.5, sigma0=40.0, seed=None):
    """
    Fast approximate fBM via gaussian blurs at multiple scales.
    Produces smooth cloud-like fields in [0,1].
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, shape).astype(np.float32)
    h, w = shape
    acc = np.zeros_like(base)
    amp = 1.0
    sigma = sigma0
    for _ in range(octaves):
        acc += amp * ndi.gaussian_filter(base, sigma=sigma, mode="reflect")
        amp *= gain
        sigma /= lacunarity
    acc -= acc.min()
    acc /= acc.max() + 1e-8
    return acc


def percentile_threshold(field, coverage):
    """Pick threshold to achieve ~coverage fraction above threshold."""
    return np.percentile(field, 100 * (1 - coverage))


# ---------- Haze ----------
def simulate_haze(image01, beta=1.2, A=(0.85, 0.9, 1.0), strength=1.0, seed=None):
    """
    Koschmieder-like: I = J * t + A * (1 - t), with spatial t from smooth noise.
    - beta: scattering strength (higher -> more haze)
    - A: airlight color (tuple per channel; blue-ish lifts sky/haze)
    - strength: overall mix multiplier [0..1]
    """
    img = image01.astype(np.float32)
    H, W = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]

    # Smooth pseudo-depth field in [0,1]
    depth = fbm_noise((H, W), octaves=4, sigma0=H / 5, seed=seed)
    # Transmittance
    t = np.exp(-beta * depth).astype(np.float32)  # in (0,1]

    # Expand to channels
    tC = t[..., None] if ch > 1 else t
    A = np.asarray(A, np.float32)
    if ch == 1:
        A = np.asarray([A[0] if A.ndim else A], np.float32)
        A = A[0]
    else:
        if A.size == 1:
            A = np.full((ch,), float(A), np.float32)
        elif A.size != ch:
            A = np.resize(A, (ch,)).astype(np.float32)

    # Apply haze
    if ch == 1:
        hazy = img * t + A * (1 - t)
    else:
        hazy = img * tC + A * (1 - tC)

    # Global strength blend (0=no haze, 1=full haze)
    out = (1 - strength) * img + strength * hazy
    return np.clip(out, 0, 1)


# ---------- Clouds + shadows ----------
def simulate_clouds(
    image01,
    coverage=0.25,  # fraction of image cloudy
    softness=0.6,  # 0 hard edges, 1 very soft
    cloud_brightness=1.0,  # albedo of cloud layer
    cloud_tint=(1.0, 1.0, 1.0),
    halo_blur=3.0,  # small glow around cloud tops
    shadow_strength=0.35,  # 0..1 how dark shadows are
    shadow_blur=7.0,  # softness of shadow
    sun_shift=(20, 15),  # (dy, dx) pixel shift for shadow projection
    seed=None,
):
    """
    Composites textured clouds and their shadows over the image.
    """
    img = image01.astype(np.float32)
    H, W = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]

    # Cloud field
    field = fbm_noise((H, W), octaves=6, sigma0=H / 5, seed=seed)
    thr = percentile_threshold(field, coverage)
    # Soft mask around threshold
    k = max(1e-3, softness * 12.0)
    alpha = 1 / (1 + np.exp(-(field - thr) * k))  # smoothstep via sigmoid
    # Optional halo/glow
    if halo_blur > 0:
        alpha = np.clip(alpha + 0.25 * ndi.gaussian_filter(alpha, halo_blur), 0, 1)

    # Cloud color
    tint = np.asarray(cloud_tint, np.float32)
    if ch == 1:
        cloud = np.full(
            (H, W), cloud_brightness * float(tint[0] if tint.size > 0 else 1.0), np.float32
        )
    else:
        if tint.size == 1:
            tint = np.full((ch,), float(tint), np.float32)
        elif tint.size != ch:
            tint = np.resize(tint, (ch,)).astype(np.float32)
        cloud = np.clip(tint * cloud_brightness, 0, 1)[None, None, :].repeat(H, 0).repeat(W, 1)

    # Composite clouds
    aC = alpha[..., None] if ch > 1 else alpha
    with_clouds = (1 - aC) * img + aC * cloud

    # Shadows: shift cloud mask along sun, blur, darken underlying image
    dy, dx = sun_shift
    shadow = np.zeros_like(alpha)
    y0, y1 = max(0, dy), min(H, H + dy)
    x0, x1 = max(0, dx), min(W, W + dx)
    yy0, yy1 = max(0, -dy), min(H, H - dy)
    xx0, xx1 = max(0, -dx), min(W, W - dx)
    shadow[y0:y1, x0:x1] = alpha[yy0:yy1, xx0:xx1]
    if shadow_blur > 0:
        shadow = ndi.gaussian_filter(shadow, shadow_blur)

    sC = shadow[..., None] if ch > 1 else shadow
    darkener = 1.0 - shadow_strength * sC
    with_shadows = np.clip(with_clouds * darkener, 0, 1)

    return with_shadows, alpha  # return mask too if you want to visualize
