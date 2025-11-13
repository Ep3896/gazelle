#!/usr/bin/env python
import os, sys
import cv2, numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from pathlib import Path
import argparse






# ---------- repo root ----------
PROJ = Path(__file__).resolve().parents[1]

# ---------- CLI / ENV ----------
p = argparse.ArgumentParser()
p.add_argument("--model", default=os.getenv("GAZELLE_MODEL", "gazelle_dinov2_vitb14"),
               choices=["gazelle_dinov2_vitb14", "gazelle_dinov2_vitb14_inout",
                        "gazelle_dinov2_vitl14", "gazelle_dinov2_vitl14_inout"])
p.add_argument("--ckpt", default=os.getenv("GAZELLE_CKPT", ""),
               help="Path to checkpoint .pt (defaults per model if empty)")
p.add_argument("--cam", type=int, default=int(os.getenv("GAZELLE_CAM", "0")))
args, _ = p.parse_known_args()

MODEL_NAME = args.model

# Default checkpoints by model (repo-relative, portable)
_default_ckpts = {
    "gazelle_dinov2_vitb14":         PROJ / "checkpoints" / "gazelle_dinov2_vitb14.pt",
    "gazelle_dinov2_vitb14_inout":   PROJ / "checkpoints" / "gazelle_dinov2_vitb14.pt",
    "gazelle_dinov2_vitl14":         PROJ / "checkpoints" / "gazelle_dinov2_vitl14.pt",
    "gazelle_dinov2_vitl14_inout":   PROJ / "checkpoints" / "gazelle_dinov2_vitl14.pt",
}
CHECKPOINT = Path(args.ckpt) if args.ckpt else _default_ckpts[MODEL_NAME]

# # ---------- Offline DINOv2 redirection ----------
# # Prefer repo-relative official clone; allow env override.
# D2_LOCAL = Path(os.getenv("D2_LOCAL", PROJ / "data" / "models" / "dinov2_official")).resolve()
# hubconf_path = D2_LOCAL / "hubconf.py"
# if not hubconf_path.is_file():
#     sys.stderr.write(
#         f"[ERROR] D2_LOCAL does not contain hubconf.py:\n"
#         f"        D2_LOCAL={D2_LOCAL}\n"
#         f"        expected={hubconf_path}\n"
#     )
#     sys.exit(1)

# # Local torch hub cache inside repo (portable)
# torch_hub_dir = PROJ / "torch_hub_cache" / "hub"
# os.environ.setdefault("TORCH_HUB_DIR", str(torch_hub_dir))
# os.environ.setdefault("TORCH_HOME", os.environ["TORCH_HUB_DIR"])
# os.environ["TORCH_HUB_DISABLE_FORK_CHECK"] = "1"

# _orig_hub_load = torch.hub.load
# def _localize_hub(repo_or_dir, model, *a, **kw):
#     if isinstance(repo_or_dir, str) and repo_or_dir.lower() == "facebookresearch/dinov2":
#         kw.setdefault("source", "local")
#         kw.setdefault("trust_repo", True)
#         repo_or_dir = str(D2_LOCAL)
#     return _orig_hub_load(repo_or_dir, model, *a, **kw)
# torch.hub.load = _localize_hub
# # -----------------------------------------------


# ---------- Offline DINOv2 redirection (auto-setup) ----------
import subprocess, shutil

def _maybe_clone_dinov2(dst: Path) -> bool:
    """Try a shallow clone of facebookresearch/dinov2 into dst. Return True on success."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("git") is None:
            return False
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        repo = f"https://{token+'@' if token else ''}github.com/facebookresearch/dinov2.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo, str(dst)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False

def _find_cached_dinov2() -> Path | None:
    """Search local torch hub cache for a dinov2 clone (with hubconf.py)."""
    hub_root = Path(os.environ.get("TORCH_HUB_DIR", PROJ / "torch_hub_cache" / "hub"))
    if not hub_root.is_dir():
        return None
    for p in hub_root.rglob("hubconf.py"):
        if "dinov2" in p.as_posix():
            # Heuristic: expect dinov2/hub/backbones.py around
            if (p.parent / "dinov2" / "hub" / "backbones.py").exists():
                return p.parent
    return None

# Target location (repo-relative by default, overridable via env)
D2_LOCAL = Path(os.getenv("D2_LOCAL", PROJ / "data" / "models" / "dinov2_official")).resolve()
hubconf_path = D2_LOCAL / "hubconf.py"

ALLOW_ONLINE = os.environ.get("GAZELLE_ALLOW_ONLINE", "1").lower() in ("1", "true", "yes")

# 1) try existing local clone at D2_LOCAL
if not hubconf_path.is_file():
    # 2) try a cached dinov2 in torch hub
    cached = _find_cached_dinov2()
    if cached:
        D2_LOCAL = cached
        hubconf_path = D2_LOCAL / "hubconf.py"

# 3) if still missing and online allowed, try to clone
if not hubconf_path.is_file() and ALLOW_ONLINE:
    if _maybe_clone_dinov2(D2_LOCAL):
        hubconf_path = D2_LOCAL / "hubconf.py"

# 4) if still missing and online disabled, fail with a clear message
USE_LOCAL = hubconf_path.is_file()
if not USE_LOCAL and not ALLOW_ONLINE:
    sys.stderr.write(
        "[ERROR] DINOv2 is not available locally and online access is disabled.\n"
        f"        Set D2_LOCAL to a valid clone or export GAZELLE_ALLOW_ONLINE=1.\n"
        f"        Expected hubconf at: {hubconf_path}\n"
    )
    sys.exit(1)

# Local torch hub cache inside repo (portable)
torch_hub_dir = PROJ / "torch_hub_cache" / "hub"
os.environ.setdefault("TORCH_HUB_DIR", str(torch_hub_dir))
os.environ.setdefault("TORCH_HOME", os.environ["TORCH_HUB_DIR"])
os.environ["TORCH_HUB_DISABLE_FORK_CHECK"] = "1"

_orig_hub_load = torch.hub.load
def _localize_hub(repo_or_dir, model, *a, **kw):
    # If we have a local DINOv2, force local source; otherwise fall back to upstream.
    if isinstance(repo_or_dir, str) and repo_or_dir.lower() == "facebookresearch/dinov2":
        if USE_LOCAL:
            kw.setdefault("source", "local")
            kw.setdefault("trust_repo", True)
            repo_or_dir = str(D2_LOCAL)
        else:
            kw.setdefault("trust_repo", True)
    return _orig_hub_load(repo_or_dir, model, *a, **kw)

torch.hub.load = _localize_hub
# -------------------------------------------------------------

from gazelle.model import get_gazelle_model

# ---------- Finalize checkpoint & camera ----------
from urllib.request import urlretrieve

# Known official checkpoints (from the original repo releases)
_ckpt_urls = {
    "gazelle_dinov2_vitb14":       "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14.pt",
    "gazelle_dinov2_vitb14_inout": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14.pt",
    "gazelle_dinov2_vitl14":       "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt",
    "gazelle_dinov2_vitl14_inout": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt",
}

if not CHECKPOINT.is_file():
    allow_online = os.environ.get("GAZELLE_ALLOW_ONLINE", "1").lower() in ("1", "true", "yes")
    url = _ckpt_urls.get(MODEL_NAME)

    if url is not None and allow_online:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Checkpoint not found, downloading {MODEL_NAME} from:\n       {url}")
        try:
            urlretrieve(url, CHECKPOINT)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download checkpoint for {MODEL_NAME} from {url}: {e}"
            ) from e

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found and could not be downloaded: {CHECKPOINT}")

CAM_INDEX = args.cam
ALPHA, BETA = 0.6, 0.4

# ---------------------------------------------------------------------------

# Load model + transform
model, transform = get_gazelle_model(MODEL_NAME)
state = torch.load(CHECKPOINT, map_location="cpu")  # no 'trust_repo' here
# If your torch supports it and the file is a pure state_dict, you may use:
# state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
model.load_gazelle_state_dict(state)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Open webcam
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam index {CAM_INDEX}")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_FPS,          30)

print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Warning: failed to read frame; retrying…")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)

        img_tensor = transform(pil_img).unsqueeze(0).to(device)  # [1,3,448,448]
        bboxes = [[(0.0, 0.0, 1.0, 1.0)]]  # full-frame person

        with torch.no_grad():
            output = model({"images": img_tensor, "bboxes": bboxes})

        heat = output["heatmap"][0][0]              # [64,64]
        inout = None if output.get("inout") is None else float(output["inout"][0][0])

        H, W = frame_bgr.shape[:2]
        heat_up = F.interpolate(
            heat[None, None, ...], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
        heat_np = heat_up.detach().float().cpu().numpy()
        denom = (heat_np.max() - heat_np.min())
        heat_np = (heat_np - heat_np.min()) / (denom + 1e-8)
        heat_uint8 = (heat_np * 255.0).astype(np.uint8)

        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_bgr, ALPHA, heat_color, BETA, 0)

        if inout is not None:
            txt = f"inout: {inout:.2f} (>=0.5 ≈ in-frame)"
            cv2.putText(
                overlay, txt, (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if inout >= 0.5 else (0, 0, 255), 2
            )

        cv2.imshow("Gazelle webcam (heatmap overlay)", overlay)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):  # ESC/q
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
