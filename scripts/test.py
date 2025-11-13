#!/usr/bin/env python
import os, sys
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


from pathlib import Path
PROJ = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.getenv("GAZELLE_CKPT", PROJ / "checkpoints" / "gazelle_dinov2_vitl14.pt"))
if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# --- Offline DINOv2 redirection (single, correct patch) ---------------------
D2_LOCAL = os.environ.get(
    "D2_LOCAL",
    "/home/enricopiacenti/Desktop/EP/projects/gazelle/data/models/dinov2_official",
)
D2_LOCAL = os.path.abspath(D2_LOCAL)
hubconf_path = os.path.join(D2_LOCAL, "hubconf.py")
if not os.path.isfile(hubconf_path):
    sys.stderr.write(
        f"[ERROR] D2_LOCAL does not contain hubconf.py:\n"
        f"        D2_LOCAL={D2_LOCAL}\n"
        f"        expected={hubconf_path}\n"
    )
    sys.exit(1)

os.environ.setdefault(
    "TORCH_HUB_DIR",
    "/home/enricopiacenti/Desktop/EP/projects/gazelle/torch_hub_cache/hub",
)
os.environ.setdefault("TORCH_HOME", os.environ["TORCH_HUB_DIR"])
os.environ["TORCH_HUB_DISABLE_FORK_CHECK"] = "1"

_orig_hub_load = torch.hub.load
def _localize_hub(repo_or_dir, model, *args, **kwargs):
    if isinstance(repo_or_dir, str) and repo_or_dir.lower() == "facebookresearch/dinov2":
        kwargs.setdefault("source", "local")
        kwargs.setdefault("trust_repo", True)
        repo_or_dir = D2_LOCAL  # absolute, verified above
    return _orig_hub_load(repo_or_dir, model, *args, **kwargs)
torch.hub.load = _localize_hub
# ---------------------------------------------------------------------------

from gazelle.model import get_gazelle_model

# ---- Configuration ---------------------------------------------------------
# Keep model + checkpoint consistent: use vitb14_inout with vitb14 weights
MODEL_NAME = "gazelle_dinov2_vitl14_inout"
CHECKPOINT = "./checkpoints/gazelle_dinov2_vitl14.pt" 

CAM_INDEX = 0
ALPHA = 0.6
BETA  = 0.4
# ---------------------------------------------------------------------------

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
