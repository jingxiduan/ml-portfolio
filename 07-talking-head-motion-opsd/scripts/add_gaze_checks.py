"""Add eye-gaze (iris vs eye-corner) to outputs/persona_clip_checks.json."""
import json, os, sys, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import video_metrics as vm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

opts = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=vm.MP_MODEL),
    running_mode=vision.RunningMode.IMAGE, output_face_blendshapes=True, num_faces=1)
LMK = vision.FaceLandmarker.create_from_options(opts)
import mediapipe as _mp
MPIMG = _mp.Image

def gaze(path, n=10):
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    gx, gy = [], []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total*(i+0.5)/n)); ok, fr = cap.read()
        if not ok: continue
        r = LMK.detect(MPIMG(image_format=_mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        if not r.face_landmarks: continue
        pts = r.face_landmarks[0]
        if len(pts) < 478: continue
        def c(a, b):
            return np.array([(pts[a].x+pts[b].x)/2, (pts[a].y+pts[b].y)/2])
        for ic, e0, e1 in ((468, 33, 133), (473, 362, 263)):
            iris = np.array([pts[ic].x, pts[ic].y]); ec = c(e0, e1)
            w = np.linalg.norm(np.array([pts[e1].x, pts[e1].y]) - np.array([pts[e0].x, pts[e0].y])) + 1e-6
            d = (iris - ec) / w
            gx.append(d[0]); gy.append(d[1])
    cap.release()
    if not gx: return {}
    return {"gaze_x_mean": float(np.mean(gx)), "gaze_x_abs": float(np.mean(np.abs(gx))),
            "gaze_y_mean": float(np.mean(gy)), "gaze_y_abs": float(np.mean(np.abs(gy)))}

rows = json.load(open(os.path.join(ROOT, "outputs", "persona_clip_checks.json")))
for r in rows:
    r.update(gaze(r["base"]))
    print("%-12s %s %-4s gaze_x=%+.3f |x|=%.3f gaze_y=%+.3f |y|=%.3f" % (
        r["industry"], r["idx"], r["portrait"], r.get("gaze_x_mean", 0),
        r.get("gaze_x_abs", 0), r.get("gaze_y_mean", 0), r.get("gaze_y_abs", 0)), flush=True)
json.dump(rows, open(os.path.join(ROOT, "outputs", "persona_clip_checks.json"), "w"), indent=1)
print("[ok]")
