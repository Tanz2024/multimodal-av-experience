"""
Gesture WS server (FastAPI) — Windows camera fixes + reliable debug window

Fixes included:
- Strong camera open logic (tries DSHOW, MSMF, DEFAULT + env override GESTURE_CAMERA)
- Hard fail logging when camera can't open / can't read frames
- Debug window ALWAYS shows when GESTURE_DEBUG_WINDOW=1
- WebSocket does NOT require client to send messages
- Finger count 0..5 + thumbs_up + pinch + smooth volume 0..1 and 0..100
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional, Tuple, List, Union

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ml" / "artifacts" / "gesture_model.joblib"

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

DEBUG_WINDOW = (os.getenv("GESTURE_DEBUG_WINDOW") or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    send_hz: float = 10.0

    camera_index: int = 0
    reconnect_sec: float = 2.0

    count_stable_n: int = 3
    thumbs_stable_n: int = 3

    pinch_in_thresh: float = 0.05
    pinch_out_thresh: float = 0.08
    pinch_cooldown_sec: float = 1.0

    model_enabled: bool = True
    model_conf_thresh: float = 0.85
    model_stable_n: int = 7
    model_cooldown_sec: float = 0.9

    vol_lower_ratio: float = 0.04
    vol_upper_ratio: float = 0.28


SETTINGS = Settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gesture-fastapi")

clients = set()
queue: asyncio.Queue = asyncio.Queue()
stop_event = Event()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

app = FastAPI()
app.state.last_payload = None
app.state.last_sent_ts = None
app.state.camera_ok = False
app.state.model_ok = True


# ----------------------------
# Helpers
# ----------------------------
def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    n1 = (bax * bax + bay * bay) ** 0.5
    n2 = (bcx * bcx + bcy * bcy) ** 0.5
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cosv = max(-1.0, min(1.0, dot / (n1 * n2)))
    return float(np.degrees(np.arccos(cosv)))


def _finger_count(hand_landmarks) -> Tuple[int, bool, List[bool], bool]:
    coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

    # Thumb
    thumb_mcp = coords[2]
    thumb_ip = coords[3]
    thumb_tip = coords[4]
    thumb_ang = _angle(thumb_mcp, thumb_ip, thumb_tip)
    thumb_mcp_tip = ((thumb_tip[0] - thumb_mcp[0]) ** 2 + (thumb_tip[1] - thumb_mcp[1]) ** 2) ** 0.5
    thumb_mcp_ip = ((thumb_ip[0] - thumb_mcp[0]) ** 2 + (thumb_ip[1] - thumb_mcp[1]) ** 2) ** 0.5
    thumb_extended = bool(thumb_ang >= 155.0 and thumb_mcp_ip > 0.0 and thumb_mcp_tip >= (thumb_mcp_ip * 1.15))

    # Fingers
    mcp_ids = [5, 9, 13, 17]
    pip_ids = [6, 10, 14, 18]
    tip_ids = [8, 12, 16, 20]

    fingers_up: List[bool] = []
    count = 1 if thumb_extended else 0
    for i in range(4):
        ang = _angle(coords[mcp_ids[i]], coords[pip_ids[i]], coords[tip_ids[i]])
        tip_y = coords[tip_ids[i]][1]
        pip_y = coords[pip_ids[i]][1]
        is_up = bool(ang >= 160.0 and (tip_y + 0.015) < pip_y)
        fingers_up.append(is_up)
        if is_up:
            count += 1

    # Thumbs up pose
    dx = abs(thumb_tip[0] - thumb_mcp[0])
    dy = thumb_mcp[1] - thumb_tip[1]
    thumb_up_pose = bool(thumb_extended and dy > 0.05 and dy >= (dx * 0.8))
    thumbs_up = bool(thumb_up_pose and (sum(1 for up in fingers_up if up) <= 1))

    return count, thumbs_up, fingers_up, thumb_extended


def _broadcast_sync(loop: asyncio.AbstractEventLoop, payload: dict) -> None:
    if loop and not loop.is_closed():
        loop.call_soon_threadsafe(queue.put_nowait, payload)


def _probe_camera_read(cap: cv2.VideoCapture, tries: int = 20, sleep_sec: float = 0.03) -> bool:
    for _ in range(max(1, int(tries))):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0):
            return True
        time.sleep(max(0.0, float(sleep_sec)))
    return False


def _parse_gesture_camera_env() -> Tuple[List[int], Optional[str]]:
    """
    GESTURE_CAMERA supports:
      - "0"
      - "0,1,2"
      - a pipeline/path/url string
    Returns (camera_indices, pipeline_str)
    """
    src = (os.getenv("GESTURE_CAMERA") or "").strip()
    if not src:
        return [], None

    if "," in src and all(p.strip().isdigit() for p in src.split(",") if p.strip()):
        return [int(p.strip()) for p in src.split(",") if p.strip()], None

    if src.isdigit():
        return [int(src)], None

    return [], src


def _try_open_index(idx: int) -> Optional[cv2.VideoCapture]:
    """
    On Windows try DSHOW then MSMF then DEFAULT.
    On others try DEFAULT.
    """
    if os.name == "nt":
        # DSHOW
        cap = cv2.VideoCapture(int(idx), cv2.CAP_DSHOW)
        if cap.isOpened() and _probe_camera_read(cap):
            logger.info("Camera opened (index=%s, backend=DSHOW)", idx)
            return cap
        cap.release()

        # MSMF
        cap = cv2.VideoCapture(int(idx), cv2.CAP_MSMF)
        if cap.isOpened() and _probe_camera_read(cap):
            logger.info("Camera opened (index=%s, backend=MSMF)", idx)
            return cap
        cap.release()

    # DEFAULT
    cap = cv2.VideoCapture(int(idx))
    if cap.isOpened() and _probe_camera_read(cap):
        logger.info("Camera opened (index=%s, backend=DEFAULT)", idx)
        return cap
    cap.release()

    return None


def _open_camera() -> Optional[cv2.VideoCapture]:
    indices, pipeline = _parse_gesture_camera_env()

    # Pipeline / URL / file / gstreamer
    if pipeline:
        # Try gstreamer first
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened() and _probe_camera_read(cap):
            logger.info("Camera opened (pipeline via GStreamer)")
            return cap
        cap.release()

        # Try default backend too (some URLs/files work without gstreamer)
        cap = cv2.VideoCapture(pipeline)
        if cap.isOpened() and _probe_camera_read(cap):
            logger.info("Camera opened (pipeline via DEFAULT)")
            return cap
        cap.release()

        logger.warning("Failed to open pipeline source from GESTURE_CAMERA")
        return None

    # Indices
    if not indices:
        indices = [SETTINGS.camera_index, 0, 1, 2]

    tried: List[int] = []
    for idx in [c for i, c in enumerate(indices) if c not in indices[:i]]:
        tried.append(idx)
        cap = _try_open_index(idx)
        if cap is not None:
            return cap

    logger.warning("Camera open failed. Tried indices: %s", tried)
    return None


# ----------------------------
# Worker
# ----------------------------
def gesture_worker(loop: asyncio.AbstractEventLoop) -> None:
    recent_count = deque(maxlen=SETTINGS.count_stable_n)
    recent_thumbs = deque(maxlen=SETTINGS.thumbs_stable_n)
    recent_model = deque(maxlen=SETTINGS.model_stable_n)

    pinch_state = False
    last_pinch_trigger = 0.0

    last_model_trigger = 0.0
    last_sent = 0.0
    last_log = 0.0

    last_frame_ts = None
    fps_smooth = 0.0
    vol_smooth: Optional[float] = None

    model = None
    if SETTINGS.model_enabled and joblib and MODEL_PATH.exists():
        try:
            model = joblib.load(str(MODEL_PATH))
            logger.info("Loaded gesture model: %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load gesture model (%s): %s", MODEL_PATH, exc)
            app.state.model_ok = False

    cap: Optional[cv2.VideoCapture] = None

    try:
        while not stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap = _open_camera()
                if cap is None:
                    app.state.camera_ok = False
                    logger.error("Camera open failed. Retrying in %ss...", SETTINGS.reconnect_sec)
                    time.sleep(SETTINGS.reconnect_sec)
                    continue

                app.state.camera_ok = True
                logger.info("Camera OK")

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as hands:
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        logger.warning("Camera read failed (ok=%s). Reopening...", ok)
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)

                    gesture = "none"
                    stable = False
                    conf = 1.0

                    finger_count: Optional[int] = None
                    count_stable = False

                    thumbs_up = False
                    thumbs_up_stable = False
                    fingers_up: List[bool] = []
                    thumb_extended = False

                    pinch = False
                    pinch_dist = None

                    model_label = None
                    model_conf = None
                    model_stable = False

                    volume_level = None
                    volume_percent = None

                    now = time.time()

                    if res.multi_hand_landmarks:
                        hand = res.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                        raw_count, thumbs_up, fingers_up, thumb_extended = _finger_count(hand)
                        finger_count = int(raw_count)  # 0..5
                        recent_count.append(finger_count)
                        if len(recent_count) == SETTINGS.count_stable_n and len(set(recent_count)) == 1:
                            count_stable = True

                        recent_thumbs.append(bool(thumbs_up))
                        thumbs_up_stable = len(recent_thumbs) == SETTINGS.thumbs_stable_n and all(recent_thumbs)

                        # Pinch distance (normalized)
                        coords = [(lm.x, lm.y) for lm in hand.landmark]
                        dx = coords[4][0] - coords[8][0]
                        dy = coords[4][1] - coords[8][1]
                        pinch_dist = float((dx * dx + dy * dy) ** 0.5)
                        pinch = pinch_dist < SETTINGS.pinch_in_thresh

                        # FPS smoothing
                        if last_frame_ts is not None:
                            dt = max(1e-3, now - last_frame_ts)
                            fps = 1.0 / dt
                            fps_smooth = fps if fps_smooth <= 0.0 else (fps_smooth * 0.9 + fps * 0.1)
                        last_frame_ts = now

                        # Volume mapping (pixels)
                        h2, w2, _ = frame.shape
                        min_dim = float(min(w2, h2))
                        lower = SETTINGS.vol_lower_ratio * min_dim
                        upper = SETTINGS.vol_upper_ratio * min_dim

                        thumb_px = (int(coords[4][0] * w2), int(coords[4][1] * h2))
                        index_px = (int(coords[8][0] * w2), int(coords[8][1] * h2))
                        line_len_px = float(
                            ((thumb_px[0] - index_px[0]) ** 2 + (thumb_px[1] - index_px[1]) ** 2) ** 0.5
                        )

                        if upper <= lower:
                            vol_raw = 0.0
                        else:
                            vol_raw = (line_len_px - lower) / (upper - lower)
                        vol_raw = float(max(0.0, min(1.0, vol_raw)))
                        vol_smooth = vol_raw if vol_smooth is None else (vol_smooth * 0.85 + vol_raw * 0.15)

                        volume_level = float(vol_smooth)
                        volume_percent = int(volume_level * 100)

                        # Pinch events (hysteresis + cooldown)
                        if pinch_state:
                            if pinch_dist > SETTINGS.pinch_out_thresh:
                                if now - last_pinch_trigger >= SETTINGS.pinch_cooldown_sec:
                                    gesture = "squeeze_out"
                                    stable = True
                                    last_pinch_trigger = now
                                pinch_state = False
                        else:
                            if pinch_dist < SETTINGS.pinch_in_thresh:
                                if now - last_pinch_trigger >= SETTINGS.pinch_cooldown_sec:
                                    gesture = "squeeze_in"
                                    stable = True
                                    last_pinch_trigger = now
                                pinch_state = True

                        # Optional ML classifier
                        if model is not None:
                            h, w, _ = frame.shape
                            xs = [lm.x for lm in hand.landmark]
                            ys = [lm.y for lm in hand.landmark]
                            x1 = max(int(min(xs) * w) - 20, 0)
                            y1 = max(int(min(ys) * h) - 20, 0)
                            x2 = min(int(max(xs) * w) + 20, w - 1)
                            y2 = min(int(max(ys) * h) + 20, h - 1)

                            crop = frame[y1:y2, x1:x2]
                            if crop.size != 0:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
                                feat = resized.astype(np.float32).flatten() / 255.0
                                X = np.array(feat, dtype=np.float32).reshape(1, -1)

                                probs = model.predict_proba(X)[0]
                                idx = int(np.argmax(probs))
                                model_conf = float(probs[idx])
                                pred = str(model.classes_[idx])

                                if model_conf >= SETTINGS.model_conf_thresh:
                                    recent_model.append(pred)
                                else:
                                    recent_model.append("uncertain")

                                if (
                                    len(recent_model) == SETTINGS.model_stable_n
                                    and len(set(recent_model)) == 1
                                    and recent_model[0] != "uncertain"
                                ):
                                    model_label = recent_model[0]
                                    model_stable = True
                                else:
                                    model_label = pred
                                    model_stable = False

                                conf = model_conf
                            else:
                                recent_model.clear()

                        # Map stable ML gesture triggers
                        if model_label:
                            now2 = time.time()
                            if model_stable and (now2 - last_model_trigger) >= SETTINGS.model_cooldown_sec:
                                if model_label == "thumbs":
                                    gesture = "thumbs"
                                    stable = True
                                elif model_label == "fist":
                                    gesture = "fist"
                                    stable = True
                                elif model_label == "rad":
                                    gesture = "squeeze_in"
                                    stable = True
                                elif model_label == "straight":
                                    gesture = "squeeze_out"
                                    stable = True
                                elif model_label in ("peace", "okay", "five"):
                                    gesture = model_label
                                    stable = True
                                last_model_trigger = now2

                    payload = {
                        "gesture": gesture,
                        "confidence": round(float(conf), 3),
                        "stable": bool(stable),

                        "finger_count": finger_count,
                        "count_stable": bool(count_stable),

                        "thumbs_up": bool(thumbs_up),
                        "thumbs_up_stable": bool(thumbs_up_stable) if res.multi_hand_landmarks else False,

                        "pinch": bool(pinch),
                        "pinch_dist": round(float(pinch_dist), 3) if pinch_dist is not None else None,

                        "volume_level": round(float(volume_level), 3) if volume_level is not None else None,
                        "volume_percent": int(volume_percent) if volume_percent is not None else None,

                        "ai_gesture": model_label,
                        "ai_confidence": round(float(model_conf), 3) if model_conf is not None else None,
                        "ai_stable": bool(model_stable) if model_label is not None else False,

                        # debug fields
                        "thumb_extended": bool(thumb_extended),
                        "fingers_up": fingers_up,

                        "fps": round(float(fps_smooth), 1) if fps_smooth > 0.0 else None,
                        "timestamp": time.time(),
                    }

                    # Send at fixed rate
                    now_send = time.time()
                    send_interval = 1.0 / max(1.0, SETTINGS.send_hz)
                    if (now_send - last_sent) >= send_interval:
                        last_sent = now_send
                        _broadcast_sync(loop, payload)
                        app.state.last_payload = payload
                        app.state.last_sent_ts = now_send

                        if (now_send - last_log) >= 1.0:
                            logger.info(
                                "WS SEND finger=%s thumbs=%s vol=%s%% gesture=%s camera_ok=%s",
                                payload["finger_count"],
                                payload["thumbs_up"],
                                payload["volume_percent"],
                                payload["gesture"],
                                app.state.camera_ok,
                            )
                            last_log = now_send

                        # Debug window
                        if DEBUG_WINDOW:
                            cv2.putText(
                                frame,
                                f"Gesture: {payload['gesture']} conf:{payload['confidence']}",
                                (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                            if payload["fps"] is not None:
                                cv2.putText(
                                    frame,
                                    f"FPS: {payload['fps']}",
                                    (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 0, 255),
                                    2,
                                )

                            if payload["finger_count"] is not None:
                                cv2.putText(
                                    frame,
                                    f"Fingers: {payload['finger_count']} (stable:{payload['count_stable']})",
                                    (20, 85),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 200, 0),
                                    2,
                                )

                            if payload["thumbs_up"]:
                                cv2.putText(
                                    frame,
                                    "Thumbs Up",
                                    (20, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 200, 0),
                                    2,
                                )

                            if payload["volume_percent"] is not None:
                                cv2.putText(
                                    frame,
                                    f"Volume: {payload['volume_percent']}%",
                                    (20, 135),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2,
                                )

                            if res.multi_hand_landmarks:
                                h2, w2, _ = frame.shape
                                coords2 = [(lm.x, lm.y) for lm in res.multi_hand_landmarks[0].landmark]
                                thumb_px = (int(coords2[4][0] * w2), int(coords2[4][1] * h2))
                                index_px = (int(coords2[8][0] * w2), int(coords2[8][1] * h2))
                                cv2.circle(frame, thumb_px, 8, (255, 0, 255), cv2.FILLED)
                                cv2.circle(frame, index_px, 8, (255, 0, 255), cv2.FILLED)
                                cv2.line(frame, thumb_px, index_px, (255, 0, 255), 3)

                            cv2.putText(
                                frame,
                                "Press Q to quit",
                                (20, 165),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )

                            cv2.imshow("Gesture Debug", frame)
                            k = cv2.waitKey(1) & 0xFF
                            if k in (ord("q"), ord("Q")):
                                stop_event.set()
                                break

    except Exception:
        logger.exception("Gesture worker crashed")
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()


# ----------------------------
# Async broadcaster
# ----------------------------
async def broadcaster():
    while True:
        payload = await queue.get()
        if not clients:
            continue
        msg = json.dumps(payload)
        await asyncio.gather(*[ws.send_text(msg) for ws in list(clients)], return_exceptions=True)


# ----------------------------
# API routes
# ----------------------------
@app.get("/health")
async def health():
    return {"ok": True, "camera_ok": app.state.camera_ok, "model_ok": app.state.model_ok}


@app.get("/status")
async def status():
    return {
        "last_payload": app.state.last_payload,
        "last_sent_ts": app.state.last_sent_ts,
        "camera_ok": app.state.camera_ok,
        "model_ok": app.state.model_ok,
        "clients": len(clients),
    }


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    app.state.loop = loop
    app.state.broadcaster_task = asyncio.create_task(broadcaster())
    app.state.worker_task = asyncio.create_task(asyncio.to_thread(gesture_worker, loop))


@app.on_event("shutdown")
async def on_shutdown():
    stop_event.set()
    task = getattr(app.state, "broadcaster_task", None)
    if task:
        task.cancel()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    # Send initial payload
    initial = app.state.last_payload or {
        "gesture": "none",
        "confidence": 0.0,
        "stable": False,
        "finger_count": None,
        "count_stable": False,
        "thumbs_up": False,
        "thumbs_up_stable": False,
        "pinch": False,
        "pinch_dist": None,
        "volume_level": None,
        "volume_percent": None,
        "ai_gesture": None,
        "ai_confidence": None,
        "ai_stable": False,
        "fps": None,
        "timestamp": time.time(),
    }

    try:
        await websocket.send_text(json.dumps(initial))
    except Exception:
        clients.discard(websocket)
        return

    # IMPORTANT: do NOT require client to send anything
    try:
        while True:
            await asyncio.sleep(60)
            # Optional keep-alive ping (uncomment if needed)
            # await websocket.send_text(json.dumps({"type": "ping", "ts": time.time()}))
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
