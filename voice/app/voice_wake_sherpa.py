import asyncio
import json
import logging
import math
import os
import queue
import subprocess
import tempfile
import time
from threading import Event, Lock, Thread
from urllib.parse import urlencode

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


try:
    from dotenv import load_dotenv as _load_dotenv
    _override = os.getenv("DOTENV_OVERRIDE", "true").lower() == "true"
    _load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=_override)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set before launch

from .voice_helpers import (
    Settings,
    normalize_tokens,
    norm_join,
    load_intent_config,
    classify_intent,
    match_intent_contains,
    CommandExecutor,
    AudioDucker,
    play_wav,
    resolve_device,
    resolve_sample_rate,
    resolve_input_channels,
    resolve_openwakeword_model_path,
    OpenWakeWordDetector,
    NoiseReducer,
    VOICE_PRESET,
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8010"))

SETTINGS = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("voice-wake-sherpa")

clients = set()
event_queue = queue.Queue()
tts_queue = queue.Queue()
audio_queue = queue.Queue(maxsize=100)
stop_event = Event()
tts_stop_event = Event()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.last_payload = None
app.state.last_sent_ts = None
app.state.voice_ok = False
app.state.mode = "SLEEPING"
app.state.cooldown_until = 0.0
app.state.noise_reducer = None
app.state.session = {
    "mode": "SLEEPING",
    "transcript_partial": "",
    "last_final": "",
    "selected_experience": None,
    "selected_option": None,
    "navigate": None,
    "volume": 0.0,
    "last_intent": None,
    "last_response": "",
    "intent_seq": 0,
    "voice_rms": 0.0,
    # If noise reduction is disabled, there's no profile to collect.
    "noise_profile_ready": (not SETTINGS.noise_reduction_enabled),
    # OpenWakeWord debug (filled while sleeping).
    "oww_score": 0.0,
    "oww_threshold": float(getattr(SETTINGS, "wakeword_threshold", 0.0) or 0.0),
    "debug": "",
    "always_listening": False,
    "listen_until": 0.0,
    "asr_source": "none",
    # Transcript-first debug fields
    "last_deepgram_text": "",
    "last_deepgram_is_final": False,
    "last_final_raw": "",
    "transcript_log": [],
}
app.state.metrics = {
    "last_wake_ts": 0.0,
    "last_final_ts": 0.0,
    "last_tts_ts": 0.0,
    "wakes": 0,
    "audio_frames": 0,
    "commands_executed": 0,
    "asr_errors": 0,
    "tts_errors": 0,
    "audio_queue_drops": 0,
}


def _emit(payload):
    event_queue.put(_json_sanitize(payload))


def _json_sanitize(obj):
    """
    Starlette uses strict JSON rendering (no NaN/Inf). Sanitize any non-finite
    floats (and numpy scalars) to keep endpoints and WS events stable.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    # Numpy scalars (or similar) often have .item()
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_sanitize(item())
        except Exception:
            return str(obj)

    return str(obj)


def _emit_state():
    _emit(
        _json_sanitize(
            {"type": "state_update", "source": "voice", "state": dict(app.state.session)}
        )
    )


def _enqueue_tts(text: str):
    if not SETTINGS.tts_enabled:
        return
    tts_queue.put(text)


def _stop_intent(text: str) -> bool:
    t = norm_join(text)
    return "stop" in t or "all off" in t or "sound off" in t or "mute" in t


def _strip_wake_prefix(text: str) -> str:
    normalized = norm_join(text)
    if not normalized:
        return ""

    wake_phrases = []
    if SETTINGS.wake_phrase:
        wake_phrases.append(norm_join(SETTINGS.wake_phrase))
    if SETTINGS.wake_word:
        wake_phrases.append(norm_join(SETTINGS.wake_word))

    wake_phrases = [p for p in wake_phrases if p]
    for phrase in wake_phrases:
        if normalized == phrase:
            return ""
        prefix = f"{phrase} "
        if normalized.startswith(prefix):
            return normalized[len(prefix):].strip()

    return normalized


def _load_intent_bundle(path: str):
    phrases, responses, voice_preset, classifier = load_intent_config(path)
    return {
        "path": path,
        "phrases": phrases,
        "responses": responses,
        "voice_preset": voice_preset,
        "classifier": classifier,
    }


def _load_intent_bundles():
    deep_path = SETTINGS.intent_config_deepgram_path or SETTINGS.intent_config_path
    vosk_path = SETTINGS.intent_config_vosk_path or SETTINGS.intent_config_path
    bundles = {
        "deepgram": _load_intent_bundle(deep_path),
        "vosk": _load_intent_bundle(vosk_path),
    }
    return bundles


INTENT_BUNDLES = _load_intent_bundles()
VOICE_PRESET = dict(INTENT_BUNDLES.get("deepgram", {}).get("voice_preset", VOICE_PRESET))
executor = CommandExecutor(app)


def _intent_bundle_for_source(source: str | None):
    if source == "vosk":
        return INTENT_BUNDLES.get("vosk") or INTENT_BUNDLES.get("deepgram")
    return INTENT_BUNDLES.get("deepgram") or INTENT_BUNDLES.get("vosk")


def match_intent_exact(text: str, phrases: dict[str, list[str]]) -> str | None:
    normalized = norm_join(text)
    if not normalized:
        return None
    for intent, examples in phrases.items():
        for ex in examples:
            if normalized == norm_join(ex):
                return intent
    return None


class DeepgramStream:
    def __init__(self, settings: Settings, sample_rate: int):
        self.api_key = (settings.deepgram_api_key or "").strip()
        self.model = (settings.deepgram_model or "nova-3").strip()
        self.language = (settings.deepgram_language or "en-US").strip()
        self.endpointing_ms = int(max(0, settings.deepgram_endpointing_ms or 0))
        self.sample_rate = int(sample_rate)
        self.events = queue.Queue()
        self.connected = False
        self._ws = None
        self._ws_kind = ""
        self._lock = Lock()
        self._receiver = None

    def start(self):
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        create_connection = None
        ws_sync_connect = None
        try:
            from websocket import create_connection as _create_connection
            create_connection = _create_connection
        except Exception as exc:
            try:
                import websocket as _websocket
                create_connection = getattr(_websocket, "create_connection", None)
            except Exception:
                create_connection = None
            if create_connection is None:
                try:
                    from websockets.sync.client import connect as _ws_sync_connect
                    ws_sync_connect = _ws_sync_connect
                except Exception:
                    ws_sync_connect = None
                if ws_sync_connect is None:
                    raise RuntimeError(f"Missing websocket client dependency: {exc}") from exc

        params = {
            "model": self.model,
            "language": self.language,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "channels": "1",
            "interim_results": "true",
            "endpointing": str(self.endpointing_ms),
            "punctuate": "true",
            "smart_format": "true",
        }
        url = "wss://api.deepgram.com/v1/listen?" + urlencode(params)
        header_list = [f"Authorization: Token {self.api_key}"]
        header_map = {"Authorization": f"Token {self.api_key}"}

        if create_connection is not None:
            ws = create_connection(url, header=header_list, timeout=10, enable_multithread=True)
            self._ws_kind = "websocket-client"
        else:
            ws = ws_sync_connect(url, additional_headers=header_map, open_timeout=10)
            self._ws_kind = "websockets-sync"
        self._ws = ws
        self.connected = True

        def _recv_loop():
            while self.connected:
                try:
                    msg = ws.recv()
                except Exception as exc:
                    self.events.put({"type": "Error", "error": str(exc)})
                    break
                if msg is None:
                    break
                if isinstance(msg, bytes):
                    continue
                try:
                    payload = json.loads(msg)
                except Exception:
                    continue
                self.events.put(payload)
            self.connected = False

        self._receiver = Thread(target=_recv_loop, daemon=True)
        self._receiver.start()

    def send_audio(self, pcm_bytes: bytes):
        if not self.connected or not pcm_bytes:
            return
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        try:
            if self._ws_kind == "websocket-client":
                ws.send_binary(pcm_bytes)
            else:
                ws.send(pcm_bytes)
        except Exception as exc:
            self.connected = False
            self.events.put({"type": "Error", "error": str(exc)})

    def drain_events(self, limit: int = 64) -> list[dict]:
        out = []
        for _ in range(limit):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self):
        self.connected = False
        with self._lock:
            ws = self._ws
            self._ws = None
        if ws is None:
            return
        try:
            if self._ws_kind == "websocket-client":
                ws.send(json.dumps({"type": "CloseStream"}))
            else:
                ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass


class VoskStream:
    def __init__(self, settings: Settings, sample_rate: int):
        self.model_path = (settings.vosk_model_path or "").strip()
        self.sample_rate = int(sample_rate)
        self.available = False
        self._model = None
        self._rec = None
        self._last_partial = ""

    def start(self):
        if not self.model_path:
            raise RuntimeError("VOSK_MODEL_PATH is not set")
        if not os.path.isdir(self.model_path):
            raise RuntimeError(f"VOSK model path not found: {self.model_path}")
        try:
            import vosk
        except Exception as exc:
            raise RuntimeError(f"Missing vosk dependency: {exc}") from exc
        self._model = vosk.Model(self.model_path)
        self._rec = vosk.KaldiRecognizer(self._model, float(self.sample_rate))
        try:
            self._rec.SetWords(False)
        except Exception:
            pass
        self.available = True

    def process_audio(self, pcm_bytes: bytes) -> list[dict]:
        if not self.available or self._rec is None or not pcm_bytes:
            return []
        out = []
        try:
            if self._rec.AcceptWaveform(pcm_bytes):
                res = json.loads(self._rec.Result() or "{}")
                txt = (res.get("text") or "").strip()
                if txt:
                    self._last_partial = ""
                    out.append(
                        {
                            "type": "Results",
                            "channel": {"alternatives": [{"transcript": txt}]},
                            "is_final": True,
                            "source": "vosk",
                        }
                    )
            else:
                part = json.loads(self._rec.PartialResult() or "{}")
                txt = (part.get("partial") or "").strip()
                if txt and txt != self._last_partial:
                    self._last_partial = txt
                    out.append(
                        {
                            "type": "Results",
                            "channel": {"alternatives": [{"transcript": txt}]},
                            "is_final": False,
                            "source": "vosk",
                        }
                    )
        except Exception as exc:
            out.append({"type": "Error", "error": str(exc), "source": "vosk"})
        return out


def tts_worker():
    ducker = AudioDucker(SETTINGS)
    while not stop_event.is_set():
        text = tts_queue.get()
        if text is None:
            break
        if not SETTINGS.tts_enabled:
            continue
        if SETTINGS.tts_backend != "piper":
            continue

        preset_model_path = VOICE_PRESET.get("model_path", "")
        model_path = preset_model_path or SETTINGS.piper_model_path
        if not model_path:
            logger.warning("PIPER_MODEL_PATH not set; skipping TTS")
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name

            length_scale = str(VOICE_PRESET.get("length_scale", SETTINGS.piper_length_scale))
            noise_scale = VOICE_PRESET.get("noise_scale", SETTINGS.piper_noise_scale)
            noise_w = VOICE_PRESET.get("noise_w", SETTINGS.piper_noise_w)

            cmd = [
                SETTINGS.piper_path,
                "--model",
                model_path,
                "--output_file",
                out_path,
                "--length_scale",
                length_scale,
            ]
            if noise_scale not in (None, ""):
                cmd.extend(["--noise_scale", str(noise_scale)])
            if noise_w not in (None, ""):
                cmd.extend(["--noise_w", str(noise_w)])
            if SETTINGS.piper_speaker:
                cmd.extend(["--speaker", SETTINGS.piper_speaker])

            ducker.duck()
            tts_stop_event.clear()
            app.state.metrics["last_tts_ts"] = time.time()

            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                app.state.metrics["tts_errors"] += 1
                err = (proc.stderr or b"")[:4000].decode("utf-8", errors="replace").strip()
                out = (proc.stdout or b"")[:2000].decode("utf-8", errors="replace").strip()
                logger.warning("Piper exited with code %s. stderr=%r stdout=%r", proc.returncode, err, out)
                continue

            try:
                if (not os.path.isfile(out_path)) or (os.path.getsize(out_path) <= 0):
                    app.state.metrics["tts_errors"] += 1
                    logger.warning("TTS output wav missing/empty: %s", out_path)
                    continue
            except Exception as exc:
                app.state.metrics["tts_errors"] += 1
                logger.warning("TTS output wav check failed: %s", exc)
                continue

            ok = play_wav(out_path, tts_stop_event)
            if not ok:
                app.state.metrics["tts_errors"] += 1
                logger.warning("TTS playback failed")

        except Exception as exc:
            app.state.metrics["tts_errors"] += 1
            logger.warning("TTS failed: %s", exc)

        finally:
            ducker.unduck()
            try:
                os.remove(out_path)
            except Exception:
                pass
            # Brief post-TTS silence window so the mic doesn't pick up TTS echo
            if SETTINGS.tts_cooldown_sec > 0:
                time.sleep(SETTINGS.tts_cooldown_sec)


def voice_worker(loop):
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        logger.error("Missing deps for voice: %s", exc)
        return

    def _select_loudest_channel_int16(data: bytes, channels: int) -> tuple[bytes, int, list[float]]:
        """
        sounddevice RawInputStream returns interleaved int16 PCM when channels > 1.
        Some USB mics expose multiple channels where one may be silent; pick the
        channel with the highest RMS per frame and return mono int16 bytes.
        """
        try:
            if channels is None or int(channels) <= 1:
                return data, 0, []
            ch = int(channels)
            x = np.frombuffer(data, dtype=np.int16)
            frames = int(len(x) // ch)
            if frames <= 0:
                return data, 0, []
            x = x[: frames * ch].reshape(frames, ch).astype(np.float32)
            rms_by_ch = np.sqrt(np.mean((x / 32768.0) ** 2, axis=0)).astype(float).tolist()
            best = int(np.argmax(np.asarray(rms_by_ch, dtype=np.float32)))
            mono = x[:, best].astype(np.int16).tobytes()
            return mono, best, [float(r) for r in rms_by_ch]
        except Exception:
            return data, 0, []

    def _resample_f32(audio_f32, src_sr: int, dst_sr: int):
        if src_sr == dst_sr:
            return audio_f32
        if audio_f32 is None or len(audio_f32) == 0:
            return audio_f32
        try:
            src_sr = int(src_sr)
            dst_sr = int(dst_sr)
            if src_sr <= 0 or dst_sr <= 0:
                return audio_f32
            new_len = int(round(len(audio_f32) * (dst_sr / src_sr)))
            if new_len <= 1:
                return audio_f32[:1]
            x_old = np.linspace(0.0, 1.0, num=len(audio_f32), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            return np.interp(x_new, x_old, audio_f32).astype(np.float32)
        except Exception:
            return audio_f32

    detector = OpenWakeWordDetector(
        SETTINGS.wakeword_model_path,
        threshold=SETTINGS.wakeword_threshold,
    )
    if detector.available:
        logger.info("OpenWakeWord loaded: %s", SETTINGS.wakeword_model_path)
    else:
        logger.warning("OpenWakeWord not available. Fallback to Deepgram wake phrase.")

    audio_source = SETTINGS.audio_source
    always_listening = SETTINGS.always_listening or SETTINGS.wake_word == ""
    if audio_source not in ("device", "browser"):
        audio_source = "device"

    target_sample_rate = int(getattr(SETTINGS, "sample_rate", 16000) or 16000)

    if audio_source == "browser":
        device = None
        input_sample_rate = target_sample_rate
    else:
        device = resolve_device(SETTINGS)
        input_sample_rate = resolve_sample_rate(SETTINGS, device)

    noise_reducer = NoiseReducer(SETTINGS, target_sample_rate)
    app.state.noise_reducer = noise_reducer
    deepgram = None
    dg_next_connect_ts = 0.0
    vosk_stream = None
    vosk_ready = False
    if SETTINGS.vosk_model_path:
        try:
            vosk_stream = VoskStream(SETTINGS, target_sample_rate)
            vosk_stream.start()
            vosk_ready = True
            logger.info("Vosk fallback initialized: %s", SETTINGS.vosk_model_path)
        except Exception as exc:
            logger.warning("Vosk fallback disabled: %s", exc)
    else:
        logger.info("Vosk fallback disabled: VOSK_MODEL_PATH not set")

    wake_active = False
    wake_start = 0.0
    listen_until = 0.0
    cooldown_until = 0.0
    last_partial_ts = 0.0
    last_rms_emit = 0.0
    utterance_start = None
    last_speech_ts = 0.0

    fallback_hits = 0
    fallback_window_start = 0.0

    executing_until = 0.0
    executing_hold_sec = float(os.getenv("EXECUTING_HOLD_SEC", "0.2"))

    app.state.voice_ok = True
    wake_prompt_block_sec = float(os.getenv("WAKE_PROMPT_BLOCK_SEC", "1.2"))

    def set_mode(new_mode: str, now: float):
        if app.state.mode == new_mode:
            return
        app.state.mode = new_mode
        app.state.session["mode"] = new_mode
        if new_mode == "SLEEPING":
            app.state.session["transcript_partial"] = ""
        _emit_state()

    if always_listening:
        wake_active = True
        set_mode("LISTENING", time.time())

    app.state.session["always_listening"] = bool(always_listening)
    dg_partial_text = ""
    dg_cmd_finals: list[str] = []
    cmd_asr_source = "deepgram"

    def _ensure_deepgram(now_ts: float) -> bool:
        nonlocal deepgram, dg_next_connect_ts
        if deepgram is not None and deepgram.connected:
            return True
        if now_ts < dg_next_connect_ts:
            return False
        try:
            if deepgram is not None:
                deepgram.close()
            deepgram = DeepgramStream(SETTINGS, target_sample_rate)
            deepgram.start()
            logger.info(
                "Deepgram stream initialized (model=%s language=%s endpointing_ms=%s)",
                SETTINGS.deepgram_model,
                SETTINGS.deepgram_language,
                SETTINGS.deepgram_endpointing_ms,
            )
            app.state.session["debug"] = ""
            app.state.voice_ok = True
            return True
        except Exception as exc:
            app.state.metrics["asr_errors"] += 1
            app.state.session["debug"] = f"Deepgram connect failed: {exc}"
            _emit_state()
            dg_next_connect_ts = now_ts + 2.0
            app.state.voice_ok = False
            logger.warning("Deepgram connect failed: %s", exc)
            return False

    def _reset_cmd_buffer():
        nonlocal dg_partial_text, dg_cmd_finals, cmd_asr_source
        dg_partial_text = ""
        dg_cmd_finals = []
        cmd_asr_source = "deepgram"

    def _record_transcript(text: str, is_final: bool, now_ts: float):
        if not text:
            return
        app.state.session["last_deepgram_text"] = text
        app.state.session["last_deepgram_is_final"] = bool(is_final)
        if not SETTINGS.transcript_debug_enabled:
            return
        max_items = int(max(1, getattr(SETTINGS, "transcript_debug_max", 25)))
        log = app.state.session.get("transcript_log")
        if not isinstance(log, list):
            log = []
        log.append(
            {
                "ts": round(float(now_ts), 3),
                "text": text,
                "is_final": bool(is_final),
            }
        )
        if len(log) > max_items:
            log = log[-max_items:]
        app.state.session["transcript_log"] = log

    def _trigger_wake(now_ts: float):
        nonlocal wake_active, wake_start, listen_until, last_partial_ts, utterance_start
        nonlocal fallback_hits, cooldown_until
        wake_active = True
        wake_start = now_ts
        listen_until = now_ts + SETTINGS.listen_window_sec
        app.state.session["listen_until"] = listen_until
        fallback_hits = 0
        last_partial_ts = 0.0
        utterance_start = None
        _reset_cmd_buffer()
        app.state.metrics["last_wake_ts"] = now_ts
        app.state.metrics["wakes"] += 1
        set_mode("LISTENING", now_ts)
        _emit({"source": "voice", "event": "wake", "timestamp": now_ts})
        deep_bundle = _intent_bundle_for_source("deepgram") or {}
        deep_responses = deep_bundle.get("responses", {}) if isinstance(deep_bundle, dict) else {}
        app.state.session["last_response"] = (
            SETTINGS.wake_response
            or deep_responses.get("greet", "")
            or "How may I help you?"
        )
        app.state.session["last_intent"] = "greet"
        _emit_state()
        _enqueue_tts(app.state.session["last_response"])
        if wake_prompt_block_sec > 0:
            cooldown_until = max(cooldown_until, now_ts + wake_prompt_block_sec)
            app.state.cooldown_until = cooldown_until

    def handle_frame(data):
        nonlocal wake_active, wake_start, listen_until, cooldown_until, last_partial_ts, last_rms_emit
        nonlocal fallback_hits, fallback_window_start, utterance_start, last_speech_ts
        nonlocal executing_until, deepgram, dg_next_connect_ts
        nonlocal dg_partial_text, cmd_asr_source

        now = time.time()
        in_cooldown = now < cooldown_until
        app.state.metrics["audio_frames"] += 1

        audio_f32_in = (np.frombuffer(data, dtype=np.int16).astype(np.float32)) / 32768.0
        if not np.isfinite(audio_f32_in).all():
            audio_f32_in = np.nan_to_num(audio_f32_in, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        rms_raw = float(np.sqrt(np.mean(audio_f32_in * audio_f32_in)))
        audio_f32_raw = _resample_f32(audio_f32_in, input_sample_rate, target_sample_rate)

        # If we just executed something, show EXECUTING briefly then transition to COOLDOWN.
        if app.state.mode == "EXECUTING" and executing_until and now >= executing_until:
            set_mode("COOLDOWN", now)

        # End of cooldown/executing: return to idle/listening even if the room is silent.
        if (
            not in_cooldown
            and app.state.mode in ("COOLDOWN", "EXECUTING")
            and (not wake_active or always_listening)
        ):
            cooldown_until = 0.0
            app.state.cooldown_until = 0.0
            set_mode("LISTENING" if always_listening else "SLEEPING", now)

        # ── Noise profile collection (runs during SLEEPING / idle) ─────────────
        if not wake_active and not always_listening and app.state.mode == "SLEEPING":
            noise_reducer.update_profile(audio_f32_raw)
            if noise_reducer.profile_ready and not app.state.session.get("noise_profile_ready"):
                app.state.session["noise_profile_ready"] = True
                # ── Auto noise gate calibration ────────────────────────────────
                if SETTINGS.noise_gate_auto:
                    residual = noise_reducer.calibrate()
                    if residual is not None:
                        new_gate = residual * SETTINGS.noise_gate_multiplier
                        # Clamp to a safe range so extreme music can't break things.
                        # NOTE: if this is too high, wake/ASR may never see "speech",
                        # especially on laptop mics / browser audio.
                        new_gate = max(0.002, min(0.03, new_gate))
                        SETTINGS.noise_gate_rms = new_gate
                        logger.info(
                            "Auto noise gate set: residual=%.5f × %.1f = %.4f",
                            residual, SETTINGS.noise_gate_multiplier, new_gate,
                        )
                _emit_state()

        # ── Spectral noise reduction ───────────────────────────────────────────
        # Only reduce AFTER the ambient profile is built.
        # Before that, stationary mode would estimate noise from the current chunk —
        # if you're speaking, your voice IS the chunk and gets subtracted from itself.
        if noise_reducer.profile_ready:
            audio_f32 = noise_reducer.reduce(audio_f32_raw)
        else:
            audio_f32 = audio_f32_raw  # Pass raw audio until profile is ready

        # Defensive: some DSP/noise-reduction paths can emit NaN/Inf.
        if not np.isfinite(audio_f32).all():
            audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        rms = float(np.sqrt(np.mean(audio_f32 * audio_f32)))
        app.state.session["voice_rms_raw"] = rms_raw
        app.state.session["voice_rms"] = rms

        if (now - last_rms_emit) > 0.2:
            last_rms_emit = now
            _emit_state()

        gain = SETTINGS.target_rms / max(rms, 1e-6)
        gain = max(1.0, min(SETTINGS.max_gain, gain))
        audio_f32n = np.clip(audio_f32 * gain, -1.0, 1.0)
        try:
            app.state.session["voice_rms_norm"] = float(np.sqrt(np.mean(audio_f32n * audio_f32n)))
        except Exception:
            app.state.session["voice_rms_norm"] = None
        data_norm = (audio_f32n * 32767.0).astype(np.int16).tobytes()

        def reset_listening(to_idle: bool = True):
            nonlocal wake_active, utterance_start, last_speech_ts, listen_until
            wake_active = False
            utterance_start = None
            last_speech_ts = 0.0
            listen_until = 0.0
            app.state.session["listen_until"] = 0.0
            _reset_cmd_buffer()
            if to_idle:
                set_mode("SLEEPING", now)

        def finalize_command():
            nonlocal cooldown_until, wake_active, utterance_start
            nonlocal executing_until

            if not always_listening and app.state.mode != "LISTENING":
                return

            text = " ".join([t for t in dg_cmd_finals if t]).strip()
            if not text:
                text = (dg_partial_text or "").strip()
            _reset_cmd_buffer()
            app.state.session["last_final_raw"] = text

            app.state.session["transcript_partial"] = ""
            _emit_state()

            if not text:
                app.state.session["debug"] = "No speech detected"
                _emit_state()
                if always_listening:
                    set_mode("LISTENING", now)
                else:
                    reset_listening(to_idle=True)
                return

            normalized_text = norm_join(text)
            intent_text = _strip_wake_prefix(text)

            if intent_text:
                text = intent_text
                normalized_text = intent_text

            # In ALWAYS_LISTEN mode, treat a short "jarvis"/"hey jarvis" as a greet so the
            # system still provides the talkback prompt users expect.
            if (
                always_listening
                and len(normalized_text.split()) <= 3
                and (
                    (SETTINGS.wake_phrase and norm_join(SETTINGS.wake_phrase) in normalized_text)
                    or (SETTINGS.wake_word and norm_join(SETTINGS.wake_word) in normalized_text)
                )
            ):
                app.state.session["last_intent"] = "greet"
                app.state.session["last_response"] = SETTINGS.wake_response
                app.state.session["last_final"] = text
                app.state.session["intent_seq"] += 1
                app.state.session["debug"] = ""
                _emit_state()
                _enqueue_tts(app.state.session["last_response"])
                set_mode("LISTENING", now)
                _reset_cmd_buffer()
                return

            payload = {
                "source": "voice",
                "text": text,
                "is_final": True,
                "timestamp": now,
            }
            _emit(payload)
            app.state.metrics["last_final_ts"] = now

            if _stop_intent(text):
                tts_stop_event.set()

            bundle = _intent_bundle_for_source(cmd_asr_source) or {}
            intent_phrases = bundle.get("phrases", {}) if isinstance(bundle, dict) else {}
            intent_responses = bundle.get("responses", {}) if isinstance(bundle, dict) else {}
            intent_classifier = bundle.get("classifier", None) if isinstance(bundle, dict) else None

            intent = match_intent_exact(text, intent_phrases)
            if not intent:
                # Substring fallback: catches intent phrases embedded in longer ASR outputs.
                intent = match_intent_contains(text, intent_phrases)
            if not intent and intent_classifier is not None:
                intent, _ = classify_intent(text, intent_classifier, SETTINGS.intent_threshold)

            if intent:
                app.state.session["last_intent"] = intent
                app.state.session["last_response"] = intent_responses.get(intent, "")
                app.state.session["last_final"] = text
                app.state.session["debug"] = ""
                app.state.session["intent_seq"] += 1
                _emit_state()

                if intent == "cancel":
                    reset_listening(to_idle=True)
                    if app.state.session["last_response"]:
                        _enqueue_tts(app.state.session["last_response"])
                    return

                executor.execute(intent)
                if app.state.session["last_response"]:
                    _enqueue_tts(app.state.session["last_response"])

            else:
                utterance_age = now - (utterance_start or now)
                if utterance_age < 1.0:
                    # Too early/partial: stay in listening without error.
                    _reset_cmd_buffer()
                    utterance_start = None
                    set_mode("LISTENING", now)
                    return

                app.state.metrics["asr_errors"] += 1
                app.state.session["last_intent"] = "unknown"
                app.state.session["last_final"] = text
                app.state.session["last_response"] = intent_responses.get(
                    "unknown",
                    "Sorry, I can't help with that.",
                )
                app.state.session["debug"] = "No matching intent"
                app.state.session["intent_seq"] += 1
                _emit_state()

                if app.state.session["last_response"]:
                    _enqueue_tts(app.state.session["last_response"])

            cooldown_until = now + SETTINGS.cooldown_sec
            app.state.cooldown_until = cooldown_until
            if always_listening:
                set_mode("LISTENING", now)
                _reset_cmd_buffer()
                wake_active = True
            else:
                set_mode("EXECUTING", now)
                executing_until = now + max(0.05, min(0.5, executing_hold_sec))
                reset_listening(to_idle=False)

        # Save Deepgram usage: only stream ASR audio when we actually need transcription.
        needs_asr_stream = bool(
            always_listening
            or wake_active
            or in_cooldown
            or (not detector.available)
        )

        if needs_asr_stream:
            if _ensure_deepgram(now):
                deepgram.send_audio(data_norm)
                events = deepgram.drain_events()
                asr_source = "deepgram"
            else:
                if vosk_ready and vosk_stream is not None:
                    events = vosk_stream.process_audio(data_norm)
                    asr_source = "vosk"
                    app.state.voice_ok = True
                else:
                    events = []
                    asr_source = "none"
        else:
            if deepgram is not None and deepgram.connected:
                try:
                    deepgram.close()
                except Exception:
                    pass
                deepgram = None
            events = []
            asr_source = "none"
        app.state.session["asr_source"] = asr_source

        for event in events:
            if event.get("type") == "Error":
                app.state.metrics["asr_errors"] += 1
                src = event.get("source") or asr_source or "asr"
                app.state.session["debug"] = f"{src} error: {event.get('error', 'unknown')}"
                _emit_state()
                if src == "deepgram":
                    try:
                        if deepgram is not None:
                            deepgram.close()
                    except Exception:
                        pass
                    deepgram = None
                    dg_next_connect_ts = now + 1.5
                    app.state.voice_ok = False
                continue

            if event.get("type") != "Results":
                continue

            channel = event.get("channel") or {}
            alternatives = channel.get("alternatives") or []
            transcript = ""
            if alternatives:
                transcript = (alternatives[0].get("transcript") or "").strip()
            if not transcript:
                continue

            normalized = norm_join(transcript)
            is_final = bool(event.get("is_final"))
            _record_transcript(transcript, is_final, now)
            in_cooldown_now = time.time() < cooldown_until

            if in_cooldown_now and _stop_intent(transcript):
                tts_stop_event.set()
                cooldown_until = 0.0
                app.state.cooldown_until = cooldown_until
                app.state.session["transcript_partial"] = ""
                app.state.session["debug"] = ""
                set_mode("SLEEPING", time.time())
                continue

            if not wake_active and not always_listening:
                if not detector.available:
                    wake_phrase = norm_join(SETTINGS.wake_phrase or "")
                    wake_word = norm_join(SETTINGS.wake_word or "")
                    wake_match = (
                        (wake_phrase and wake_phrase in normalized)
                        or (wake_word and wake_word in normalized)
                    )
                    if wake_match:
                        if now - fallback_window_start > SETTINGS.fallback_hit_window_sec:
                            fallback_window_start = now
                            fallback_hits = 0
                        fallback_hits += 1
                        if fallback_hits >= SETTINGS.fallback_required_hits:
                            _trigger_wake(now)
                    else:
                        fallback_hits = 0
                continue

            if is_final:
                dg_cmd_finals.append(transcript)
                cmd_asr_source = (event.get("source") or asr_source or "deepgram")
                dg_partial_text = ""
            else:
                cmd_asr_source = (event.get("source") or asr_source or "deepgram")
                dg_partial_text = transcript

        if detector.available and not wake_active and not in_cooldown and not always_listening:
            oww_score = detector.score(audio_f32n)
            app.state.session["oww_score"] = oww_score
            app.state.session["oww_threshold"] = float(getattr(detector, "threshold", 0.0) or 0.0)

            if oww_score >= float(getattr(detector, "threshold", 0.0) or 0.0):
                _trigger_wake(now)
                return

        if wake_active and not always_listening and now > listen_until:
            reset_listening(to_idle=True)
            return

        if in_cooldown:
            return

        if not wake_active and not always_listening:
            return

        partial = (dg_partial_text or "").strip()

        # Gate/speech detection (used only for utterance timing/VAD):
        # - Use RMS from the *raw* stream so noise reduction doesn't suppress speech.
        # - After wake, lower threshold so quiet mics can still start/finalize utterances.
        speech_threshold = SETTINGS.noise_gate_rms
        if wake_active or always_listening:
            speech_threshold = max(0.0010, SETTINGS.noise_gate_rms * 0.20)
        speech = (rms_raw >= speech_threshold) or bool(partial)

        if partial and (now - last_partial_ts) > 0.35:
            app.state.session["transcript_partial"] = partial
            _emit_state()
            last_partial_ts = now
            payload = {
                "source": "voice",
                "text": partial,
                "is_final": False,
                "timestamp": now,
            }
            _emit(payload)

        if speech:
            last_speech_ts = now
            if utterance_start is None:
                utterance_start = now

        if utterance_start is not None:
            silence_elapsed = (now - last_speech_ts) >= SETTINGS.vad_silence_sec
            too_long = (now - utterance_start) >= SETTINGS.max_utterance_sec
            if silence_elapsed or too_long:
                finalize_command()

    try:
        if audio_source == "browser":
            logger.info("Voice capture started (browser stream)")
            buffer = bytearray()
            target = SETTINGS.block_size * 2

            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if chunk is None:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    continue

                buffer.extend(chunk)
                while len(buffer) >= target:
                    frame = bytes(buffer[:target])
                    del buffer[:target]
                    try:
                        handle_frame(frame)
                    except Exception as exc:
                        app.state.metrics["asr_errors"] += 1
                        app.state.session["debug"] = f"Frame processing error: {exc}"
                        _emit_state()
                        logger.exception("Frame processing error (browser): %s", exc)
            return

        channels = resolve_input_channels(SETTINGS, device)
        if channels <= 0:
            logger.error("Selected AUDIO_DEVICE has no input channels: %r", device)
            return

        with sd.RawInputStream(
            samplerate=input_sample_rate,
            blocksize=SETTINGS.block_size,
            dtype="int16",
            channels=channels,
            device=device,
        ) as stream:
            logger.info("Voice capture started (device)")
            while not stop_event.is_set():
                data, _ = stream.read(SETTINGS.block_size)
                if channels and channels > 1:
                    mono, best_ch, rms_by_ch = _select_loudest_channel_int16(data, channels)
                    app.state.session["mic_channel"] = best_ch
                    if rms_by_ch:
                        app.state.session["mic_rms_channels"] = [round(float(r), 6) for r in rms_by_ch]
                    try:
                        handle_frame(mono)
                    except Exception as exc:
                        app.state.metrics["asr_errors"] += 1
                        app.state.session["debug"] = f"Frame processing error: {exc}"
                        _emit_state()
                        logger.exception("Frame processing error (device mono): %s", exc)
                else:
                    try:
                        handle_frame(data)
                    except Exception as exc:
                        app.state.metrics["asr_errors"] += 1
                        app.state.session["debug"] = f"Frame processing error: {exc}"
                        _emit_state()
                        logger.exception("Frame processing error (device): %s", exc)
    except Exception as exc:
        app.state.voice_ok = False
        app.state.session["debug"] = f"Voice worker crashed: {exc}"
        _emit_state()
        logger.exception("Voice worker crashed: %s", exc)
    finally:
        app.state.voice_ok = False
        if deepgram is not None:
            deepgram.close()


async def broadcaster():
    while True:
        payload = await asyncio.to_thread(event_queue.get)
        if payload is None:
            break
        if not clients:
            continue

        msg = json.dumps(payload)
        results = await asyncio.gather(
            *[ws.send_text(msg) for ws in list(clients)],
            return_exceptions=True,
        )
        for ws, res in zip(list(clients), results):
            if isinstance(res, Exception):
                clients.discard(ws)


@app.get("/health")
async def health():
    return _json_sanitize({
        "ok": True,
        "voice_ok": app.state.voice_ok,
        "mode": app.state.mode,
        "cooldown_until": app.state.cooldown_until,
        "metrics": dict(app.state.metrics),
    })


@app.get("/status")
async def status():
    return _json_sanitize({
        "ok": True,
        "voice_ok": app.state.voice_ok,
        "session": dict(app.state.session),
        "metrics": dict(app.state.metrics),
        "voice_preset": dict(VOICE_PRESET),
        "clients": len(clients),
    })


@app.post("/reload_intents")
async def reload_intents():
    global INTENT_BUNDLES, VOICE_PRESET
    INTENT_BUNDLES = _load_intent_bundles()
    VOICE_PRESET = dict(INTENT_BUNDLES.get("deepgram", {}).get("voice_preset", VOICE_PRESET))
    logger.info(
        "Intent configs reloaded: deepgram=%s vosk=%s",
        SETTINGS.intent_config_deepgram_path,
        SETTINGS.intent_config_vosk_path,
    )
    return _json_sanitize(
        {
            "ok": True,
            "intent_config_deepgram_path": SETTINGS.intent_config_deepgram_path,
            "intent_config_vosk_path": SETTINGS.intent_config_vosk_path,
            "intent_count_deepgram": len((INTENT_BUNDLES.get("deepgram") or {}).get("phrases", {}) or {}),
            "intent_count_vosk": len((INTENT_BUNDLES.get("vosk") or {}).get("phrases", {}) or {}),
        }
    )


@app.post("/noise_profile/reset")
async def reset_noise_profile():
    reducer = getattr(app.state, "noise_reducer", None)
    if reducer is None:
        return {"ok": False, "reason": "noise_reducer not initialized yet"}
    if not getattr(reducer, "enabled", False):
        return {"ok": False, "reason": "noise reduction disabled"}

    reducer.reset_profile()
    app.state.session["noise_profile_ready"] = False
    _emit_state()
    logger.info("Noise profile reset requested via API")
    return {"ok": True, "noise_profile_ready": False}


@app.get("/models")
async def models_check():
    """Check which model files and Python packages are present on this machine."""
    import shutil
    from pathlib import Path as _Path

    def _check_path(path: str, is_dir: bool = False) -> dict:
        if not path:
            return {"exists": False, "path": "", "reason": "not configured in .env"}
        p = _Path(path)
        exists = p.is_dir() if is_dir else p.is_file()
        return {
            "exists": exists,
            "path": str(p),
            "reason": "" if exists else ("directory not found" if is_dir else "file not found"),
        }

    def _check_pkg(name: str) -> bool:
        try:
            __import__(name.replace("-", "_"))
            return True
        except ImportError:
            return False

    piper_exe_path = shutil.which(SETTINGS.piper_path) or SETTINGS.piper_path
    piper_exe_exists = shutil.which(SETTINGS.piper_path) is not None or _Path(SETTINGS.piper_path).is_file()
    oww_requested = (SETTINGS.wakeword_model_path or "").strip()
    oww_path, oww_name = resolve_openwakeword_model_path(oww_requested)
    if not oww_path and oww_requested:
        # Configured but couldn't resolve to an existing file.
        # Surface the requested value so the UI doesn't misleadingly show "not configured".
        oww_check = {"exists": False, "path": oww_requested, "reason": "file not found"}
    else:
        oww_check = _check_path(oww_path, is_dir=False)
    deepgram_key_ok = bool((SETTINGS.deepgram_api_key or "").strip())

    return _json_sanitize({
        "models": {
            "deepgram_api": {
                "exists": deepgram_key_ok,
                "path": "DEEPGRAM_API_KEY",
                "label": "Deepgram API Key",
                "reason": "" if deepgram_key_ok else "DEEPGRAM_API_KEY missing",
            },
            "vosk_model": {
                **_check_path(SETTINGS.vosk_model_path, is_dir=True),
                "label": "Vosk Model Directory",
            },
            "piper_model": {
                **_check_path(SETTINGS.piper_model_path, is_dir=False),
                "label": "Piper TTS Voice (.onnx)",
            },
            "piper_exe": {
                "exists": piper_exe_exists,
                "path": piper_exe_path,
                "label": "Piper Executable",
                "reason": "" if piper_exe_exists else "not found in PATH or file missing",
            },
            "openwakeword_model": {
                **oww_check,
                "label": f"OpenWakeWord Model (.tflite){f' ({oww_name})' if oww_name else ''}",
                "name": oww_name,
            },
        },
        "packages": {
            "websocket_client":      {"installed": _check_pkg("websocket"),           "label": "websocket-client"},
            "vosk":                  {"installed": _check_pkg("vosk"),                "label": "vosk"},
            "noisereduce":           {"installed": _check_pkg("noisereduce"),         "label": "noisereduce"},
            "openwakeword":          {"installed": _check_pkg("openwakeword"),        "label": "openwakeword"},
            "sounddevice":           {"installed": _check_pkg("sounddevice"),         "label": "sounddevice"},
            "python_dotenv":         {"installed": _check_pkg("dotenv"),              "label": "python-dotenv"},
            "sentence_transformers": {"installed": _check_pkg("sentence_transformers"), "label": "sentence-transformers"},
        },
        "settings": {
            "tts_enabled":       SETTINGS.tts_enabled,
            "duck_enabled":      SETTINGS.duck_enabled,
            "noise_reduction":   SETTINGS.noise_reduction_enabled,
            "noise_gate_auto":   SETTINGS.noise_gate_auto,
            "always_listening":  SETTINGS.always_listening,
            "wake_word":         SETTINGS.wake_word,
            "wake_phrase":       SETTINGS.wake_phrase,
            "audio_source":      SETTINGS.audio_source,
            "asr_backend":       SETTINGS.asr_backend,
            "deepgram_model":    SETTINGS.deepgram_model,
            "deepgram_language": SETTINGS.deepgram_language,
            "vosk_model_path":   SETTINGS.vosk_model_path,
            "intent_config_deepgram_path": SETTINGS.intent_config_deepgram_path,
            "intent_config_vosk_path": SETTINGS.intent_config_vosk_path,
            "transcript_debug_enabled": SETTINGS.transcript_debug_enabled,
            "transcript_debug_max": SETTINGS.transcript_debug_max,
            "use_embeddings":    os.getenv("USE_EMBEDDINGS", "false").lower() == "true",
        },
        "runtime": {
            "noise_profile_ready": app.state.session.get("noise_profile_ready", False),
            "noise_gate_rms":      round(SETTINGS.noise_gate_rms, 5),
            "voice_ok":            app.state.voice_ok,
        },
    })


@app.get("/mic_test")
async def mic_test():
    device_info = None
    selected_device = None
    try:
        import sounddevice as sd

        selected_device = resolve_device(SETTINGS)
        default_in = sd.default.device[0] if hasattr(sd.default, "device") else None
        if default_in is not None:
            device_info = sd.query_devices(default_in)
    except Exception:
        device_info = None

    return {
        "ok": True,
        "voice_rms": app.state.session.get("voice_rms", 0.0),
        "voice_rms_raw": app.state.session.get("voice_rms_raw", 0.0),
        "voice_rms_norm": app.state.session.get("voice_rms_norm", None),
        "audio_device_env": SETTINGS.device,
        "selected_device": selected_device,
        "default_input_device": device_info,
    }


@app.get("/mic_test_stream")
def mic_test_stream(duration_sec: float = 1.5):
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        return {"ok": False, "error": f"Missing deps: {exc}"}

    device = resolve_device(SETTINGS)
    sample_rate = resolve_sample_rate(SETTINGS, device)
    channels = resolve_input_channels(SETTINGS, device)
    if channels <= 0:
        return {
            "ok": False,
            "error": "Selected audio device has no input channels. Set AUDIO_DEVICE to an input device or leave it empty.",
        }
    duration_sec = max(0.2, min(5.0, float(duration_sec)))

    peak_rms = 0.0
    peak_rms_channels = None
    selected_channel = None
    samples = 0

    try:
        device_info = None
        try:
            device_info = sd.query_devices(device) if device is not None else None
        except Exception:
            device_info = None

        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=SETTINGS.block_size,
            dtype="int16",
            channels=channels,
            device=device,
        ) as stream:
            start = time.time()
            while (time.time() - start) < duration_sec:
                data, _ = stream.read(SETTINGS.block_size)
                x = np.frombuffer(data, dtype=np.int16)
                if channels > 1:
                    frames = int(len(x) // channels)
                    if frames > 0:
                        x2 = x[: frames * channels].reshape(frames, channels).astype(np.float32) / 32768.0
                        rms_by = np.sqrt(np.mean(x2 * x2, axis=0)).astype(float).tolist()
                        if peak_rms_channels is None:
                            peak_rms_channels = [0.0 for _ in range(channels)]
                        for i, r in enumerate(rms_by):
                            peak_rms_channels[i] = max(float(peak_rms_channels[i]), float(r))
                        best = int(np.argmax(np.asarray(rms_by, dtype=np.float32)))
                        selected_channel = best
                        peak_rms = max(peak_rms, float(rms_by[best]))
                    else:
                        peak_rms = max(peak_rms, 0.0)
                else:
                    audio_f32 = x.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(audio_f32 * audio_f32)))
                    peak_rms = max(peak_rms, rms)
                samples += 1
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "duration_sec": duration_sec,
        "audio_device_env": SETTINGS.device,
        "selected_device": device,
        "selected_device_info": device_info,
        "sample_rate": sample_rate,
        "channels": channels,
        "selected_channel": selected_channel,
        "peak_rms_channels": peak_rms_channels,
        "peak_rms": peak_rms,
        "samples": samples,
    }


@app.get("/devices")
async def list_devices():
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        default_in = sd.default.device[0] if hasattr(sd.default, "device") else None
        return {
            "ok": True,
            "default_input_index": default_in,
            "devices": devices,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/mic_live")
async def mic_live():
    """
    Returns the live RMS values from the running voice worker (no device open).
    Use this when AUDIO_SOURCE=device because /mic_test_stream opens a new stream and
    may conflict with the active capture.
    """
    return _json_sanitize(
        {
            "ok": True,
            "audio_device_env": SETTINGS.device,
            "audio_source": SETTINGS.audio_source,
            "voice_rms_raw": app.state.session.get("voice_rms_raw", 0.0),
            "voice_rms": app.state.session.get("voice_rms", 0.0),
            "voice_rms_norm": app.state.session.get("voice_rms_norm", None),
            "mic_channel": app.state.session.get("mic_channel", None),
            "mic_rms_channels": app.state.session.get("mic_rms_channels", None),
            "mode": app.state.mode,
        }
    )


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    app.state.loop = loop

    try:
        logger.info("Voice app file: %s", __file__)
        logger.info("Voice cwd: %s", os.getcwd())
        logger.info(
            "Voice config: AUDIO_SOURCE=%s AUDIO_DEVICE=%r ALWAYS_LISTEN=%s",
            SETTINGS.audio_source,
            SETTINGS.device,
            SETTINGS.always_listening,
        )
    except Exception:
        pass

    global INTENT_BUNDLES, VOICE_PRESET
    INTENT_BUNDLES = _load_intent_bundles()
    VOICE_PRESET = dict(INTENT_BUNDLES.get("deepgram", {}).get("voice_preset", VOICE_PRESET))

    app.state.broadcaster_task = asyncio.create_task(broadcaster())
    app.state.worker_task = asyncio.create_task(asyncio.to_thread(voice_worker, loop))
    app.state.tts_task = asyncio.create_task(asyncio.to_thread(tts_worker))


@app.on_event("shutdown")
async def on_shutdown():
    stop_event.set()
    event_queue.put(None)
    tts_queue.put(None)

    task = getattr(app.state, "broadcaster_task", None)
    if task:
        task.cancel()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Frontend doesn't need to send anything; this keeps the handler alive
                # without requiring incoming messages.
                continue
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


@app.websocket("/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            data = msg.get("bytes")
            if not data:
                continue
            try:
                audio_queue.put_nowait(data)
            except queue.Full:
                drops = int(app.state.metrics.get("audio_queue_drops", 0)) + 1
                app.state.metrics["audio_queue_drops"] = drops
                if drops == 1 or drops % 50 == 0:
                    logger.warning("Audio queue full; dropped %d websocket audio frames", drops)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
