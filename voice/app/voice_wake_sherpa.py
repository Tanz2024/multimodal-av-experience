import asyncio
import json
import logging
import math
import os
import queue
import subprocess
import tempfile
import time
import random
from collections import deque
from threading import Event, Lock, Thread
from urllib.parse import urlencode

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


try:
    from dotenv import load_dotenv as _load_dotenv
    # Always load project .env as source of truth for runtime audio config.
    _load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set before launch

from .voice_helpers import (
    Settings,
    normalize_tokens,
    norm_join,
    load_intent_config,
    classify_intent,
    match_intent_contains,
    match_intent_keywords,
    CommandExecutor,
    AudioDucker,
    play_wav,
    resolve_device,
    resolve_sample_rate,
    resolve_input_channels,
    resolve_openwakeword_model_path,
    OpenWakeWordDetector,
    MicrosoftKeywordDetector,
    NoiseReducer,
    VOICE_PRESET,
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8010"))

SETTINGS = Settings()

DEMO_GRAMMAR_LOCK = os.getenv("DEMO_GRAMMAR_LOCK", "true").lower() == "true"
DEMO_ALLOWED_INTENTS = {
    i.strip()
    for i in os.getenv(
        "DEMO_ALLOWED_INTENTS",
        "greet,help,bye,back,cancel,next,previous,mute,all_sound_off,welcome,kitchen,invisible,cinema,lounge,iconic,open_1,open_2,open_3,open_4,open_5,open_6,open_7,option_1,option_2,option_3,option_4,option_5,option_6,option_7,volume_up,volume_down",
    ).split(",")
    if i.strip()
}
INTENT_CONFIRM_THRESHOLD = float(os.getenv("INTENT_CONFIRM_THRESHOLD", "0.82"))
INTENT_CONFIRM_WINDOW_SEC = float(os.getenv("INTENT_CONFIRM_WINDOW_SEC", "5.0"))
CONFIRM_FOR_INTENTS = {
    i.strip()
    for i in os.getenv(
        "INTENT_CONFIRM_FOR",
        "volume_up,volume_down,mute,all_sound_off,open_1,open_2,open_3,open_4,open_5,open_6,open_7,welcome,kitchen,invisible,cinema,lounge,iconic",
    ).split(",")
    if i.strip()
}
COMMAND_RATE_LIMIT_SEC = float(os.getenv("COMMAND_RATE_LIMIT_SEC", "0.35"))
COMMAND_DEBOUNCE_SEC = float(os.getenv("COMMAND_DEBOUNCE_SEC", "1.0"))
COMMAND_TEXT_DEBOUNCE_SEC = float(os.getenv("COMMAND_TEXT_DEBOUNCE_SEC", "2.0"))
COMMAND_QUEUE_MAX = int(os.getenv("COMMAND_QUEUE_MAX", "100"))
_audio_queue_max = max(20, int(os.getenv("AUDIO_QUEUE_MAX", "400")))
COMMAND_TIMEOUT_SEC = float(os.getenv("COMMAND_TIMEOUT_SEC", "0.8"))
COMMAND_MAX_RETRIES = int(os.getenv("COMMAND_MAX_RETRIES", "2"))
COMMAND_LOG_MAX = int(os.getenv("COMMAND_LOG_MAX", "200"))
ACK_MODE = (os.getenv("COMMAND_ACK_MODE", "mock") or "mock").strip().lower()
ACK_FAIL_RATE = max(0.0, min(1.0, float(os.getenv("COMMAND_ACK_FAIL_RATE", "0.05"))))
ACK_MIN_MS = max(0, int(os.getenv("COMMAND_ACK_MIN_MS", "60")))
ACK_MAX_MS = max(ACK_MIN_MS, int(os.getenv("COMMAND_ACK_MAX_MS", "180")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("voice-wake-sherpa")

clients = set()
event_queue = queue.Queue()
tts_queue = queue.Queue()
audio_queue = queue.Queue(maxsize=_audio_queue_max)
command_queue = queue.Queue(maxsize=COMMAND_QUEUE_MAX)
stop_event = Event()
tts_stop_event = Event()

app = FastAPI()
CORS_ALLOW_ORIGIN_REGEX = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.last_payload = None
app.state.last_sent_ts = None
app.state.voice_ok = False
app.state.mode = "SLEEPING"
app.state.cooldown_until = 0.0
app.state.tts_playing = False
app.state.last_tts_text = ""
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
    "pending_confirmation": None,
    "command_log": [],
    "last_action": None,
    "last_tts_backend": "",
    "last_tts_error": "",
    "last_tts_text": "",
    "tts_playing": False,
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
    "cmd_queued": 0,
    "cmd_ack_ok": 0,
    "cmd_ack_fail": 0,
    "cmd_retry": 0,
    "cmd_dropped": 0,
    "cmd_rate_limited": 0,
    "cmd_debounced": 0,
    "cmd_confirmations": 0,
    "cmd_confirm_accept": 0,
    "cmd_confirm_reject": 0,
    "cmd_blocked_by_grammar": 0,
    "deepgram_failovers": 0,
    "asr_fallback_to_vosk": 0,
    "cmd_latency_ms_last": 0.0,
    "commands_failed": 0,
}
app.state.last_command_ts = 0.0
app.state.last_intent = ""
app.state.last_intent_ts = 0.0
app.state.last_command_text = ""
app.state.last_command_text_ts = 0.0
app.state.pending_confirmation = None


def _emit(payload):
    event_queue.put(_json_sanitize(payload))


def _status_payload():
    return _json_sanitize(
        {
            "ok": True,
            "voice_ok": app.state.voice_ok,
            "session": dict(app.state.session),
            "metrics": dict(app.state.metrics),
            "command_queue_depth": command_queue.qsize(),
            "voice_preset": dict(VOICE_PRESET),
            "clients": len(clients),
            "timestamp": round(time.time(), 3),
        }
    )


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
        {
            "type": "state_update",
            "source": "voice",
            "state": dict(app.state.session),
            "status": _status_payload(),
        }
    )


def _enqueue_tts(text: str):
    if not SETTINGS.tts_enabled:
        return
    text = str(text or "").strip()
    if not text:
        return
    # When rapid wake/intent events produce the same phrase, avoid replaying
    # the exact text if it is already in-flight.
    if app.state.tts_playing and getattr(app.state, "last_tts_text", "") == text:
        return
    app.state.last_tts_text = text
    app.state.session["last_tts_text"] = text
    # Keep TTS aligned with the latest command: interrupt current playback
    # and drop older queued responses to avoid stale/out-of-order speech.
    tts_stop_event.set()
    while True:
        try:
            _ = tts_queue.get_nowait()
            tts_queue.task_done()
        except queue.Empty:
            break
    tts_queue.put(text)


def _append_command_log(entry: dict):
    log = app.state.session.get("command_log")
    if not isinstance(log, list):
        log = []
    log.append(entry)
    if len(log) > COMMAND_LOG_MAX:
        log = log[-COMMAND_LOG_MAX:]
    app.state.session["command_log"] = log


def _log_command_event(event: str, **kwargs):
    payload = {"ts": round(time.time(), 3), "event": event, **kwargs}
    _append_command_log(_json_sanitize(payload))
    app.state.session["last_action"] = payload
    logger.info("command_event=%s data=%s", event, {k: v for k, v in kwargs.items()})
    _emit({"type": "command_event", "source": "voice", "payload": payload})


def _confirm_yes(text: str) -> bool:
    toks = set(normalize_tokens(text))
    return bool(toks & {"yes", "yeah", "yep", "correct", "right", "confirm", "do", "ok", "okay"})


def _confirm_no(text: str) -> bool:
    toks = set(normalize_tokens(text))
    return bool(toks & {"no", "nope", "wrong", "cancel", "stop", "dont", "don't"})


def _emergency_all_off(text: str) -> bool:
    toks = set(normalize_tokens(text))
    panic = {"emergency", "panic", "urgent", "immediately", "immediate"}
    off = {"off", "mute", "stop", "silence", "shutdown"}
    return bool(toks & panic and toks & off)


class CommandAdapter:
    def send(self, intent: str, payload: dict, timeout_sec: float) -> bool:
        raise NotImplementedError


class MockCommandAdapter(CommandAdapter):
    def send(self, intent: str, payload: dict, timeout_sec: float) -> bool:
        del payload, timeout_sec
        delay_ms = random.randint(ACK_MIN_MS, ACK_MAX_MS)
        time.sleep(delay_ms / 1000.0)
        return random.random() >= ACK_FAIL_RATE


def _adapter() -> CommandAdapter:
    # Placeholder for future hardware adapters; currently mock ACK simulation.
    if ACK_MODE == "mock":
        return MockCommandAdapter()
    return MockCommandAdapter()


def _stop_intent(text: str) -> bool:
    t = norm_join(text)
    return "stop" in t or "all off" in t or "sound off" in t or "mute" in t


def _strip_wake_prefix(text: str) -> str:
    normalized = norm_join(text)
    if not normalized:
        return ""

    wake_phrases = _wake_phrase_variants()

    # Remove wake words anywhere in the sentence (not only prefix),
    # while keeping token boundaries stable.
    wake_token_lists = [p.split() for p in wake_phrases if p]
    if not wake_token_lists:
        return normalized

    # Prefer longest phrase first (e.g. "hey tom" before "tom").
    wake_token_lists.sort(key=len, reverse=True)

    tokens = normalized.split()
    out_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        for wt in wake_token_lists:
            wlen = len(wt)
            if wlen > 0 and tokens[i:i + wlen] == wt:
                i += wlen
                matched = True
                break
        if not matched:
            out_tokens.append(tokens[i])
            i += 1

    return " ".join(out_tokens).strip()


def _wake_token_aliases(token: str) -> list[str]:
    t = norm_join(token)
    if not t:
        return []
    aliases = {t}
    # Tolerate common ASR variants for the same wake pronunciation.
    if t in {"tom", "thom"}:
        aliases.update({"tom", "thom", "dom", "tam"})
    return [a for a in aliases if a]


def _wake_phrase_variants() -> list[str]:
    variants: set[str] = set()
    base_phrases: list[str] = []
    if SETTINGS.wake_phrase:
        base_phrases.append(norm_join(SETTINGS.wake_phrase))
    if SETTINGS.wake_word:
        base_phrases.append(norm_join(SETTINGS.wake_word))

    for phrase in base_phrases:
        toks = [t for t in phrase.split() if t]
        if not toks:
            continue
        if len(toks) == 1:
            for a in _wake_token_aliases(toks[0]):
                variants.add(a)
            continue
        if len(toks) == 2:
            a0 = _wake_token_aliases(toks[0]) or [toks[0]]
            a1 = _wake_token_aliases(toks[1]) or [toks[1]]
            for x in a0:
                for y in a1:
                    variants.add(f"{x} {y}".strip())
            # Also include just the wake word part for lenient fallback.
            for y in a1:
                variants.add(y)
            continue
        variants.add(" ".join(toks))

    # Deterministic ordering: prefer longer phrases first.
    return sorted(variants, key=lambda p: len(p.split()), reverse=True)


def _wake_token_match(a: str, b: str) -> bool:
    a = norm_join(a)
    b = norm_join(b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Compare across known aliases first (e.g. tom/thom/dom).
    a_alias = set(_wake_token_aliases(a))
    b_alias = set(_wake_token_aliases(b))
    if a_alias & b_alias:
        return True
    # Allow a single small edit for ASR drift.
    if abs(len(a) - len(b)) > 1:
        return False
    i = 0
    j = 0
    edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) == len(b):
            i += 1
            j += 1
        elif len(a) > len(b):
            i += 1
        else:
            j += 1
    if i < len(a) or j < len(b):
        edits += 1
    return edits <= 1


def _wake_phrase_match(text: str, wake_phrases: list[str]) -> bool:
    normalized = norm_join(text)
    if not normalized:
        return False
    if any(wp and wp in normalized for wp in wake_phrases):
        return True

    tokens = normalized.split()
    for phrase in wake_phrases:
        ptoks = [t for t in norm_join(phrase).split() if t]
        if not ptoks or len(ptoks) > len(tokens):
            continue
        w = len(ptoks)
        for start in range(0, len(tokens) - w + 1):
            if all(_wake_token_match(tokens[start + i], ptoks[i]) for i in range(w)):
                return True
    return False


def _soft_wake_greeting_match(text: str) -> bool:
    """
    Fallback for low-quality headset transcripts where wake phrase may be
    misheard as a short greeting like "huh"/"hey"/"hi".
    """
    allowed = {
        tok
        for tok in normalize_tokens(os.getenv("WAKE_GREETING_FALLBACK", "hey hi hello huh"))
        if tok
    }
    toks = normalize_tokens(text or "")
    if not toks or not allowed:
        return False
    # Keep this conservative: only very short utterances can soft-wake.
    if len(toks) > 2:
        return False
    return any(t in allowed for t in toks)


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
_WAKE_RESPONSES = ["How may I help you?", "What's next?", "Ready, go ahead.", "Listening.", "What can I do?"]
_wake_response_idx = 0
executor = CommandExecutor(app)
command_adapter = _adapter()


def _enqueue_command_action(intent: str, text: str, confidence: float, source: str) -> bool:
    now = time.time()
    # Emergency commands bypass debounce/rate-limit.
    emergency = intent in {"all_sound_off", "mute"}
    norm_text = norm_join(text or "")

    if (not emergency) and (now - app.state.last_command_ts) < COMMAND_RATE_LIMIT_SEC:
        app.state.metrics["cmd_rate_limited"] += 1
        app.state.session["debug"] = "Command rate-limited"
        _log_command_event("rate_limited", intent=intent, text=text)
        _emit_state()
        return False

    if (not emergency) and intent == app.state.last_intent and (now - app.state.last_intent_ts) < COMMAND_DEBOUNCE_SEC:
        app.state.metrics["cmd_debounced"] += 1
        app.state.session["debug"] = "Command ignored (debounced)"
        _log_command_event("debounced", intent=intent, text=text)
        _emit_state()
        return False
    if (
        (not emergency)
        and norm_text
        and norm_text == app.state.last_command_text
        and (now - app.state.last_command_text_ts) < COMMAND_TEXT_DEBOUNCE_SEC
    ):
        app.state.metrics["cmd_debounced"] += 1
        app.state.session["debug"] = "Command ignored (repeat transcript)"
        _log_command_event("debounced_text", intent=intent, text=text)
        _emit_state()
        return False

    item = {
        "intent": intent,
        "text": text,
        "confidence": float(confidence or 0.0),
        "source": source,
        "queued_at": now,
        "attempt": 0,
    }
    try:
        if emergency:
            # Flush queued non-emergency commands and prioritize all-off behavior.
            while not command_queue.empty():
                try:
                    command_queue.get_nowait()
                    command_queue.task_done()
                except Exception:
                    break
        command_queue.put_nowait(item)
    except queue.Full:
        app.state.metrics["cmd_dropped"] += 1
        app.state.session["debug"] = "Command queue full"
        _log_command_event("queue_full", intent=intent, text=text)
        _emit_state()
        return False

    app.state.metrics["cmd_queued"] += 1
    app.state.last_command_ts = now
    app.state.last_intent = intent
    app.state.last_intent_ts = now
    app.state.last_command_text = norm_text
    app.state.last_command_text_ts = now
    _log_command_event("queued", intent=intent, confidence=round(float(confidence or 0.0), 3), source=source)
    _emit_state()
    return True


def command_worker():
    while not stop_event.is_set():
        item = command_queue.get()
        if item is None:
            break
        intent = str(item.get("intent") or "")
        text = str(item.get("text") or "")
        confidence = float(item.get("confidence") or 0.0)
        queued_at = float(item.get("queued_at") or time.time())

        success = False
        for attempt in range(1, COMMAND_MAX_RETRIES + 2):
            item["attempt"] = attempt
            ack = command_adapter.send(intent, item, COMMAND_TIMEOUT_SEC)
            if ack:
                success = True
                app.state.metrics["cmd_ack_ok"] += 1
                break
            app.state.metrics["cmd_ack_fail"] += 1
            if attempt <= COMMAND_MAX_RETRIES:
                app.state.metrics["cmd_retry"] += 1
                time.sleep(min(0.2 * attempt, 0.8))

        latency_ms = max(0.0, (time.time() - queued_at) * 1000.0)
        app.state.metrics["cmd_latency_ms_last"] = round(latency_ms, 2)
        if success:
            executor.execute(intent)
            _log_command_event(
                "executed",
                intent=intent,
                confidence=round(confidence, 3),
                latency_ms=round(latency_ms, 2),
                text=text,
            )
        else:
            app.state.metrics["commands_failed"] = int(app.state.metrics.get("commands_failed", 0)) + 1
            _log_command_event(
                "failed",
                intent=intent,
                confidence=round(confidence, 3),
                latency_ms=round(latency_ms, 2),
                text=text,
            )
        command_queue.task_done()


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


COMMAND_GATE_BASE_TOKENS = {
    "welcome",
    "kitchen",
    "invisible",
    "cinema",
    "lounge",
    "iconic",
    "open",
    "option",
    "next",
    "previous",
    "prev",
    "back",
    "cancel",
    "mute",
    "silence",
    "volume",
    "louder",
    "quieter",
    "raise",
    "lower",
    "increase",
    "decrease",
    "up",
    "down",
    "sound",
    "off",
    "help",
    "command",
    "commands",
    "show",
    "start",
    "launch",
    "select",
    "choose",
    "pick",
    "experience",
    "experiences",
    "menu",
    "home",
    "screen",
    "panel",
    "movie",
    "theater",
    "theatre",
    "film",
    "cook",
    "cooking",
    "party",
    "entertainment",
    "design",
    "hello",
    "hi",
    "hey",
    "tom",
}
COMMAND_GATE_IGNORE_TOKENS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "and",
    "or",
    "with",
    "this",
    "that",
    "please",
}
COMMAND_HINT_TOKENS = set(COMMAND_GATE_BASE_TOKENS)

EXPERIENCE_OPTION_LABELS = {
    "welcome": ["Soft Welcome", "Luxury Ambient"],
    "kitchen": ["Silent", "Background Ambience", "Entertaining Mode"],
    "invisible": ["Silent", "Background Ambience", "Entertaining Mode"],
    "cinema": ["Invisible Cinema", "Live Performance", "Luxury Apartment Night"],
    "lounge": ["Background Lounge", "Cocktail Evening", "Silent Design Focus"],
    "iconic": ["Background Lounge", "Cocktail Evening", "Silent Design Focus"],
    "off": ["Confirm Mute"],
}


def _contextual_intent_from_text(text: str, current_experience: str | None) -> str | None:
    """
    Map short follow-up utterances (e.g. "soft", "cocktail", "live") to option
    intents based on the currently selected experience.
    """
    exp = str(current_experience or "").strip().lower()
    if not exp:
        return None

    tokens = set(normalize_tokens(text))
    if not tokens:
        return None

    if exp == "welcome":
        if tokens & {"soft", "gentle", "warm"}:
            return "opt_soft_welcome"
        if tokens & {"luxury", "ambient", "premium"}:
            return "opt_luxury_ambient"

    if exp in {"kitchen", "invisible"}:
        if tokens & {"silent", "silence", "quiet"}:
            return "opt_silent"
        if tokens & {"ambience", "ambiance", "background"}:
            return "opt_background_ambience"
        if tokens & {"entertaining", "entertain", "party"}:
            return "opt_entertaining_mode"

    if exp == "cinema":
        if tokens & {"invisible"}:
            return "opt_invisible_cinema"
        if tokens & {"live", "performance"}:
            return "opt_live_performance"
        if tokens & {"luxury", "apartment", "night"}:
            return "opt_luxury_apartment_night"

    if exp in {"lounge", "iconic"}:
        if tokens & {"background", "lounge"}:
            return "opt_background_lounge"
        if tokens & {"cocktail", "evening"}:
            return "opt_cocktail_evening"
        if tokens & {"design", "focus", "silent"}:
            return "opt_design_focus"

    return None


def _refresh_command_hint_tokens():
    dynamic_tokens = set(COMMAND_GATE_BASE_TOKENS)
    for bundle in INTENT_BUNDLES.values():
        phrases = (bundle or {}).get("phrases", {})
        if not isinstance(phrases, dict):
            continue
        for examples in phrases.values():
            if not isinstance(examples, list):
                continue
            for ex in examples:
                for tok in normalize_tokens(str(ex)):
                    if len(tok) >= 3 and tok not in COMMAND_GATE_IGNORE_TOKENS:
                        dynamic_tokens.add(tok)
    COMMAND_HINT_TOKENS.clear()
    COMMAND_HINT_TOKENS.update(dynamic_tokens)


_refresh_command_hint_tokens()


def _looks_like_command_text(text: str) -> bool:
    """
    Heuristic gate for ALWAYS_LISTEN mode so casual speech doesn't trigger
    repeated "unknown" responses.
    """
    toks = set(normalize_tokens(text))
    if not toks:
        return False
    if toks & COMMAND_HINT_TOKENS:
        return True
    digits = {"1", "2", "3", "4", "5", "6", "7", "one", "two", "three", "four", "five", "six", "seven"}
    return bool(toks & digits and ("open" in toks or "option" in toks))


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
            transient_errors = 0
            while self.connected:
                try:
                    msg = ws.recv()
                except Exception as exc:
                    err_text = str(exc)
                    err_low = err_text.lower()

                    # Ignore temporary recv timeouts.
                    if "timed out" in err_low or "timeout" in err_low:
                        continue

                    # Retry briefly on likely transient socket issues.
                    if any(k in err_low for k in ("tempor", "again", "would block", "econnreset", "connection reset")):
                        transient_errors += 1
                        if transient_errors <= 3:
                            time.sleep(0.2)
                            continue

                    self.events.put({"type": "Error", "error": err_text})
                    break
                if msg is None:
                    break
                transient_errors = 0
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

    def _synthesize_azure(text: str, out_path: str) -> tuple[bool, str]:
        key = (SETTINGS.azure_tts_key_1 or SETTINGS.azure_tts_key_2 or "").strip()
        region = (SETTINGS.speech_region or "").strip()
        if not key:
            return False, "Azure TTS key is not set"
        if not region:
            return False, "SPEECH_REGION is not set"

        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception as exc:
            return False, f"azure-cognitiveservices-speech import failed: {exc}"

        try:
            speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
            if SETTINGS.azure_tts_voice:
                speech_config.speech_synthesis_voice_name = SETTINGS.azure_tts_voice
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
            )
            audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            result = synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return True, ""
            if result.reason == speechsdk.ResultReason.Canceled:
                details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
                return False, f"Azure synthesis canceled: reason={details.reason} error={details.error_details!r}"
            return False, f"Azure synthesis failed with reason={result.reason}"
        except Exception as exc:
            return False, f"Azure synthesis exception: {exc}"

    def _synthesize_piper(text: str, out_path: str) -> tuple[bool, str]:
        preset_model_path = VOICE_PRESET.get("model_path", "")
        model_path = preset_model_path or SETTINGS.piper_model_path
        if not model_path:
            return False, "PIPER_MODEL_PATH not set"

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

        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"")[:4000].decode("utf-8", errors="replace").strip()
            out = (proc.stdout or b"")[:2000].decode("utf-8", errors="replace").strip()
            return False, f"Piper exited with code {proc.returncode}. stderr={err!r} stdout={out!r}"
        return True, ""

    while not stop_event.is_set():
        text = tts_queue.get()
        if text is None:
            break
        if not SETTINGS.tts_enabled:
            continue

        app.state.tts_playing = False
        app.state.session["tts_playing"] = False
        app.state.session["last_tts_backend"] = ""
        app.state.session["last_tts_error"] = ""

        out_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name

            ducker.duck()
            tts_stop_event.clear()
            app.state.metrics["last_tts_ts"] = time.time()
            synth_started_ts = time.time()

            ok = False
            err = ""
            backend = (SETTINGS.tts_backend or "piper").strip().lower()

            # `azure`/`auto`: Azure primary, Piper fallback.
            # `piper`: Piper only.
            if backend in {"azure", "auto"}:
                app.state.session["last_tts_backend"] = "azure"
                ok, azure_err = _synthesize_azure(text, out_path)
                if not ok:
                    logger.warning("Azure TTS failed, falling back to Piper: %s", azure_err)
                    app.state.session["last_tts_backend"] = "piper"
                    ok, piper_err = _synthesize_piper(text, out_path)
                    if not ok:
                        err = f"Azure failed: {azure_err} | Piper failed: {piper_err}"
                        app.state.session["last_tts_error"] = err
            else:
                app.state.session["last_tts_backend"] = "piper"
                ok, err = _synthesize_piper(text, out_path)
                if not ok:
                    app.state.session["last_tts_error"] = err

            if not ok:
                app.state.metrics["tts_errors"] += 1
                logger.warning("TTS synthesis failed: %s", err)
                continue
            logger.info(
                "TTS synthesis ok backend=%s synth_ms=%.1f text_len=%d",
                app.state.session.get("last_tts_backend", ""),
                (time.time() - synth_started_ts) * 1000.0,
                len(text or ""),
            )

            try:
                if (not os.path.isfile(out_path)) or (os.path.getsize(out_path) <= 0):
                    app.state.metrics["tts_errors"] += 1
                    app.state.session["last_tts_error"] = "output wav missing/empty"
                    logger.warning("TTS output wav missing/empty: %s", out_path)
                    continue
            except Exception as exc:
                app.state.metrics["tts_errors"] += 1
                app.state.session["last_tts_error"] = f"output wav check failed: {exc}"
                logger.warning("TTS output wav check failed: %s", exc)
                continue

            app.state.tts_playing = True
            app.state.session["tts_playing"] = True
            _emit_state()
            ok = play_wav(out_path, tts_stop_event)
            if not ok:
                app.state.metrics["tts_errors"] += 1
                app.state.session["last_tts_error"] = "playback failed"
                logger.warning("TTS playback failed")
            else:
                if not app.state.session.get("last_tts_error"):
                    app.state.session["last_tts_error"] = ""

        except Exception as exc:
            app.state.metrics["tts_errors"] += 1
            app.state.session["last_tts_error"] = f"tts worker exception: {exc}"
            logger.warning("TTS failed: %s", exc)

        finally:
            app.state.tts_playing = False
            app.state.session["tts_playing"] = False
            ducker.unduck()
            if out_path:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            # Brief post-TTS silence window so the mic doesn't pick up TTS echo
            if SETTINGS.tts_cooldown_sec > 0:
                app.state.cooldown_until = max(
                    float(getattr(app.state, "cooldown_until", 0.0) or 0.0),
                    time.time() + SETTINGS.tts_cooldown_sec,
                )
                time.sleep(SETTINGS.tts_cooldown_sec)


def voice_worker(loop):
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        logger.error("Missing deps for voice: %s", exc)
        return

    def _select_loudest_channel_int16(
        data: bytes,
        channels: int,
        channel_lock: int = -1,
    ) -> tuple[bytes, int, list[float]]:
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
            if 0 <= int(channel_lock) < ch:
                best = int(channel_lock)
            else:
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

    def _is_device_invalidated_error(exc: Exception) -> bool:
        txt = str(exc or "").lower()
        return (
            "audclnt_e_device_invalidated" in txt
            or "wasapi error -2004287484" in txt
            or ("device" in txt and "invalidated" in txt)
        )

    def _is_insufficient_memory_error(exc: Exception) -> bool:
        txt = str(exc or "").lower()
        return "insufficient memory" in txt or "paerrorcode -9992" in txt

    def _is_blocking_api_unsupported_error(exc: Exception) -> bool:
        txt = str(exc or "").lower()
        return "blocking api not supported yet" in txt or "wdm-ks error" in txt

    def _is_wdmks_device(device) -> bool:
        try:
            import sounddevice as _sd

            idx = device
            if idx is None:
                default_dev = _sd.default.device[0] if hasattr(_sd.default, "device") else None
                idx = default_dev
            if idx is None:
                return False
            info = _sd.query_devices(idx)
            hostapi_idx = int(info.get("hostapi") or 0)
            hostapis = _sd.query_hostapis()
            if 0 <= hostapi_idx < len(hostapis):
                host_name = str(hostapis[hostapi_idx].get("name", "")).lower()
                return "wdm-ks" in host_name
        except Exception:
            return False
        return False

    def _stream_open_candidates():
        cfg_device = resolve_device(SETTINGS)
        cfg_sr = resolve_sample_rate(SETTINGS, cfg_device)
        cfg_ch = resolve_input_channels(SETTINGS, cfg_device)

        raw = [
            # User-selected device first.
            (cfg_device, cfg_sr, cfg_ch, "configured"),
            (cfg_device, int(getattr(SETTINGS, "sample_rate", 16000) or 16000), 1, "configured-16k-mono"),
            # Fall back to default input device with conservative params.
            (None, int(getattr(SETTINGS, "sample_rate", 16000) or 16000), 1, "default-16k-mono"),
            (None, 48000, 1, "default-48k-mono"),
            (None, 44100, 1, "default-44k-mono"),
        ]
        out = []
        seen = set()
        for dev, sr, ch, label in raw:
            try:
                sr = int(sr)
                ch = int(ch)
            except Exception:
                continue
            if sr <= 0 or ch <= 0:
                continue
            key = (str(dev), sr, ch)
            if key in seen:
                continue
            seen.add(key)
            out.append((dev, sr, ch, label))
        return out

    model_path = (SETTINGS.wakeword_model_path or "").strip()
    if model_path.lower().endswith(".table"):
        detector = MicrosoftKeywordDetector(model_path)
        if detector.available:
            logger.info("Microsoft Keyword detector loaded: %s", model_path)
        else:
            logger.warning(
                "Microsoft Keyword detector failed: %s — fallback to Deepgram.",
                detector.last_error,
            )
    else:
        detector = OpenWakeWordDetector(model_path, threshold=SETTINGS.wakeword_threshold)
        if detector.available:
            logger.info("OpenWakeWord loaded: %s", model_path)
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

    # Keep ~500ms of recent audio so first command words are not clipped on wake.
    frame_sec = float(SETTINGS.block_size) / max(1.0, float(target_sample_rate))
    pre_roll_frames = max(2, min(12, int(round(0.5 / max(0.01, frame_sec)))))
    pre_roll_buffer = deque(maxlen=pre_roll_frames)

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
    non_command_prompt_cooldown_sec = float(os.getenv("NON_COMMAND_PROMPT_COOLDOWN_SEC", "3.0"))
    last_non_command_prompt_ts = 0.0

    app.state.voice_ok = True
    wake_prompt_block_sec = float(os.getenv("WAKE_PROMPT_BLOCK_SEC", "1.2"))
    followup_listen_sec = float(os.getenv("FOLLOWUP_LISTEN_SEC", "12.0"))
    enable_barge_in = os.getenv("ENABLE_BARGE_IN", "true").lower() == "true"
    wake_parallel_vosk = os.getenv("WAKE_PARALLEL_VOSK", "true").lower() == "true"
    ms_wake_only_test = os.getenv("MS_WAKE_ONLY_TEST", "false").lower() == "true"
    # Extra gain path for low-input microphones (e.g. Bluetooth headsets) while sleeping.
    wake_target_rms = float(os.getenv("WAKE_TARGET_RMS", "0.08"))
    wake_max_gain = float(os.getenv("WAKE_MAX_GAIN", "256.0"))

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
    wake_phrases = _wake_phrase_variants()

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

    def _intent_response(intent: str, fallback_responses: dict) -> str:
        if not intent:
            return ""
        if intent.startswith("option_"):
            try:
                idx = int(intent.replace("option_", "")) - 1
            except Exception:
                idx = -1
            exp = str(app.state.session.get("selected_experience") or "")
            labels = EXPERIENCE_OPTION_LABELS.get(exp) or []
            if 0 <= idx < len(labels):
                return f"Selecting {labels[idx]}."
        return str(fallback_responses.get(intent, ""))

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
        # In ALWAYS_LISTEN mode, greet handling comes from command finalization.
        # Speaking here as well causes duplicate wake talkback.
        if not always_listening:
            _enqueue_tts(app.state.session["last_response"])
        if wake_prompt_block_sec > 0 and not always_listening:
            cooldown_until = max(cooldown_until, now_ts + wake_prompt_block_sec)
            app.state.cooldown_until = cooldown_until
        # Push pre-roll frames into ASR stream so we keep words said right after wake.
        if _ensure_deepgram(now_ts) and deepgram is not None and deepgram.connected:
            for frame in pre_roll_buffer:
                deepgram.send_audio(frame)

    def handle_frame(data):
        nonlocal wake_active, wake_start, listen_until, cooldown_until, last_partial_ts, last_rms_emit
        nonlocal fallback_hits, fallback_window_start, utterance_start, last_speech_ts
        nonlocal executing_until, deepgram, dg_next_connect_ts
        nonlocal last_non_command_prompt_ts
        nonlocal dg_partial_text, cmd_asr_source

        now = time.time()
        # Respect cooldown requests from other threads (e.g. TTS worker).
        shared_cooldown_until = float(getattr(app.state, "cooldown_until", 0.0) or 0.0)
        if shared_cooldown_until > cooldown_until:
            cooldown_until = shared_cooldown_until
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
        # Use raw input for wake boosting to avoid over-suppressing quiet speech.
        wake_gain = wake_target_rms / max(rms_raw, 1e-6)
        wake_gain = max(1.0, min(wake_max_gain, wake_gain))
        audio_f32_wake = np.clip(audio_f32_in * wake_gain, -1.0, 1.0)
        audio_f32_wake = _resample_f32(audio_f32_wake, input_sample_rate, target_sample_rate)
        wake_data_norm = (audio_f32_wake * 32767.0).astype(np.int16).tobytes()
        pre_roll_buffer.append(data_norm)

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
            nonlocal cooldown_until, wake_active, utterance_start, listen_until
            nonlocal executing_until
            nonlocal last_non_command_prompt_ts

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

            # In ALWAYS_LISTEN mode, treat a short "tom"/"hey tom" as a greet so the
            # system still provides the talkback prompt users expect.
            if (
                always_listening
                and len(normalized_text.split()) <= 3
                and _wake_phrase_match(normalized_text, wake_phrases)
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
            _log_command_event("transcript_final", text=text, asr_source=cmd_asr_source)

            if _stop_intent(text):
                tts_stop_event.set()

            bundle = _intent_bundle_for_source(cmd_asr_source) or {}
            intent_phrases = bundle.get("phrases", {}) if isinstance(bundle, dict) else {}
            intent_responses = bundle.get("responses", {}) if isinstance(bundle, dict) else {}
            intent_classifier = bundle.get("classifier", None) if isinstance(bundle, dict) else None

            pending = app.state.pending_confirmation
            if pending and time.time() <= float(pending.get("expires_at") or 0.0):
                if _confirm_yes(text):
                    app.state.metrics["cmd_confirm_accept"] += 1
                    intent = str(pending.get("intent") or "")
                    intent_conf = float(pending.get("confidence") or 0.0)
                    app.state.pending_confirmation = None
                    app.state.session["pending_confirmation"] = None
                    app.state.session["debug"] = ""
                    _log_command_event("confirmation_yes", intent=intent)
                    if intent:
                        app.state.session["last_response"] = f"Confirmed {intent.replace('_', ' ')}."
                        _enqueue_tts(app.state.session["last_response"])
                elif _confirm_no(text):
                    app.state.metrics["cmd_confirm_reject"] += 1
                    app.state.pending_confirmation = None
                    app.state.session["pending_confirmation"] = None
                    app.state.session["last_intent"] = "cancel"
                    app.state.session["last_response"] = "Okay, cancelled."
                    app.state.session["debug"] = "Pending command cancelled"
                    app.state.session["intent_seq"] += 1
                    _log_command_event("confirmation_no")
                    _emit_state()
                    _enqueue_tts(app.state.session["last_response"])
                    cooldown_until = now + SETTINGS.cooldown_sec
                    app.state.cooldown_until = cooldown_until
                    if always_listening:
                        set_mode("LISTENING", now)
                    else:
                        set_mode("EXECUTING", now)
                        executing_until = now + max(0.05, min(0.5, executing_hold_sec))
                        reset_listening(to_idle=False)
                    return
                else:
                    app.state.session["last_response"] = "Please say yes or no."
                    app.state.session["debug"] = "Awaiting confirmation"
                    _emit_state()
                    _enqueue_tts(app.state.session["last_response"])
                    return
            else:
                if pending:
                    app.state.pending_confirmation = None
                    app.state.session["pending_confirmation"] = None
                intent = None
                intent_conf = 0.0

            if _emergency_all_off(text):
                intent = "all_sound_off"
                intent_conf = 1.0

            if not intent:
                intent = match_intent_exact(text, intent_phrases)
                if intent:
                    intent_conf = 1.0
            if not intent:
                # Substring fallback: catches intent phrases embedded in longer ASR outputs.
                intent = match_intent_contains(text, intent_phrases)
                if intent:
                    intent_conf = 0.93
            if not intent:
                # Keyword fallback for robust command handling in noisy environments.
                intent = match_intent_keywords(text)
                if intent:
                    intent_conf = 0.88
            if not intent:
                # Context fallback: allow short option words after choosing an experience.
                current_exp = app.state.session.get("selected_experience")
                intent = _contextual_intent_from_text(text, current_exp)
                if intent:
                    intent_conf = 0.86
            if not intent and intent_classifier is not None:
                intent, intent_conf = classify_intent(text, intent_classifier, SETTINGS.intent_threshold)

            if intent:
                if DEMO_GRAMMAR_LOCK and intent not in DEMO_ALLOWED_INTENTS:
                    app.state.metrics["cmd_blocked_by_grammar"] += 1
                    app.state.session["debug"] = f"Blocked by demo grammar: {intent}"
                    app.state.session["last_intent"] = "unknown"
                    app.state.session["last_response"] = "That command is not allowed in this demo."
                    app.state.session["intent_seq"] += 1
                    _log_command_event("blocked_grammar", intent=intent, text=text)
                    _emit_state()
                    _enqueue_tts(app.state.session["last_response"])
                    return

                if (
                    intent in CONFIRM_FOR_INTENTS
                    and float(intent_conf or 0.0) < max(0.0, min(1.0, INTENT_CONFIRM_THRESHOLD))
                ):
                    app.state.metrics["cmd_confirmations"] += 1
                    app.state.pending_confirmation = {
                        "intent": intent,
                        "text": text,
                        "confidence": float(intent_conf or 0.0),
                        "expires_at": time.time() + INTENT_CONFIRM_WINDOW_SEC,
                    }
                    app.state.session["pending_confirmation"] = dict(app.state.pending_confirmation)
                    app.state.session["last_intent"] = "confirm"
                    app.state.session["last_response"] = f"Did you mean {intent.replace('_', ' ')}?"
                    app.state.session["debug"] = (
                        f"Low confidence {float(intent_conf or 0.0):.2f}, awaiting confirmation"
                    )
                    app.state.session["intent_seq"] += 1
                    _log_command_event(
                        "confirmation_requested",
                        intent=intent,
                        confidence=round(float(intent_conf or 0.0), 3),
                        text=text,
                    )
                    _emit_state()
                    _enqueue_tts(app.state.session["last_response"])
                    return

                app.state.session["last_intent"] = intent
                app.state.session["last_response"] = _intent_response(intent, intent_responses)
                app.state.session["last_final"] = text
                app.state.session["debug"] = ""
                app.state.session["intent_seq"] += 1
                _emit_state()

                if intent == "cancel":
                    reset_listening(to_idle=True)
                    if app.state.session["last_response"]:
                        _enqueue_tts(app.state.session["last_response"])
                    return

                _enqueue_command_action(intent, text=text, confidence=float(intent_conf or 0.0), source=cmd_asr_source)
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
                if always_listening and not _looks_like_command_text(text):
                    # In ALWAYS_LISTEN mode, provide a throttled prompt so users know
                    # the mic is active and what kind of utterance is expected.
                    app.state.metrics["asr_errors"] += 1
                    app.state.session["last_intent"] = "unknown"
                    app.state.session["last_final"] = text
                    app.state.session["last_response"] = intent_responses.get(
                        "unknown",
                        "I did not understand that. Please try a showroom command.",
                    )
                    app.state.session["debug"] = "Ignored non-command speech"
                    app.state.session["intent_seq"] += 1
                    _emit_state()
                    if (
                        app.state.session["last_response"]
                        and (now - last_non_command_prompt_ts) >= max(0.2, non_command_prompt_cooldown_sec)
                    ):
                        _enqueue_tts(app.state.session["last_response"])
                        last_non_command_prompt_ts = now
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

                # In ALWAYS_LISTEN mode, unknown TTS can become noisy/repetitive.
                if app.state.session["last_response"] and not always_listening:
                    _enqueue_tts(app.state.session["last_response"])

            keep_followup_session = bool(
                (not always_listening)
                and intent
                and intent not in {"cancel", "bye", "back", "all_sound_off", "mute"}
            )

            cooldown_until = now + SETTINGS.cooldown_sec
            app.state.cooldown_until = cooldown_until
            if always_listening:
                set_mode("LISTENING", now)
                _reset_cmd_buffer()
                wake_active = True
            elif keep_followup_session:
                # Keep listening for short follow-up commands like:
                # "hey tom welcome" -> "soft" -> "volume up".
                wake_active = True
                listen_until = now + max(1.5, followup_listen_sec)
                app.state.session["listen_until"] = listen_until
                set_mode("LISTENING", now)
                _reset_cmd_buffer()
                utterance_start = None
            else:
                set_mode("EXECUTING", now)
                executing_until = now + max(0.05, min(0.5, executing_hold_sec))
                reset_listening(to_idle=False)

        # Save Deepgram usage: only stream ASR audio when we actually need transcription.
        needs_asr_stream = bool(
            always_listening
            or wake_active
            or in_cooldown
            or SETTINGS.force_asr_wake_phrase
            or (not detector.available)
            or (rms_raw > (SETTINGS.noise_gate_rms * 0.7))
        )
        # Diagnostic mode: while sleeping, rely only on Microsoft keyword detector.
        # This helps verify whether the custom .table model itself is triggering.
        if (
            ms_wake_only_test
            and detector.available
            and not wake_active
            and not always_listening
            and not in_cooldown
        ):
            needs_asr_stream = False
        if needs_asr_stream:
            # While sleeping, prefer wake-boosted frames so "hey tom" can be
            # recognized on low-sensitivity microphones.
            asr_frame = wake_data_norm if (not wake_active and not always_listening) else data_norm
            deepgram_events: list[dict] = []
            vosk_events: list[dict] = []
            if _ensure_deepgram(now):
                deepgram.send_audio(asr_frame)
                deepgram_events = deepgram.drain_events()
                asr_source = "deepgram"
            else:
                if vosk_ready and vosk_stream is not None:
                    vosk_events = vosk_stream.process_audio(asr_frame)
                    asr_source = "vosk"
                    app.state.voice_ok = True
                    app.state.metrics["asr_fallback_to_vosk"] += 1
                else:
                    asr_source = "none"

            # Wake-only safety net: keep a local Vosk stream running in parallel
            # while sleeping so wake phrase detection doesn't depend on a single ASR.
            if (
                wake_parallel_vosk
                and not wake_active
                and not always_listening
                and vosk_ready
                and vosk_stream is not None
            ):
                extra = vosk_stream.process_audio(asr_frame)
                if extra:
                    vosk_events.extend(extra)

            events = deepgram_events + vosk_events
            if deepgram_events and vosk_events:
                asr_source = "hybrid"
            elif vosk_events and asr_source == "none":
                asr_source = "vosk"
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
                    app.state.metrics["deepgram_failovers"] += 1
                continue

            if event.get("type") != "Results":
                continue

            channel = event.get("channel") or {}
            alternatives = channel.get("alternatives") or []
            transcript = ""
            confidence = None
            if alternatives:
                first_alt = alternatives[0] or {}
                transcript = (first_alt.get("transcript") or "").strip()
                try:
                    confidence = float(first_alt.get("confidence"))
                except Exception:
                    confidence = None
            if not transcript:
                continue

            normalized = norm_join(transcript)
            wake_match = _wake_phrase_match(normalized, wake_phrases)
            soft_wake_min_rms = float(os.getenv("WAKE_SOFT_MIN_RMS", "0.001"))
            soft_wake = (
                (not wake_match)
                and (not wake_active)
                and (not always_listening)
                and bool(SETTINGS.force_asr_wake_phrase)
                and (rms_raw >= soft_wake_min_rms)
                and _soft_wake_greeting_match(transcript)
            )
            if soft_wake:
                wake_match = True
                app.state.session["debug"] = f"Soft wake fallback: {transcript!r}"
            is_final = bool(event.get("is_final"))
            event_source = event.get("source") or asr_source or "deepgram"
            if (
                is_final
                and event_source == "deepgram"
                and confidence is not None
                and confidence < SETTINGS.asr_min_conf
                and not wake_match
            ):
                app.state.session["debug"] = (
                    f"Deepgram confidence {confidence:.2f} below ASR_MIN_CONF {SETTINGS.asr_min_conf:.2f}"
                )
                _emit_state()
                continue
            _record_transcript(transcript, is_final, now)
            in_cooldown_now = time.time() < cooldown_until

            # Echo suppression: if ASR hears our own currently playing TTS phrase,
            # ignore it to prevent self-trigger loops (command -> TTS -> ASR -> command).
            if app.state.tts_playing:
                current_tts_norm = norm_join(str(getattr(app.state, "last_tts_text", "") or ""))
                if (
                    current_tts_norm
                    and normalized
                    and (
                        current_tts_norm in normalized
                        or normalized in current_tts_norm
                    )
                ):
                    app.state.session["debug"] = "Ignoring transcript matched to active TTS"
                    _emit_state()
                    continue

            # Barge-in: if user starts speaking while wake response TTS is still playing,
            # stop TTS and keep processing the command instead of ignoring it.
            if (
                enable_barge_in
                and app.state.tts_playing
                and (wake_active or always_listening)
                and not _stop_intent(transcript)
            ):
                tts_stop_event.set()
                app.state.session["debug"] = "Barge-in: stopping TTS to capture command"
                _emit_state()

            if app.state.tts_playing and _stop_intent(transcript):
                tts_stop_event.set()
                app.state.session["debug"] = "TTS interrupted by stop command"
                _emit_state()
                continue
            if in_cooldown_now and _stop_intent(transcript):
                tts_stop_event.set()
                cooldown_until = 0.0
                app.state.cooldown_until = cooldown_until
                app.state.session["transcript_partial"] = ""
                app.state.session["debug"] = ""
                set_mode("SLEEPING", time.time())
                continue
            # Allow explicit wake phrase to re-wake even during cooldown/TTS while idle.
            if (
                wake_match
                and not wake_active
                and not always_listening
            ):
                if app.state.tts_playing:
                    tts_stop_event.set()
                cooldown_until = 0.0
                app.state.cooldown_until = 0.0
                app.state.session["debug"] = ""
                _trigger_wake(now)
                continue
            # Allow re-wake during cooldown once TTS is no longer playing.
            if (
                in_cooldown_now
                and not app.state.tts_playing
                and not wake_active
                and not always_listening
                and wake_match
            ):
                cooldown_until = 0.0
                app.state.cooldown_until = 0.0
                app.state.session["debug"] = ""
                _trigger_wake(now)
                continue
            # If we are already in a follow-up listening session, don't keep blocking
            # commands just because cooldown is still ticking.
            if in_cooldown_now and wake_active and not app.state.tts_playing:
                cooldown_until = 0.0
                app.state.cooldown_until = 0.0
                in_cooldown_now = False
            if (
                (app.state.tts_playing and not wake_active and not always_listening)
                or (in_cooldown_now and not wake_active and not always_listening)
            ) and not _stop_intent(transcript):
                app.state.session["debug"] = "Ignoring transcript during TTS/cooldown"
                _emit_state()
                continue

            if not wake_active and not always_listening:
                if wake_match:
                    if now - fallback_window_start > SETTINGS.fallback_hit_window_sec:
                        fallback_window_start = now
                        fallback_hits = 0
                    fallback_hits += 1
                    full_phrase_hit = any((" " in wp) and (wp in normalized) for wp in wake_phrases)
                    required_hits = 1 if full_phrase_hit else max(
                        1, int(SETTINGS.fallback_required_hits)
                    )
                    if fallback_hits >= required_hits:
                        _trigger_wake(now)
                else:
                    fallback_hits = 0
                continue

            if is_final:
                dg_cmd_finals.append(transcript)
                cmd_asr_source = event_source
                dg_partial_text = ""
                # ASR endpointing says utterance ended; execute immediately.
                if wake_active or always_listening:
                    finalize_command()
                    break
            else:
                cmd_asr_source = event_source
                dg_partial_text = transcript

        if detector.available and not wake_active and not in_cooldown and not always_listening:
            oww_score = detector.score(audio_f32_wake, src_sample_rate=target_sample_rate)
            app.state.session["oww_score"] = oww_score
            app.state.session["oww_threshold"] = float(getattr(detector, "threshold", 1.0) or 1.0)

            # Microsoft detector returns 1.0 on trigger, OWW returns a float score
            threshold = float(getattr(detector, "threshold", 1.0) or 1.0)
            if oww_score >= threshold:
                logger.info("Wake word detected! score=%.3f threshold=%.3f", oww_score, threshold)
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
            speech_threshold = max(SETTINGS.min_speech_rms, SETTINGS.noise_gate_rms * 0.20)
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

        reconnect_delay_sec = 0.75
        open_fail_streak = 0
        while not stop_event.is_set():
            stream_opened = False
            for device, input_sample_rate, channels, label in _stream_open_candidates():
                if stop_event.is_set():
                    break
                if channels <= 0:
                    continue
                if _is_wdmks_device(device):
                    # Blocking read mode is unstable/unsupported for many WDM-KS paths.
                    logger.warning("Skipping WDM-KS input candidate (%s device=%r)", label, device)
                    continue
                try:
                    with sd.RawInputStream(
                        samplerate=input_sample_rate,
                        blocksize=SETTINGS.block_size,
                        dtype="int16",
                        channels=channels,
                        device=device,
                    ) as stream:
                        stream_opened = True
                        open_fail_streak = 0
                        reconnect_delay_sec = 0.75
                        logger.info(
                            "Voice capture started (%s device=%r sr=%s ch=%s)",
                            label,
                            device,
                            input_sample_rate,
                            channels,
                        )
                        # Clear previous transient audio-device errors once capture recovers.
                        if "device invalidated" in str(app.state.session.get("debug", "")).lower() or "mic stream error" in str(app.state.session.get("debug", "")).lower():
                            app.state.session["debug"] = ""
                            _emit_state()

                        while not stop_event.is_set():
                            try:
                                data, _ = stream.read(SETTINGS.block_size)
                            except Exception as exc:
                                if stop_event.is_set():
                                    break
                                app.state.metrics["asr_errors"] += 1
                                if _is_device_invalidated_error(exc):
                                    msg = f"Mic device invalidated; reopening stream: {exc}"
                                    app.state.session["debug"] = msg
                                    _emit_state()
                                    logger.warning("%s", msg)
                                    break
                                # Any read error: break to reopen with next candidate.
                                raise

                            if channels and channels > 1:
                                mono, best_ch, rms_by_ch = _select_loudest_channel_int16(
                                    data,
                                    channels,
                                    SETTINGS.mic_channel_lock,
                                )
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

                        # Stream ended gracefully or invalidated; retry candidate selection.
                        break
                except Exception as exc:
                    if stop_event.is_set():
                        break
                    app.state.metrics["asr_errors"] += 1
                    open_fail_streak += 1
                    app.state.session["debug"] = f"Mic stream error; retrying: {exc}"
                    _emit_state()
                    if open_fail_streak <= 3 or (open_fail_streak % 5) == 0:
                        logger.warning(
                            "Mic stream open/read failed (%s); retrying: %s",
                            label,
                            exc,
                        )
                    if _is_insufficient_memory_error(exc):
                        # PortAudio/WASAPI can get wedged after device hot-swap. Re-init backend.
                        try:
                            sd._terminate()
                        except Exception:
                            pass
                        try:
                            sd._initialize()
                        except Exception:
                            pass
                        reconnect_delay_sec = min(4.0, reconnect_delay_sec * 1.35)
                    if _is_blocking_api_unsupported_error(exc):
                        # Fast-track to the next candidate/device backend.
                        continue
                if stream_opened:
                    break

            if not stream_opened:
                # No candidate succeeded this cycle.
                app.state.session["debug"] = "Mic unavailable; retrying with fallback devices/configs"
                _emit_state()
                logger.warning("Mic stream unavailable; all open candidates failed")
            time.sleep(reconnect_delay_sec)
    except Exception as exc:
        app.state.voice_ok = False
        app.state.session["debug"] = f"Voice worker crashed: {exc}"
        _emit_state()
        logger.exception("Voice worker crashed: %s", exc)
    finally:
        app.state.voice_ok = False
        if deepgram is not None:
            deepgram.close()
        try:
            detector.close()
        except Exception:
            pass


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
        "command_queue_depth": command_queue.qsize(),
        "metrics": dict(app.state.metrics),
    })


@app.get("/status")
async def status():
    return _status_payload()


@app.get("/metrics")
async def metrics():
    return _json_sanitize(
        {
            "ok": True,
            "queue": {
                "depth": command_queue.qsize(),
                "max": COMMAND_QUEUE_MAX,
            },
            "metrics": dict(app.state.metrics),
            "asr": {
                "source": app.state.session.get("asr_source", "none"),
                "voice_ok": app.state.voice_ok,
                "deepgram_model": SETTINGS.deepgram_model,
                "vosk_enabled": bool(SETTINGS.vosk_model_path),
            },
            "safety": {
                "demo_grammar_lock": DEMO_GRAMMAR_LOCK,
                "allowed_intents": sorted(DEMO_ALLOWED_INTENTS),
                "confirm_threshold": INTENT_CONFIRM_THRESHOLD,
                "rate_limit_sec": COMMAND_RATE_LIMIT_SEC,
                "debounce_sec": COMMAND_DEBOUNCE_SEC,
            },
        }
    )


@app.get("/asr/validate_failover")
async def validate_failover():
    fallbacks = int(app.state.metrics.get("asr_fallback_to_vosk", 0))
    failovers = int(app.state.metrics.get("deepgram_failovers", 0))
    return _json_sanitize(
        {
            "ok": True,
            "deepgram_failovers": failovers,
            "vosk_fallback_frames": fallbacks,
            "validation": {
                "fallback_path_active": bool(fallbacks > 0 or failovers > 0),
                "voice_ok": bool(app.state.voice_ok),
                "current_asr_source": app.state.session.get("asr_source", "none"),
            },
        }
    )


@app.post("/asr/failover_test")
async def asr_failover_test(simulate: bool = True):
    """
    Lightweight validation hook for monitoring pipeline behavior without hardware changes.
    """
    if simulate:
        app.state.metrics["deepgram_failovers"] += 1
        if SETTINGS.vosk_model_path:
            app.state.metrics["asr_fallback_to_vosk"] += 1
    return _json_sanitize(
        {
            "ok": True,
            "simulated": bool(simulate),
            "deepgram_failovers": int(app.state.metrics.get("deepgram_failovers", 0)),
            "vosk_fallback_frames": int(app.state.metrics.get("asr_fallback_to_vosk", 0)),
            "vosk_configured": bool(SETTINGS.vosk_model_path),
        }
    )


@app.post("/ui_sync")
async def ui_sync(request: Request):
    """
    Sync manual frontend UI state so backend command context stays aligned.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    session = app.state.session
    changed = False

    if "selected_experience" in payload:
        exp = payload.get("selected_experience")
        if exp is None or isinstance(exp, str):
            session["selected_experience"] = exp
            changed = True

    if "selected_option" in payload:
        opt = payload.get("selected_option")
        if opt is None or isinstance(opt, str):
            session["selected_option"] = opt
            changed = True

    if changed:
        session["navigate"] = None
        _emit_state()

    return _json_sanitize(
        {
            "ok": True,
            "synced": changed,
            "selected_experience": session.get("selected_experience"),
            "selected_option": session.get("selected_option"),
        }
    )


@app.post("/reload_intents")
async def reload_intents():
    global INTENT_BUNDLES, VOICE_PRESET
    INTENT_BUNDLES = _load_intent_bundles()
    _refresh_command_hint_tokens()
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
async def models_check(stream: bool = False, interval_ms: int = 2000):
    """Check which model files/packages are present. Supports SSE with stream=true."""
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

    def _payload():
        piper_exe_path = shutil.which(SETTINGS.piper_path) or SETTINGS.piper_path
        piper_exe_exists = shutil.which(SETTINGS.piper_path) is not None or _Path(SETTINGS.piper_path).is_file()
        oww_requested = (SETTINGS.wakeword_model_path or "").strip()
        oww_path, oww_name = resolve_openwakeword_model_path(oww_requested)
        selected_wake_model = (oww_path or oww_requested or "").strip().lower()
        wake_model_label = (
            f"Microsoft Keyword Model (.table){f' ({oww_name})' if oww_name else ''}"
            if selected_wake_model.endswith(".table")
            else f"OpenWakeWord Model (.tflite){f' ({oww_name})' if oww_name else ''}"
        )
        if not oww_path and oww_requested:
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
                    "label": wake_model_label,
                    "name": oww_name,
                },
            },
            "packages": {
                "websocket_client":      {"installed": _check_pkg("websocket"),           "label": "websocket-client"},
                "vosk":                  {"installed": _check_pkg("vosk"),                "label": "vosk"},
                "noisereduce":           {"installed": _check_pkg("noisereduce"),         "label": "noisereduce"},
                "openwakeword":          {"installed": _check_pkg("openwakeword"),        "label": "openwakeword"},
                "azure_speech_sdk":      {"installed": _check_pkg("azure.cognitiveservices.speech"), "label": "azure-cognitiveservices-speech"},
                "sounddevice":           {"installed": _check_pkg("sounddevice"),         "label": "sounddevice"},
                "python_dotenv":         {"installed": _check_pkg("dotenv"),              "label": "python-dotenv"},
                "sentence_transformers": {"installed": _check_pkg("sentence_transformers"), "label": "sentence-transformers"},
            },
            "settings": {
                "tts_enabled":       SETTINGS.tts_enabled,
                "tts_backend":       SETTINGS.tts_backend,
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
                "demo_grammar_lock": DEMO_GRAMMAR_LOCK,
                "intent_confirm_threshold": INTENT_CONFIRM_THRESHOLD,
                "command_rate_limit_sec": COMMAND_RATE_LIMIT_SEC,
                "command_debounce_sec": COMMAND_DEBOUNCE_SEC,
                "command_queue_max": COMMAND_QUEUE_MAX,
            },
            "runtime": {
                "noise_profile_ready": app.state.session.get("noise_profile_ready", False),
                "noise_gate_rms":      round(SETTINGS.noise_gate_rms, 5),
                "voice_ok":            app.state.voice_ok,
            },
        })

    if not stream:
        return _payload()

    interval_sec = max(0.5, min(10.0, float(interval_ms) / 1000.0))

    async def _event_stream():
        while True:
            yield f"data: {json.dumps(_payload())}\n\n"
            await asyncio.sleep(interval_sec)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/mic_test")
async def mic_test(stream: bool = False, interval_ms: int = 200):
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

    def _payload():
        return _json_sanitize(
            {
                "ok": True,
                "timestamp": round(time.time(), 3),
                "audio_source": SETTINGS.audio_source,
                "mode": app.state.mode,
                "voice_rms": app.state.session.get("voice_rms", 0.0),
                "voice_rms_raw": app.state.session.get("voice_rms_raw", 0.0),
                "voice_rms_norm": app.state.session.get("voice_rms_norm", None),
                "mic_channel": app.state.session.get("mic_channel", None),
                "mic_rms_channels": app.state.session.get("mic_rms_channels", None),
                "audio_device_env": SETTINGS.device,
                "selected_device": selected_device,
                "default_input_device": device_info,
            }
        )

    if not stream:
        return _payload()

    interval_sec = max(0.05, min(2.0, float(interval_ms) / 1000.0))

    async def _event_stream():
        while True:
            yield f"data: {json.dumps(_payload())}\n\n"
            await asyncio.sleep(interval_sec)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    app.state.command_task = asyncio.create_task(asyncio.to_thread(command_worker))


@app.on_event("shutdown")
async def on_shutdown():
    stop_event.set()
    event_queue.put(None)
    tts_queue.put(None)
    command_queue.put(None)

    task = getattr(app.state, "broadcaster_task", None)
    if task:
        task.cancel()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    await websocket.send_text(json.dumps({"type": "status_snapshot", "status": _status_payload()}))
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
                # Keep stream real-time: discard the oldest queued frame, keep latest audio.
                try:
                    _ = audio_queue.get_nowait()
                    audio_queue.task_done()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(data)
                except queue.Full:
                    pass
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
