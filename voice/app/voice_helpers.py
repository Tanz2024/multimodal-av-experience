import json
import logging
import os
import re
import subprocess
import tempfile
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from threading import Event

DEFAULT_INTENT_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "intent_phrases.json")
logger = logging.getLogger("voice-wake-sherpa")


@dataclass
class Settings:
    wake_word: str = os.getenv("WAKE_WORD", "jarvis")
    wake_phrase: str = os.getenv("WAKE_PHRASE", "hey jarvis")
    sample_rate: int = 16000
    block_size: int = 2048
    wake_window_sec: float = 6.0
    max_listen_sec: float = 12.0
    cooldown_sec: float = 1.5
    fallback_hit_window_sec: float = 1.2
    fallback_required_hits: int = 2
    vad_silence_sec: float = float(os.getenv("VAD_SILENCE_SEC", "0.8"))
    intent_threshold: float = float(os.getenv("INTENT_THRESHOLD", "0.7"))
    intent_config_path: str = os.getenv(
        "INTENT_CONFIG_PATH",
        DEFAULT_INTENT_CONFIG_PATH,
    )
    intent_config_deepgram_path: str = os.getenv(
        "INTENT_CONFIG_DEEPGRAM_PATH",
        os.getenv("INTENT_CONFIG_PATH", DEFAULT_INTENT_CONFIG_PATH),
    )
    intent_config_vosk_path: str = os.getenv(
        "INTENT_CONFIG_VOSK_PATH",
        os.getenv("INTENT_CONFIG_PATH", DEFAULT_INTENT_CONFIG_PATH),
    )
    listen_window_sec: float = float(os.getenv("LISTEN_WINDOW_SEC", "8.0"))
    max_utterance_sec: float = float(os.getenv("MAX_UTTERANCE_SEC", "3.0"))
    asr_min_conf: float = float(os.getenv("ASR_MIN_CONF", "0.55"))
    tts_enabled: bool = os.getenv("TTS_ENABLED", "false").lower() == "true"
    tts_backend: str = os.getenv("TTS_BACKEND", "piper").lower()
    piper_path: str = os.getenv("PIPER_PATH", "piper")
    piper_model_path: str = os.getenv("PIPER_MODEL_PATH", "")
    piper_speaker: str = os.getenv("PIPER_SPEAKER", "")
    piper_length_scale: str = os.getenv("PIPER_LENGTH_SCALE", "1.0")
    piper_noise_scale: str = os.getenv("PIPER_NOISE_SCALE", "")
    piper_noise_w: str = os.getenv("PIPER_NOISE_W", "")
    duck_enabled: bool = os.getenv("DUCK_ENABLED", "false").lower() == "true"
    duck_ratio: float = float(os.getenv("DUCK_RATIO", "0.35"))
    audio_source: str = os.getenv("AUDIO_SOURCE", "device").lower()
    always_listening: bool = os.getenv("ALWAYS_LISTEN", "false").lower() == "true"
    noise_gate_rms: float = float(os.getenv("NOISE_GATE_RMS", "0.015"))
    target_rms: float = float(os.getenv("TARGET_RMS", "0.06"))
    max_gain: float = float(os.getenv("MAX_GAIN", "8.0"))
    wake_response: str = os.getenv("WAKE_RESPONSE", "Hello! What experience would you like to try today?")
    wakeword_model_path: str = os.getenv("OWW_MODEL_PATH", "")
    wakeword_threshold: float = float(os.getenv("OWW_THRESHOLD", "0.6"))
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-3")
    deepgram_language: str = os.getenv("DEEPGRAM_LANGUAGE", "en-US")
    deepgram_endpointing_ms: int = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "300"))
    vosk_model_path: str = os.getenv("VOSK_MODEL_PATH", "")
    transcript_debug_enabled: bool = os.getenv("TRANSCRIPT_DEBUG_ENABLED", "true").lower() == "true"
    transcript_debug_max: int = int(os.getenv("TRANSCRIPT_DEBUG_MAX", "25"))
    device: str = os.getenv("AUDIO_DEVICE", "")
    # Noise reduction: captures ambient profile during SLEEPING, then subtracts from speech
    noise_reduction_enabled: bool = os.getenv("NOISE_REDUCTION", "true").lower() == "true"
    noise_reduction_prop: float = float(os.getenv("NOISE_REDUCTION_PROP", "0.80"))
    # Auto noise gate: after profile capture, measure residual and set threshold automatically.
    # Eliminates the need to manually tune NOISE_GATE_RMS for different music / venues.
    noise_gate_auto: bool = os.getenv("NOISE_GATE_AUTO", "true").lower() == "true"
    # Gate = residual_rms × multiplier.  3.0 means "3× the post-reduction noise floor".
    # Raise if you still get false triggers; lower if quiet voices get cut off.
    noise_gate_multiplier: float = float(os.getenv("NOISE_GATE_MULTIPLIER", "3.0"))
    # ASR backend identifier surfaced to frontend status panel.
    asr_backend: str = "deepgram"
    # Extra silence window after TTS finishes before mic re-opens (avoids TTS echo)
    tts_cooldown_sec: float = float(os.getenv("TTS_COOLDOWN_SEC", "0.4"))


def normalize_tokens(text: str) -> list[str]:
    return "".join([c.lower() if c.isalnum() or c.isspace() else " " for c in text]).split()


def norm_join(text: str) -> str:
    return " ".join(normalize_tokens(text))


def _token_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    # Handle small ASR spelling drifts (e.g. "cinima" vs "cinema", "volum" vs "volume").
    if len(a) >= 4 and len(b) >= 4 and abs(len(a) - len(b)) <= 1:
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
        if edits <= 1:
            return True
    # Handle adjacent transposition errors from ASR ("welocm" vs "welcome").
    if len(a) == len(b) and len(a) >= 5:
        diffs = [i for i, (ca, cb) in enumerate(zip(a, b)) if ca != cb]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i = diffs[0]
            if a[i] == b[i + 1] and a[i + 1] == b[i]:
                return True
    # Last-resort fuzzy similarity (prefer rapidfuzz for speed, fallback to difflib).
    try:
        from rapidfuzz import fuzz
        if len(a) >= 5 and len(b) >= 5 and fuzz.ratio(a, b) >= 82.0:
            return True
    except ImportError:
        if len(a) >= 6 and len(b) >= 6:
            if SequenceMatcher(None, a, b).ratio() >= 0.82:
                return True
    return False


def _contains_token_sequence(text_tokens: list[str], ex_tokens: list[str]) -> bool:
    if not text_tokens or not ex_tokens or len(ex_tokens) > len(text_tokens):
        return False
    window = len(ex_tokens)
    for start in range(0, len(text_tokens) - window + 1):
        ok = True
        for i in range(window):
            if not _token_similar(text_tokens[start + i], ex_tokens[i]):
                ok = False
                break
        if ok:
            return True
    # Fallback: allow filler words between phrase tokens
    # (e.g. "increase the volume please" should match "increase volume").
    ti = 0
    for ex_tok in ex_tokens:
        matched = False
        while ti < len(text_tokens):
            if _token_similar(text_tokens[ti], ex_tok):
                matched = True
                ti += 1
                break
            ti += 1
        if not matched:
            return False
    return True


def match_intent_contains(text: str, phrases: dict[str, list[str]]) -> str | None:
    normalized_text = norm_join(text)
    normalized = f" {normalized_text} "
    text_tokens = normalized_text.split()
    if normalized.strip() == "":
        return None
    best = None
    best_len = 0
    for intent, exs in phrases.items():
        for ex in exs:
            ex_norm = norm_join(ex)
            exn = f" {ex_norm} "
            ex_tokens = ex_norm.split()
            matched = exn.strip() and exn in normalized
            if (not matched) and ex_tokens:
                matched = _contains_token_sequence(text_tokens, ex_tokens)
            if matched and len(exn) > best_len:
                best = intent
                best_len = len(exn)
    return best


def match_intent_keywords(text: str) -> str | None:
    """
    Keyword fallback for common showroom commands when phrase matching misses.
    """
    tokens = set(normalize_tokens(text))
    if not tokens:
        return None

    if "welcome" in tokens or ("well" in tokens and "come" in tokens):
        return "welcome"
    if "kitchen" in tokens:
        return "kitchen"
    if "invisible" in tokens:
        return "invisible"
    if "cinema" in tokens:
        return "cinema"
    if "lounge" in tokens:
        return "lounge"
    if "iconic" in tokens:
        return "iconic"

    if {"mute", "silence"} & tokens:
        return "mute"
    if {"next"} & tokens:
        return "next"
    if {"previous", "prev"} & tokens:
        return "previous"
    if {"back"} & tokens:
        return "back"

    if "volume" in tokens or "sound" in tokens:
        if {"up", "raise", "higher", "increase", "louder"} & tokens:
            return "volume_up"
        if {"down", "lower", "decrease", "quieter", "softer"} & tokens:
            return "volume_down"

    return None


DEFAULT_INTENT_PHRASES = {
    "greet": ["hello", "hi", "hey", "are you there", "can you hear me"],
    "help": ["help", "what can you do", "what can i say", "commands", "show commands", "options", "list options"],
    "bye": ["bye", "goodbye", "thanks", "thank you", "thanks for using jarvis", "see you"],
    "back": ["back", "go back", "close", "close this", "close that", "exit", "dismiss", "close the panel", "go back please"],
    "cancel": ["cancel", "never mind", "stop listening", "stop listening now", "cancel listening"],
    "next": ["next", "next option", "next one", "move next"],
    "previous": ["previous", "prev", "previous option", "go previous", "go back option", "last option"],
    "mute": ["mute", "sound off", "all off", "turn off sound", "silence", "mute audio", "turn it off", "stop sound"],
    "all_sound_off": ["all sound off", "all off", "sound off", "stop sound", "turn off sound"],
    "welcome": ["welcome", "open welcome", "open welcome experience", "welcome experience", "show welcome", "go to welcome", "open the welcome"],
    "kitchen": ["kitchen", "open kitchen", "open kitchen experience", "kitchen experience", "show kitchen", "open the kitchen"],
    "invisible": ["invisible", "open invisible", "open invisible experience", "invisible experience", "show invisible", "open the invisible"],
    "cinema": ["cinema", "open cinema", "open cinema experience", "cinema experience", "show cinema", "open the cinema"],
    "lounge": ["lounge", "entertain", "open lounge", "open lounge experience", "lounge experience", "show lounge", "open the lounge"],
    "iconic": ["iconic", "open iconic", "open iconic experience", "iconic experience", "show iconic", "open the iconic"],
    "open_1": ["open one", "open 1", "open experience one", "open experience 1"],
    "open_2": ["open two", "open 2", "open experience two", "open experience 2"],
    "open_3": ["open three", "open 3", "open experience three", "open experience 3"],
    "open_4": ["open four", "open 4", "open experience four", "open experience 4"],
    "open_5": ["open five", "open 5", "open experience five", "open experience 5"],
    "open_6": ["open six", "open 6", "open experience six", "open experience 6"],
    "open_7": ["open seven", "open 7", "open experience seven", "open experience 7"],
    "option_1": ["option one", "option 1", "select option one", "choose option one", "pick option one"],
    "option_2": ["option two", "option 2", "select option two", "choose option two", "pick option two"],
    "option_3": ["option three", "option 3", "select option three", "choose option three", "pick option three"],
    "option_4": ["option four", "option 4", "select option four", "choose option four", "pick option four"],
    "option_5": ["option five", "option 5", "select option five", "choose option five", "pick option five"],
    "option_6": ["option six", "option 6", "select option six", "choose option six", "pick option six"],
    "option_7": ["option seven", "option 7", "select option seven", "choose option seven", "pick option seven"],
    "volume_up": ["volume up", "increase volume", "turn up", "louder", "raise volume"],
    "volume_down": ["volume down", "decrease volume", "turn down", "quieter", "lower volume"],
}

DEFAULT_INTENT_RESPONSES = {
    "greet": "How may I help you?",
    "help": "You can say Welcome, Kitchen, Invisible, Cinema, Lounge, Iconic, All Sound Off, Volume up, Volume down, Next, Previous, or Back.",
    "bye": "Bye, thanks for using Jarvis.",
    "back": "Back.",
    "cancel": "Okay, I will stop listening.",
    "next": "Next.",
    "previous": "Previous.",
    "mute": "Muted.",
    "all_sound_off": "Muted.",
    "welcome": "Welcome.",
    "kitchen": "Kitchen.",
    "invisible": "Invisible.",
    "cinema": "Cinema.",
    "lounge": "Lounge.",
    "iconic": "Iconic.",
    "open_1": "Option one.",
    "open_2": "Option two.",
    "open_3": "Option three.",
    "open_4": "Option four.",
    "open_5": "Option five.",
    "open_6": "Option six.",
    "open_7": "Option seven.",
    "option_1": "Option one.",
    "option_2": "Option two.",
    "option_3": "Option three.",
    "option_4": "Option four.",
    "option_5": "Option five.",
    "option_6": "Option six.",
    "option_7": "Option seven.",
    "volume_up": "Volume up.",
    "volume_down": "Volume down.",
    "unknown": "Sorry, I can't help with that.",
}

VOICE_PRESET = {
    "model": "en_US-kathleen-low",
    "model_path": "",
    "length_scale": 0.95,
    "noise_scale": 0.6,
    "noise_w": 0.7,
}


class IntentClassifier:
    def __init__(self, phrases: dict[str, list[str]]):
        self.phrases = phrases
        self.intent_labels = list(phrases.keys())
        self.examples = []
        self.example_intents = []
        for intent, exs in phrases.items():
            for ex in exs:
                self.examples.append(ex)
                self.example_intents.append(intent)
        self.model = None
        self.embeddings = None
        self.np = None
        use_embeddings = os.getenv("USE_EMBEDDINGS", "false").lower() == "true"
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np

                logger.info("Loading NLP embedding model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embeddings = self.model.encode(self.examples, normalize_embeddings=True)
                self.np = np
                logger.info("NLP embedding model loaded successfully")
            except Exception as exc:
                logger.error("Failed to load NLP model: %s", exc)
                logger.warning("Falling back to basic string matching")
                self.model = None
                self.embeddings = None

    def predict(self, text: str):
        if not self.model or self.embeddings is None:
            normalized = norm_join(text)
            for intent, exs in self.phrases.items():
                for ex in exs:
                    if normalized == norm_join(ex):
                        return intent, 1.0
            return None, 0.0
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        sims = self.embeddings @ vec
        idx = int(self.np.argmax(sims))
        score = float(sims[idx])
        return self.example_intents[idx], score


def load_intent_config(path: str):
    phrases = dict(DEFAULT_INTENT_PHRASES)
    responses = dict(DEFAULT_INTENT_RESPONSES)
    voice_preset = dict(VOICE_PRESET)
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg_phrases = data.get("phrases", DEFAULT_INTENT_PHRASES)
            cfg_responses = data.get("responses", DEFAULT_INTENT_RESPONSES)
            cfg_voice = data.get("voice_preset", VOICE_PRESET)
            if isinstance(cfg_phrases, dict) and isinstance(cfg_responses, dict):
                phrases = cfg_phrases
                responses = cfg_responses
                if isinstance(cfg_voice, dict):
                    voice_preset = cfg_voice
    except Exception:
        pass
    return phrases, responses, voice_preset, IntentClassifier(phrases)


def classify_intent(text: str, classifier: IntentClassifier, threshold: float):
    intent, score = classifier.predict(text)
    if not intent or score < threshold:
        return None, 0.0
    return intent, score


class CommandExecutor:
    def __init__(self, app):
        self.app = app
        self._system_volume = SystemVolumeController(
            enabled=os.getenv("SYSTEM_VOLUME_ENABLED", "true").lower() == "true",
            step=float(os.getenv("SYSTEM_VOLUME_STEP", "0.06")),
        )

    def execute(self, intent: str):
        session = self.app.state.session
        opt_map = {
            "opt_soft_welcome": ("welcome", "soft"),
            "opt_luxury_ambient": ("welcome", "luxury"),
            "opt_invisible_cinema": ("cinema", "invisible"),
            "opt_live_performance": ("cinema", "live"),
            "opt_luxury_apartment_night": ("cinema", "luxury"),
        }
        open_map = {
            "open_1": "welcome",
            "open_2": "kitchen",
            "open_3": "invisible",
            "open_4": "cinema",
            "open_5": "lounge",
            "open_6": "iconic",
            "open_7": "off",
        }

        if intent in ("welcome", "kitchen", "invisible", "cinema", "lounge", "iconic"):
            session["selected_experience"] = intent
            session["selected_option"] = None
            session["navigate"] = None
        elif intent in opt_map:
            exp, opt = opt_map[intent]
            session["selected_experience"] = exp
            session["selected_option"] = opt
            session["navigate"] = None
        elif intent == "opt_background_ambience":
            exp = "invisible" if session.get("selected_experience") == "invisible" else "kitchen"
            session["selected_experience"] = exp
            session["selected_option"] = "ambience2" if exp == "invisible" else "ambience"
            session["navigate"] = None
        elif intent == "opt_entertaining_mode":
            exp = "invisible" if session.get("selected_experience") == "invisible" else "kitchen"
            session["selected_experience"] = exp
            session["selected_option"] = "entertaining2" if exp == "invisible" else "entertaining"
            session["navigate"] = None
        elif intent == "opt_background_lounge":
            exp = "iconic" if session.get("selected_experience") == "iconic" else "lounge"
            session["selected_experience"] = exp
            session["selected_option"] = "lounge2" if exp == "iconic" else "lounge"
            session["navigate"] = None
        elif intent == "opt_cocktail_evening":
            exp = "iconic" if session.get("selected_experience") == "iconic" else "lounge"
            session["selected_experience"] = exp
            session["selected_option"] = "cocktail2" if exp == "iconic" else "cocktail"
            session["navigate"] = None
        elif intent == "opt_design_focus":
            exp = "iconic" if session.get("selected_experience") == "iconic" else "lounge"
            session["selected_experience"] = exp
            session["selected_option"] = "design2" if exp == "iconic" else "design"
            session["navigate"] = None
        elif intent == "opt_silent":
            current = session.get("selected_experience")
            if current == "invisible":
                exp, opt = "invisible", "silent2"
            elif current == "lounge":
                exp, opt = "lounge", "design"
            elif current == "iconic":
                exp, opt = "iconic", "design2"
            else:
                exp, opt = "kitchen", "silent"
            session["selected_experience"] = exp
            session["selected_option"] = opt
            session["navigate"] = None
        elif intent.startswith("open_"):
            session["selected_experience"] = open_map.get(intent, intent)
            session["selected_option"] = None
            session["navigate"] = None
        elif intent.startswith("option_"):
            session["selected_option"] = intent
            session["navigate"] = None
        elif intent in ("mute", "all_sound_off"):
            session["selected_experience"] = "off"
            session["selected_option"] = "off"
            self._system_volume.set_muted(True)
            session["volume"] = 0.0
            session["navigate"] = None
        elif intent in ("back", "cancel"):
            session["selected_experience"] = None
            session["selected_option"] = None
            session["navigate"] = None
        elif intent == "next":
            # Frontend navigates; backend signals direction so frontend can cycle options
            session["navigate"] = "next"
        elif intent == "previous":
            session["navigate"] = "previous"
        elif intent == "volume_up":
            sys_vol = self._system_volume.change(+1)
            if sys_vol is not None:
                session["volume"] = sys_vol
            else:
                session["volume"] = min(1.0, float(session.get("volume") or 0.0) + 0.1)
            session["navigate"] = None
        elif intent == "volume_down":
            sys_vol = self._system_volume.change(-1)
            if sys_vol is not None:
                session["volume"] = sys_vol
            else:
                session["volume"] = max(0.0, float(session.get("volume") or 0.0) - 0.1)
            session["navigate"] = None
        elif intent in ("greet", "help", "bye"):
            # No state change needed — TTS response is sufficient
            session["navigate"] = None
        else:
            session["navigate"] = None
        self.app.state.metrics["commands_executed"] += 1


class AudioDucker:
    def __init__(self, settings: Settings):
        self._backend = None
        self._volume = None
        self._prev_scalar = None
        self._prev_pactl_volume = None
        self.settings = settings

    def _read_pactl_percent(self) -> str | None:
        try:
            proc = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return None
            match = re.search(r"(\d+)%", proc.stdout or "")
            if not match:
                return None
            return f"{match.group(1)}%"
        except Exception:
            return None

    def _init_backend(self):
        if os.name == "nt":
            try:
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._volume = interface.QueryInterface(IAudioEndpointVolume)
                self._backend = "pycaw"
            except Exception:
                self._backend = None
        else:
            self._backend = "pactl"

    def duck(self):
        if not self.settings.duck_enabled:
            return
        if self._backend is None:
            self._init_backend()
        if self._backend == "pycaw" and self._volume:
            try:
                current = self._volume.GetMasterVolumeLevelScalar()
                self._prev_scalar = current
                target = max(0.0, min(1.0, current * self.settings.duck_ratio))
                self._volume.SetMasterVolumeLevelScalar(target, None)
            except Exception:
                return
        elif self._backend == "pactl":
            try:
                current = self._read_pactl_percent()
                self._prev_pactl_volume = current
                if current is not None:
                    current_pct = int(current.rstrip("%"))
                    target_pct = max(0, min(150, int(round(current_pct * self.settings.duck_ratio))))
                    target = f"{target_pct}%"
                else:
                    target = f"{int(self.settings.duck_ratio * 100)}%"
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", target],
                    check=False,
                )
            except Exception:
                return

    def unduck(self):
        if not self.settings.duck_enabled:
            return
        if self._backend == "pycaw" and self._volume:
            try:
                restore = 1.0 if self._prev_scalar is None else self._prev_scalar
                self._volume.SetMasterVolumeLevelScalar(restore, None)
            except Exception:
                return
        elif self._backend == "pactl":
            try:
                restore = self._prev_pactl_volume or self._read_pactl_percent() or "100%"
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", restore],
                    check=False,
                )
            except Exception:
                return


class SystemVolumeController:
    """
    Best-effort OS volume controller used by voice intents.
    """

    def __init__(self, enabled: bool = True, step: float = 0.06):
        self.enabled = bool(enabled)
        self.step = max(0.01, min(0.25, float(step)))
        self._backend = None
        self._volume = None

    def _init_backend(self):
        if self._backend is not None:
            return
        if os.name == "nt":
            try:
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._volume = interface.QueryInterface(IAudioEndpointVolume)
                self._backend = "pycaw"
                return
            except Exception:
                self._backend = "none"
                return
        self._backend = "pactl"

    def _read_pactl_percent(self) -> int | None:
        try:
            proc = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return None
            match = re.search(r"(\d+)%", proc.stdout or "")
            return int(match.group(1)) if match else None
        except Exception:
            return None

    def _scalar(self) -> float | None:
        if not self.enabled:
            return None
        self._init_backend()
        if self._backend == "pycaw" and self._volume is not None:
            try:
                return float(self._volume.GetMasterVolumeLevelScalar())
            except Exception:
                return None
        if self._backend == "pactl":
            pct = self._read_pactl_percent()
            if pct is None:
                return None
            return max(0.0, min(1.0, pct / 100.0))
        return None

    def set_muted(self, muted: bool) -> bool:
        if not self.enabled:
            return False
        self._init_backend()
        if self._backend == "pycaw" and self._volume is not None:
            try:
                self._volume.SetMute(1 if muted else 0, None)
                return True
            except Exception:
                return False
        if self._backend == "pactl":
            try:
                subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if muted else "0"],
                    check=False,
                )
                return True
            except Exception:
                return False
        return False

    def change(self, direction: int) -> float | None:
        if not self.enabled:
            return None
        step = self.step if direction >= 0 else -self.step
        self._init_backend()
        if self._backend == "pycaw" and self._volume is not None:
            try:
                cur = float(self._volume.GetMasterVolumeLevelScalar())
                tgt = max(0.0, min(1.0, cur + step))
                self._volume.SetMute(0, None)
                self._volume.SetMasterVolumeLevelScalar(tgt, None)
                return tgt
            except Exception:
                return None
        if self._backend == "pactl":
            try:
                pct = int(round(abs(step) * 100))
                signed = f"+{pct}%" if direction >= 0 else f"-{pct}%"
                subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
                    check=False,
                )
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", signed],
                    check=False,
                )
                return self._scalar()
            except Exception:
                return None
        return None


def play_wav(path: str, stop_flag: Event):
    try:
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(path, dtype="float32")
        sd.play(data, sr)
        while sd.get_stream().active:
            if stop_flag.is_set():
                sd.stop()
                return False
            time.sleep(0.05)
        return True
    except Exception:
        pass
    try:
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        return True
    except Exception:
        return False


def resolve_device(settings: Settings):
    device = settings.device
    if device == "" or device is None:
        return None
    if isinstance(device, str):
        stripped = device.strip()
        if stripped.isdigit():
            return int(stripped)

        # Allow substring matching (e.g. "UMIK-1") to resolve to an actual PortAudio device index.
        try:
            import sounddevice as sd

            needle = stripped.lower()
            devices = sd.query_devices()
            candidates: list[tuple[int, dict]] = []
            for idx, info in enumerate(devices):
                name = str(info.get("name", "")).lower()
                max_in = int(info.get("max_input_channels") or 0)
                if max_in > 0 and needle and needle in name:
                    candidates.append((idx, info))

            if candidates:
                # On Windows, prefer WASAPI/WDM-KS devices over MME/DirectSound for stability.
                pref = {
                    "windows wasapi": 0,
                    "windows wdm-ks": 1,
                    "windows directsound": 2,
                    "mme": 3,
                }
                try:
                    hostapis = sd.query_hostapis()
                except Exception:
                    hostapis = ()

                def score(item: tuple[int, dict]) -> tuple[int, int]:
                    idx, info = item
                    hostapi_idx = int(info.get("hostapi") or 0)
                    host_name = ""
                    try:
                        if hostapis and 0 <= hostapi_idx < len(hostapis):
                            host_name = str(hostapis[hostapi_idx].get("name", "")).lower()
                    except Exception:
                        host_name = ""
                    host_rank = pref.get(host_name, 9)
                    # Prefer devices with higher input channels as tie-breaker.
                    max_in = int(info.get("max_input_channels") or 0)
                    return (host_rank, -max_in)

                candidates.sort(key=score)
                return candidates[0][0]
        except Exception:
            pass

        return stripped
    return device


def resolve_sample_rate(settings: Settings, device):
    if device is None:
        return settings.sample_rate
    try:
        import sounddevice as sd
        info = sd.query_devices(device)
        default_sr = info.get("default_samplerate")
        if default_sr:
            return int(default_sr)
    except Exception:
        pass
    return settings.sample_rate


def resolve_input_channels(settings: Settings, device) -> int:
    """
    Returns the number of input channels to request from sounddevice.
    0 means the selected device has no input channels.
    """
    if device is None:
        return 1
    try:
        import sounddevice as sd

        info = sd.query_devices(device)
        max_in = int(info.get("max_input_channels") or 0)
        if max_in <= 0:
            return 0
        # Prefer stereo capture when available so we can auto-pick the loudest channel
        # (some USB mics report multiple channels where one can be effectively silent).
        return 2 if max_in >= 2 else 1
    except Exception:
        return 1


def resolve_openwakeword_model_path(model_path: str) -> tuple[str, str]:
    """
    Returns (resolved_path, model_name).

    If model_path is empty, tries to pick a built-in OpenWakeWord model shipped
    with the `openwakeword` package (e.g. hey_jarvis.tflite).
    """
    if model_path:
        p = str(model_path).strip()
        if p:
            if os.path.isfile(p):
                return p, os.path.splitext(os.path.basename(p))[0]
            # If a bare filename was provided, try resolving against the default models dir.
            try:
                import openwakeword as _oww
                base = os.path.join(os.path.dirname(_oww.__file__), "resources", "models")
                candidate = os.path.join(base, p)
                if os.path.isfile(candidate):
                    return candidate, os.path.splitext(os.path.basename(candidate))[0]
            except Exception:
                pass

    try:
        import openwakeword as _oww

        base = os.path.join(os.path.dirname(_oww.__file__), "resources", "models")
        # Prefer common built-in wakewords. You can override with OWW_DEFAULT_MODEL.
        preferred = [
            os.getenv("OWW_DEFAULT_MODEL", "").strip(),
            "hey_jarvis.tflite",
            "hey_jarvis_v0.1.tflite",
            "alexa.tflite",
            "alexa_v0.1.tflite",
            "hey_mycroft.tflite",
            "hey_mycroft_v0.1.tflite",
            "hey_rhasspy.tflite",
            "hey_rhasspy_v0.1.tflite",
        ]
        for fname in preferred:
            if not fname:
                continue
            p = fname if os.path.isabs(fname) else os.path.join(base, fname)
            if os.path.isfile(p):
                return p, os.path.splitext(os.path.basename(p))[0]
    except Exception:
        pass

    return "", ""


class OpenWakeWordDetector:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.available = False
        self.model_name = None
        self.model_path = ""
        self.model = None
        self.threshold = float(threshold)

        resolved_path, resolved_name = resolve_openwakeword_model_path(model_path)
        if not resolved_path:
            return
        try:
            from openwakeword.model import Model
            self.model = Model(wakeword_models=[resolved_path])
            self.model_path = resolved_path
            self.model_name = resolved_name or os.path.splitext(os.path.basename(resolved_path))[0]
            self.available = True
        except Exception:
            self.available = False

    def score(self, audio) -> float:
        if not self.available or self.model is None:
            return 0.0
        try:
            import numpy as np

            x = np.asarray(audio)
            # openwakeword expects int16 PCM at 16k.
            if x.dtype != np.int16:
                # Assume float audio is normalized to [-1, 1].
                if np.issubdtype(x.dtype, np.floating):
                    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                    x = np.clip(x, -1.0, 1.0)
                    x = (x * 32767.0).astype(np.int16)
                else:
                    x = np.clip(x, -32768, 32767).astype(np.int16)

            pred = self.model.predict(x)
            if isinstance(pred, dict):
                if self.model_name and self.model_name in pred:
                    score = pred.get(self.model_name, 0.0)
                else:
                    # Some packaged models include versioned filenames (e.g. *_v0.1.tflite)
                    # but the prediction dict key may differ. For a single loaded wakeword,
                    # the safest fallback is "max score across keys".
                    try:
                        score = max(float(v) for v in pred.values())
                    except Exception:
                        score = 0.0
            else:
                score = 0.0

            return float(score) if score is not None else 0.0
        except Exception:
            return 0.0

    def detect(self, audio) -> bool:
        return self.score(audio) >= self.threshold


class NoiseReducer:
    """
    Spectral noise reduction for showroom use.

    During SLEEPING mode the reducer collects ambient audio (music, HVAC, crowd)
    to build a noise profile.  Once the profile is ready every incoming frame is
    cleaned with spectral subtraction before being fed to ASR, dramatically
    reducing false transcriptions from background sound.

    Requires:  pip install noisereduce
    Falls back silently if the library is absent.
    """

    # Number of 2048-sample frames to accumulate before the profile is ready.
    # At 16 kHz / 2048 block ≈ 128 ms per frame → 25 frames ≈ 3.2 seconds.
    _PROFILE_FRAMES = 25

    def __init__(self, settings: Settings, sample_rate: int = 16000):
        import logging
        self._log = logging.getLogger("voice-noise-reducer")
        self.enabled = settings.noise_reduction_enabled
        self.sample_rate = sample_rate
        self.prop_decrease = settings.noise_reduction_prop
        self._nr = None
        self._noise_profile = None
        self._profile_buf: list = []
        self.profile_ready = False

        if not self.enabled:
            return
        try:
            import noisereduce as nr
            self._nr = nr
            self._log.info("NoiseReducer ready — collecting ambient profile (%d frames)", self._PROFILE_FRAMES)
        except ImportError:
            self.enabled = False
            self._log.warning(
                "noisereduce not installed — noise reduction disabled. "
                "Run: pip install noisereduce"
            )

    def update_profile(self, audio_f32):
        """
        Feed a raw audio frame captured during SLEEPING.
        Once enough frames are collected the noise profile is locked.
        """
        if not self.enabled or self.profile_ready or self._nr is None:
            return
        self._profile_buf.append(audio_f32.copy())
        if len(self._profile_buf) >= self._PROFILE_FRAMES:
            try:
                import numpy as np
                self._noise_profile = np.concatenate(self._profile_buf)
                self.profile_ready = True
                self._profile_buf.clear()
                self._log.info(
                    "Noise profile captured (%.1f s, RMS %.4f)",
                    len(self._noise_profile) / self.sample_rate,
                    float(np.sqrt(np.mean(self._noise_profile ** 2))),
                )
            except Exception as exc:
                self._log.warning("Failed to build noise profile: %s", exc)
                self._profile_buf.clear()

    def reset_profile(self):
        """Re-collect the noise profile (call if the acoustic environment changes)."""
        self._profile_buf.clear()
        self._noise_profile = None
        self.profile_ready = False
        if self.enabled:
            self._log.info("Noise profile reset — recollecting…")

    def calibrate(self) -> float | None:
        """
        Measure residual RMS by running a chunk of the captured noise profile
        through the reducer itself.  Returns the recommended noise_gate_rms value
        (residual_rms × settings.noise_gate_multiplier).

        Call this once immediately after profile_ready becomes True.
        The result is the noise floor *after* reduction — anything below it is
        still ambient noise; anything above is likely speech.
        """
        if not self.enabled or not self.profile_ready or self._noise_profile is None:
            return None
        try:
            import numpy as np
            # Use the first 2048-sample chunk of the captured noise as a test frame
            test_chunk = self._noise_profile[:2048].astype(np.float32)
            residual = self.reduce(test_chunk)
            residual_rms = float(np.sqrt(np.mean(residual ** 2)))
            self._log.info(
                "Noise calibration: residual RMS after reduction = %.5f", residual_rms
            )
            return residual_rms
        except Exception as exc:
            self._log.warning("Calibration failed: %s", exc)
            return None

    def reduce(self, audio_f32):
        """
        Return a denoised copy of audio_f32.
        Falls back to the original frame on any error.
        """
        if not self.enabled or self._nr is None:
            return audio_f32
        try:
            import numpy as np
            if self.profile_ready and self._noise_profile is not None:
                cleaned = self._nr.reduce_noise(
                    y=audio_f32,
                    sr=self.sample_rate,
                    y_noise=self._noise_profile,
                    stationary=False,
                    prop_decrease=self.prop_decrease,
                )
            else:
                # Profile not ready yet — stationary subtraction on the chunk itself
                cleaned = self._nr.reduce_noise(
                    y=audio_f32,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.6,
                )
            return cleaned.astype(np.float32)
        except Exception:
            return audio_f32
