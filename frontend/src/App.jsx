import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import {
  CookingPot,
  Diamond,
  EyeSlash,
  FilmSlate,
  HandWaving,
  SpeakerX,
  UsersThree,
} from '@phosphor-icons/react';
import ListeningAnimation from './components/ListeningAnimation';
import VoiceWave from './components/VoiceWave';
import AudioSpectrum from './components/AudioSpectrum';
import soundEffects from './utils/soundEffects';

/* ═══════════════════════════════════════════════════════════════════════════
   Constants & Configuration
   ═══════════════════════════════════════════════════════════════════════════ */

const SLIDER_MAX = 82.5;
const IS_DEV = import.meta.env.DEV;
const IS_LOCALHOST = ['localhost', '127.0.0.1'].includes(window.location.hostname);
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws';
const WS_ORIGIN = `${WS_PROTOCOL}://${window.location.host}`;
const HTTP_ORIGIN = window.location.origin;
const VOICE_BASE_PATH =
  import.meta.env.VITE_VOICE_BASE_PATH ?? (IS_DEV ? '/voice' : '');

const VOICE_WS_URL =
  import.meta.env.VITE_VOICE_WS_URL || `${WS_ORIGIN}${VOICE_BASE_PATH}/ws`;
const VOICE_AUDIO_WS_URL =
  import.meta.env.VITE_VOICE_AUDIO_WS_URL || `${WS_ORIGIN}${VOICE_BASE_PATH}/audio`;
const VOICE_STATUS_URL =
  import.meta.env.VITE_VOICE_STATUS_URL || `${HTTP_ORIGIN}${VOICE_BASE_PATH}/status`;
const VOICE_MODELS_URL =
  import.meta.env.VITE_VOICE_MODELS_URL || `${HTTP_ORIGIN}${VOICE_BASE_PATH}/models`;
const VOICE_UI_SYNC_URL =
  import.meta.env.VITE_VOICE_UI_SYNC_URL || `${HTTP_ORIGIN}${VOICE_BASE_PATH}/ui_sync`;

const BACKEND_TTS = String(import.meta.env.VITE_BACKEND_TTS || 'false').toLowerCase() === 'true';
const BROWSER_TTS = String(import.meta.env.VITE_BROWSER_TTS || 'false').toLowerCase() === 'true';
const FORCE_BROWSER_TTS =
  String(import.meta.env.VITE_FORCE_BROWSER_TTS || 'false').toLowerCase() === 'true';

/* ═══════════════════════════════════════════════════════════════════════════
   Experience Definitions
   ═══════════════════════════════════════════════════════════════════════════ */

const experiences = [
  {
    id: 'welcome',
    title: 'Welcome Experience',
    icon: HandWaving,
    description: 'First impressions, perfectly calibrated',
    options: [
      { id: 'soft', label: 'Soft Welcome', faderLabel: 'Soft Welcome Volume' },
      { id: 'luxury', label: 'Luxury Ambient', faderLabel: 'Luxury Ambient Volume' },
    ],
  },
  {
    id: 'kitchen',
    title: 'Kitchen Experience',
    icon: CookingPot,
    description: 'Sound designed for culinary spaces',
    options: [
      { id: 'silent', label: 'Silent', faderLabel: 'Silent' },
      { id: 'ambience', label: 'Background Ambience', faderLabel: 'Background Ambience' },
      { id: 'entertaining', label: 'Entertaining Mode', faderLabel: 'Entertaining Mode' },
    ],
  },
  {
    id: 'invisible',
    title: 'Invisible Experience',
    icon: EyeSlash,
    description: 'Sound that disappears into architecture',
    options: [
      { id: 'silent2', label: 'Silent', faderLabel: 'Silent' },
      { id: 'ambience2', label: 'Background Ambience', faderLabel: 'Background Ambience' },
      { id: 'entertaining2', label: 'Entertaining Mode', faderLabel: 'Entertaining Mode' },
    ],
  },
  {
    id: 'cinema',
    title: 'Cinema Experience',
    icon: FilmSlate,
    description: 'Immersive theatrical soundscapes',
    options: [
      { id: 'invisible', label: 'Invisible Cinema', faderLabel: 'Invisible Cinema' },
      { id: 'live', label: 'Live Performance', faderLabel: 'Live Performance' },
      { id: 'luxury', label: 'Luxury Apartment Night', faderLabel: 'Luxury Apartment Night' },
    ],
  },
  {
    id: 'lounge',
    title: 'Lounge & Entertain',
    icon: UsersThree,
    description: 'Social spaces, sonically perfected',
    options: [
      { id: 'lounge', label: 'Background Lounge', faderLabel: 'Background Lounge' },
      { id: 'cocktail', label: 'Cocktail Evening', faderLabel: 'Cocktail Evening' },
      { id: 'design', label: 'Silent (Design Focus)', faderLabel: 'Silent (Design Focus)' },
    ],
  },
  {
    id: 'iconic',
    title: 'Iconic Design',
    icon: Diamond,
    description: 'Statement pieces in sound',
    options: [
      { id: 'lounge2', label: 'Background Lounge', faderLabel: 'Background Lounge' },
      { id: 'cocktail2', label: 'Cocktail Evening', faderLabel: 'Cocktail Evening' },
      { id: 'design2', label: 'Silent (Design Focus)', faderLabel: 'Silent (Design Focus)' },
    ],
  },
  {
    id: 'off',
    title: 'All Sound Off',
    icon: SpeakerX,
    description: 'Complete silence',
    options: [{ id: 'off', label: 'Confirm Mute', faderLabel: '' }],
  },
];

const EXPERIENCE_INDEX_MAP = {
  welcome: 0,
  kitchen: 1,
  invisible: 2,
  cinema: 3,
  lounge: 4,
  iconic: 5,
};

/* ═══════════════════════════════════════════════════════════════════════════
   Audio Utilities
   ═══════════════════════════════════════════════════════════════════════════ */

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
  if (outputSampleRate === inputSampleRate) return buffer;
  const ratio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const nextOffset = Math.round((i + 1) * ratio);
    const offset = Math.round(i * ratio);
    let accum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < buffer.length; j++) {
      accum += buffer[j];
      count++;
    }
    result[i] = count > 0 ? accum / count : 0;
  }
  return result;
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Text & Display Helpers
   ═══════════════════════════════════════════════════════════════════════════ */

function normalize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
}

function humanizeWakeModelName(name) {
  if (!name) return '';
  const parts = String(name).split('_').filter(Boolean);
  const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
  if (parts[0] === 'hey' && parts.length >= 2) return `Hey ${parts.slice(1).map(cap).join(' ')}`;
  return parts.map(cap).join(' ');
}

function shouldBargeIn(text) {
  return /stop|all off|sound off|mute/.test(text);
}

function fmtTime(ts) {
  if (!ts) return '--';
  return new Date(ts * 1000).toLocaleTimeString();
}

function pickFemaleVoice(voices) {
  const preferred = voices.filter((v) => /female|woman|girl/i.test(v.name + ' ' + v.voiceURI));
  if (preferred.length > 0) return preferred[0];
  const en = voices.filter((v) => (v.lang || '').toLowerCase().startsWith('en'));
  return en[0] || voices[0] || null;
}

function volumeKey(modalId, optionId) {
  return `${modalId}:${optionId}`;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Mode Display Config
   ═══════════════════════════════════════════════════════════════════════════ */

const MODE_CONFIG = {
  LISTENING: {
    label: 'Listening',
    dot: 'bg-amber-400 shadow-[0_0_12px_rgba(198,168,109,0.8)]',
    border: 'border-amber-400/50',
    text: 'text-amber-600',
    pulse: true,
  },
  EXECUTING: {
    label: 'Executing',
    dot: 'bg-sky-400 shadow-[0_0_12px_rgba(56,189,248,0.6)]',
    border: 'border-sky-400/40',
    text: 'text-sky-600',
    pulse: false,
  },
  COOLDOWN: {
    label: 'Cooldown',
    dot: 'bg-rose-400/70',
    border: 'border-rose-300/30',
    text: 'text-rose-500',
    pulse: false,
  },
  SLEEPING: {
    label: 'Sleeping',
    dot: 'bg-slate-400/50',
    border: 'border-slate-300/30',
    text: 'text-slate-500',
    pulse: false,
  },
};

/* ═══════════════════════════════════════════════════════════════════════════
   Main App Component
   ═══════════════════════════════════════════════════════════════════════════ */

export default function App() {
  // ── UI State ──────────────────────────────────────────────────────────────
  const [activeModal, setActiveModal] = useState(null);
  const [activeOptionByModal, setActiveOptionByModal] = useState({});
  const [volumes, setVolumes] = useState({});
  const [actionLabel, setActionLabel] = useState('');
  const [introDone, setIntroDone] = useState(false);

  // ── Voice State ───────────────────────────────────────────────────────────
  const [voiceStatus, setVoiceStatus] = useState('connecting');
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [voiceMode, setVoiceMode] = useState('SLEEPING');
  const [voiceDebug, setVoiceDebug] = useState('');
  const [voiceLevel, setVoiceLevel] = useState(0);

  // ── Mic State ─────────────────────────────────────────────────────────────
  const [micStreamStatus, setMicStreamStatus] = useState('idle');
  const [micStreamError, setMicStreamError] = useState('');

  // ── Backend Data ──────────────────────────────────────────────────────────
  const [statusData, setStatusData] = useState(null);
  const [modelsData, setModelsData] = useState(null);

  // ── Refs (stable across renders) ──────────────────────────────────────────
  const lastExperienceRef = useRef(null);
  const lastIntentSeqRef = useRef(0);
  const hasSeenVoiceStateRef = useRef(false);
  const lastVoiceAtRef = useRef(0);
  const voiceListRef = useRef([]);
  const alwaysListeningRef = useRef(false);
  const applyIntentRef = useRef(null);
  const activeModalRef = useRef(null);
  const activeOptionByModalRef = useRef({});
  const volumesRef = useRef({});

  // ── Browser Mic Refs ──────────────────────────────────────────────────────
  const audioWsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const audioProcessorRef = useRef(null);
  const audioStreamRef = useRef(null);

  /* ─────────────────────────────────────────────────────────────────────────
     Derived Values
     ───────────────────────────────────────────────────────────────────────── */

  const wakeHint = useMemo(() => {
    const oww = modelsData?.models?.openwakeword_model;
    if (oww?.exists && oww?.name) {
      const hint = humanizeWakeModelName(oww.name);
      if (hint) return hint;
    }
    return modelsData?.settings?.wake_phrase || modelsData?.settings?.wake_word || 'wake word';
  }, [modelsData]);

  const activeExperience = useMemo(
    () => experiences.find((exp) => exp.id === activeModal),
    [activeModal],
  );

  const backendAudioSource = modelsData?.settings?.audio_source || 'browser';
  const metrics = statusData?.metrics || {};
  const session = statusData?.session || {};
  const modeConfig = MODE_CONFIG[voiceMode] || MODE_CONFIG.SLEEPING;

  const shouldUseBrowserTts = useMemo(() => {
    if (!window.speechSynthesis) return false;
    if (FORCE_BROWSER_TTS) return true;
    if (!BACKEND_TTS) return BROWSER_TTS;

    const backendTtsEnabled = modelsData?.settings?.tts_enabled;
    const piperExeOk = modelsData?.models?.piper_exe?.exists;
    const piperModelOk = modelsData?.models?.piper_model?.exists;
    const backendTtsLooksOk =
      backendTtsEnabled !== false && piperExeOk !== false && piperModelOk !== false;

    const backendTtsErrors = Number(metrics.tts_errors || 0);
    if (backendTtsErrors > 0) return true;
    if (!backendTtsLooksOk) return true;

    return false;
  }, [modelsData, metrics.tts_errors]);

  const rms =
    typeof session.voice_rms_norm === 'number'
      ? session.voice_rms_norm
      : typeof session.voice_rms_raw === 'number'
        ? session.voice_rms_raw
        : typeof session.voice_rms === 'number'
          ? session.voice_rms
          : null;
  const rmsPct = rms !== null ? Math.min(100, Math.max(0, Math.round(rms * 200))) : null;
  const displayLevel = rmsPct !== null ? rmsPct / 100 : voiceLevel;

  useEffect(() => {
    activeModalRef.current = activeModal;
  }, [activeModal]);

  useEffect(() => {
    activeOptionByModalRef.current = activeOptionByModal;
  }, [activeOptionByModal]);

  useEffect(() => {
    volumesRef.current = volumes;
  }, [volumes]);

  /* ─────────────────────────────────────────────────────────────────────────
     TTS (Browser Speech Synthesis)
     ───────────────────────────────────────────────────────────────────────── */

  const speak = useCallback((text) => {
    if (!shouldUseBrowserTts || !window.speechSynthesis) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = 'en-US';
    utter.rate = 1.0;
    utter.pitch = 1.05;
    const voices = voiceListRef.current || window.speechSynthesis.getVoices();
    const pick = pickFemaleVoice(voices);
    if (pick) utter.voice = pick;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  }, [shouldUseBrowserTts]);

  /* ─────────────────────────────────────────────────────────────────────────
     Experience / Option Helpers
     ───────────────────────────────────────────────────────────────────────── */

  const syncUiStateToBackend = useCallback(async (partial) => {
    try {
      await fetch(VOICE_UI_SYNC_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(partial),
      });
    } catch (_) {
      // Best effort sync only.
    }
  }, []);

  const openModal = useCallback((id) => {
    setActiveModal(id);
    lastExperienceRef.current = id;
    const exp = experiences.find((e) => e.id === id);
    const defaultOpt = exp?.options?.[0]?.id;
    if (defaultOpt) {
      setActiveOptionByModal((prev) => ({ ...prev, [id]: defaultOpt }));
    }
    syncUiStateToBackend({
      selected_experience: id,
      selected_option: defaultOpt || null,
    });
  }, [syncUiStateToBackend]);

  const closeModal = useCallback(() => {
    setActiveModal(null);
    syncUiStateToBackend({
      selected_experience: null,
      selected_option: null,
    });
  }, [syncUiStateToBackend]);

  const activateOption = useCallback((modalId, optionId) => {
    setActiveOptionByModal((prev) => ({ ...prev, [modalId]: optionId }));
    syncUiStateToBackend({
      selected_experience: modalId,
      selected_option: optionId,
    });
  }, [syncUiStateToBackend]);

  const updateVolume = useCallback((modalId, optionId, value) => {
    setVolumes((prev) => ({ ...prev, [volumeKey(modalId, optionId)]: value }));
  }, []);

  const getDefaultOption = useCallback((modalId) => {
    const exp = experiences.find((e) => e.id === modalId);
    return exp?.options?.[0]?.id ?? null;
  }, []);

  const moveOption = useCallback((modalId, direction) => {
    const exp = experiences.find((e) => e.id === modalId);
    if (!exp || exp.options.length === 0) return;
    const currentMap = activeOptionByModalRef.current || {};
    const currentId = currentMap[modalId] || exp.options[0].id;
    const idx = exp.options.findIndex((o) => o.id === currentId);
    const next =
      direction === 'next'
        ? (idx + 1) % exp.options.length
        : (idx - 1 + exp.options.length) % exp.options.length;
    activateOption(modalId, exp.options[next].id);
  }, [activateOption]);

  const adjustActiveVolume = useCallback((delta) => {
    const modalId = activeModalRef.current;
    if (!modalId) return;
    const exp = experiences.find((e) => e.id === modalId);
    if (!exp || exp.options.length === 0) return;
    const currentMap = activeOptionByModalRef.current || {};
    const currentId = currentMap[modalId] || exp.options[0].id;
    const volMap = volumesRef.current || {};
    const current = volMap[volumeKey(modalId, currentId)] || 0;
    updateVolume(modalId, currentId, Math.max(0, Math.min(SLIDER_MAX, current + delta)));
  }, [updateVolume]);

  const performOpenByIndex = useCallback(
    (index) => {
      if (!experiences[index]) return false;
      const modalId = experiences[index].id;
      setActiveModal(modalId);
      lastExperienceRef.current = modalId;
      const def = getDefaultOption(modalId);
      if (def) activateOption(modalId, def);
      return true;
    },
    [getDefaultOption, activateOption],
  );

  const performSelectOption = useCallback(
    (optIndex) => {
      const modalId = activeModalRef.current;
      if (!modalId) return false;
      const exp = experiences.find((e) => e.id === modalId);
      if (!exp || !exp.options[optIndex]) return false;
      activateOption(modalId, exp.options[optIndex].id);
      return true;
    },
    [activateOption],
  );

  /* ─────────────────────────────────────────────────────────────────────────
     Intent Handler — called when backend sends a new intent_seq
     ───────────────────────────────────────────────────────────────────────── */

  const applyIntent = useCallback(
    (intent) => {
      if (!intent) return;

      // ── Error ──
      if (intent === 'unknown') {
        soundEffects.playError();
        return;
      }

      // ── Navigation ──
      if (intent === 'bye' || intent === 'back') {
        closeModal();
        soundEffects.playClick();
        return;
      }
      const currentModal = activeModalRef.current;
      if (intent === 'next' && currentModal) {
        moveOption(currentModal, 'next');
        soundEffects.playClick();
        return;
      }
      if (intent === 'previous' && currentModal) {
        moveOption(currentModal, 'prev');
        soundEffects.playClick();
        return;
      }

      // ── Mute / Sound Off ──
      if (intent === 'mute' || intent === 'all_sound_off') {
        const off = experiences.find((e) => e.id === 'off');
        if (off) {
          openModal(off.id);
          activateOption(off.id, off.options[0].id);
        }
        soundEffects.playClick();
        return;
      }

      // ── Volume ──
      if (intent === 'volume_up' || intent === 'volume_down') {
        const delta = intent === 'volume_up' ? 4.0 : -4.0;
        if (!activeModalRef.current) {
          const fallback = lastExperienceRef.current || experiences[0].id;
          openModal(fallback);
          const def = getDefaultOption(fallback);
          if (def) {
            activateOption(fallback, def);
            const current = (volumesRef.current || {})[volumeKey(fallback, def)] || 0;
            updateVolume(fallback, def, Math.max(0, Math.min(SLIDER_MAX, current + delta)));
          }
        } else {
          adjustActiveVolume(delta);
        }
        soundEffects.playClick();
        return;
      }

      // ── Option by label (experience-aware) ──
      if (intent.startsWith('opt_')) {
        const ensureModal = (modalId) => {
          if (activeModalRef.current !== modalId) openModal(modalId);
        };

        const chooseOption = (modalId, optionId) => {
          ensureModal(modalId);
          activateOption(modalId, optionId);
        };

        const within = activeModalRef.current || lastExperienceRef.current || null;

        if (intent === 'opt_soft_welcome') {
          chooseOption('welcome', 'soft');
          soundEffects.playSuccess();
          return;
        }
        if (intent === 'opt_luxury_ambient') {
          chooseOption('welcome', 'luxury');
          soundEffects.playSuccess();
          return;
        }

        if (intent === 'opt_invisible_cinema') {
          chooseOption('cinema', 'invisible');
          soundEffects.playSuccess();
          return;
        }
        if (intent === 'opt_live_performance') {
          chooseOption('cinema', 'live');
          soundEffects.playSuccess();
          return;
        }
        if (intent === 'opt_luxury_apartment_night') {
          chooseOption('cinema', 'luxury');
          soundEffects.playSuccess();
          return;
        }

        if (intent === 'opt_background_lounge') {
          if (within === 'iconic') chooseOption('iconic', 'lounge2');
          else chooseOption('lounge', 'lounge');
          soundEffects.playSuccess();
          return;
        }
        if (intent === 'opt_cocktail_evening') {
          if (within === 'iconic') chooseOption('iconic', 'cocktail2');
          else chooseOption('lounge', 'cocktail');
          soundEffects.playSuccess();
          return;
        }
        if (intent === 'opt_design_focus') {
          if (within === 'iconic') chooseOption('iconic', 'design2');
          else chooseOption('lounge', 'design');
          soundEffects.playSuccess();
          return;
        }

        if (intent === 'opt_background_ambience') {
          if (within === 'invisible') chooseOption('invisible', 'ambience2');
          else chooseOption('kitchen', 'ambience');
          soundEffects.playSuccess();
          return;
        }
        if (intent === 'opt_entertaining_mode') {
          if (within === 'invisible') chooseOption('invisible', 'entertaining2');
          else chooseOption('kitchen', 'entertaining');
          soundEffects.playSuccess();
          return;
        }
        if (intent === 'opt_silent') {
          if (within === 'invisible') chooseOption('invisible', 'silent2');
          else if (within === 'lounge') chooseOption('lounge', 'design');
          else if (within === 'iconic') chooseOption('iconic', 'design2');
          else chooseOption('kitchen', 'silent');
          soundEffects.playSuccess();
          return;
        }

        // Unknown opt_* — ignore gracefully
        return;
      }

      // ── Named Experience (welcome, kitchen, cinema, etc.) ──
      if (intent in EXPERIENCE_INDEX_MAP) {
        performOpenByIndex(EXPERIENCE_INDEX_MAP[intent]);
        soundEffects.playWhoosh();
        return;
      }

      // ── Open by number (open_1 → open_7) ──
      if (intent.startsWith('open_')) {
        const num = Number(intent.replace('open_', ''));
        if (!Number.isNaN(num)) {
          performOpenByIndex(num - 1);
          soundEffects.playWhoosh();
        }
        return;
      }

      // ── Select option by number (option_1 → option_7) ──
      if (intent.startsWith('option_')) {
        const num = Number(intent.replace('option_', ''));
        if (!Number.isNaN(num)) {
          if (!activeModalRef.current) {
            const fallback = lastExperienceRef.current || experiences[0].id;
            openModal(fallback);
            const exp = experiences.find((e) => e.id === fallback);
            if (exp?.options[num - 1]) {
              activateOption(fallback, exp.options[num - 1].id);
              soundEffects.playSuccess();
            }
            return;
          }
          performSelectOption(num - 1);
          soundEffects.playSuccess();
        }
        return;
      }

      // ── Greet, Help, Bye — no UI action needed, TTS handles it ──
    },
    [
      closeModal,
      moveOption,
      openModal,
      activateOption,
      adjustActiveVolume,
      updateVolume,
      getDefaultOption,
      performOpenByIndex,
      performSelectOption,
    ],
  );

  // Keep refs current for the WS callback (which has [] deps)
  applyIntentRef.current = applyIntent;

  /* ─────────────────────────────────────────────────────────────────────────
     Browser Microphone Streaming
     ───────────────────────────────────────────────────────────────────────── */

  const stopBrowserMic = useCallback((nextStatus = 'idle') => {
    if (audioProcessorRef.current) {
      audioProcessorRef.current.disconnect();
      audioProcessorRef.current.onaudioprocess = null;
      audioProcessorRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((t) => t.stop());
      audioStreamRef.current = null;
    }
    if (audioWsRef.current) {
      audioWsRef.current.close();
      audioWsRef.current = null;
    }
    setMicStreamStatus(nextStatus);
  }, []);

  const startBrowserMic = useCallback(async () => {
    if (modelsData?.settings?.audio_source === 'device') {
      setMicStreamError('Backend uses AUDIO_SOURCE=device — browser mic disabled.');
      setMicStreamStatus('disabled');
      return;
    }
    if (audioWsRef.current?.readyState <= 1) return;
    if (audioStreamRef.current) return;

    setMicStreamError('');
    setMicStreamStatus('connecting');

    // Create AudioContext synchronously (iOS requires a direct user interaction)
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();
    audioCtxRef.current = audioCtx;
    audioCtx.resume();

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setMicStreamError('Microphone not supported in this browser.');
        stopBrowserMic('error');
        return;
      }
      if (!window.isSecureContext && !IS_LOCALHOST) {
        setMicStreamError('Microphone blocked: requires HTTPS or localhost.');
        stopBrowserMic('blocked');
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      audioStreamRef.current = stream;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      const gain = audioCtx.createGain();
      gain.gain.value = 0; // Don't play mic audio back through speakers

      const connectAudioWs = () => {
        const ws = new WebSocket(VOICE_AUDIO_WS_URL);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => setMicStreamStatus('streaming');
        ws.onclose = () => {
          setMicStreamStatus('idle');
          if (audioStreamRef.current && audioWsRef.current === ws) {
            audioWsRef.current = null;
            setTimeout(() => {
              if (audioStreamRef.current) {
                const reconnect = new WebSocket(VOICE_AUDIO_WS_URL);
                reconnect.binaryType = 'arraybuffer';
                reconnect.onopen = () => setMicStreamStatus('streaming');
                reconnect.onclose = () => setMicStreamStatus('idle');
                reconnect.onerror = () => setMicStreamStatus('error');
                audioWsRef.current = reconnect;
              }
            }, 1500);
          }
        };
        ws.onerror = () => setMicStreamStatus('error');
        return ws;
      };

      const ws = connectAudioWs();

      processor.onaudioprocess = (event) => {
        const active = audioWsRef.current;
        if (!active || active.readyState !== 1) return;
        const input = event.inputBuffer.getChannelData(0);
        const downsampled = downsampleBuffer(input, audioCtx.sampleRate, 16000);
        active.send(floatTo16BitPCM(downsampled));
      };

      source.connect(processor);
      processor.connect(gain);
      gain.connect(audioCtx.destination);

      audioWsRef.current = ws;
      audioProcessorRef.current = processor;
    } catch (err) {
      setMicStreamError(`${err?.name ? err.name + ': ' : ''}${err?.message || String(err)}`);
      stopBrowserMic('error');
    }
  }, [modelsData, stopBrowserMic]);

  /* ─────────────────────────────────────────────────────────────────────────
     Intro / Start Experience
     ───────────────────────────────────────────────────────────────────────── */

  const startExperience = useCallback(() => {
    speak(
      `Welcome to our showroom. Say "${wakeHint}" to begin, then tell me what experience you'd like.`,
    );
    setIntroDone(true);
  }, [speak, wakeHint]);

  /* ─────────────────────────────────────────────────────────────────────────
     Effects
     ───────────────────────────────────────────────────────────────────────── */

  // Load browser TTS voices
  useEffect(() => {
    const update = () => {
      voiceListRef.current = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
    };
    update();
    if (window.speechSynthesis) window.speechSynthesis.onvoiceschanged = update;
    return () => {
      if (window.speechSynthesis) window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  // Cleanup browser mic on unmount
  useEffect(() => () => stopBrowserMic(), [stopBrowserMic]);

  // ── Voice WebSocket ───────────────────────────────────────────────────────
  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    let alive = true;

    const connect = () => {
      if (!alive) return;
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }

      try { ws = new WebSocket(VOICE_WS_URL); } catch (_) {
        reconnectTimer = setTimeout(connect, 2000);
        return;
      }

      setVoiceStatus('connecting');
      ws.addEventListener('open', () => setVoiceStatus('connected'));
      ws.addEventListener('close', () => {
        setVoiceStatus('disconnected');
        if (alive) reconnectTimer = setTimeout(connect, 1500);
      });
      ws.addEventListener('error', () => {}); // close fires next

      ws.addEventListener('message', (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (!payload) return;

          /* ── State update from backend ─────────────────────────────── */
          if (payload.type === 'state_update' && payload.state) {
            const st = payload.state;
            setVoiceMode(st.mode || 'SLEEPING');
            setVoiceTranscript(st.transcript_partial || st.last_final || '');
            setVoiceDebug(st.debug || '');
            alwaysListeningRef.current = Boolean(st.always_listening);

            if (typeof st.voice_rms === 'number') {
              setVoiceLevel(Math.min(1, st.voice_rms * 2));
              if (st.voice_rms > 0.005) lastVoiceAtRef.current = Date.now();
            }
            if (st.last_response) setActionLabel(st.last_response);

            // First state_update: just sync the seq counter, don't act.
            if (!hasSeenVoiceStateRef.current) {
              hasSeenVoiceStateRef.current = true;
              lastIntentSeqRef.current = st.intent_seq || 0;
              return;
            }

            // New intent detected
            if (st.intent_seq && st.intent_seq !== lastIntentSeqRef.current) {
              lastIntentSeqRef.current = st.intent_seq;

              // Apply intent to UI
              applyIntentRef.current?.(st.last_intent);

              // ★ FIX: Speak ALL intents including greet
              // The old code had `st.last_intent !== 'greet'` which blocked
              // the greeting response from ever being spoken.
              if (shouldUseBrowserTts && st.last_response) {
                speak(st.last_response);
              }
            }
            return;
          }

          /* ── Voice events (wake, partial, final) ───────────────────── */
          if (payload.source === 'voice') {
            if (payload.event === 'wake') {
              soundEffects.playWake();
              window.speechSynthesis?.cancel();

              // ★ FIX: Speak greeting in ALL modes including always_listening
              if (shouldUseBrowserTts) {
                speak('How may I help you?');
              }
              return;
            }

            if (payload.text) {
              const spoken = normalize(payload.text);
              lastVoiceAtRef.current = Date.now();
              setVoiceLevel(1);
              if (shouldBargeIn(spoken)) window.speechSynthesis?.cancel();
            }
            return;
          }
        } catch (_) { /* malformed message */ }
      });
    };

    connect();
    return () => {
      alive = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      try { ws?.close(); } catch (_) {}
    };
  }, []); // stable — uses refs for current callbacks

  // Decay voice level indicator
  useEffect(() => {
    const timer = setInterval(() => {
      if (Date.now() - lastVoiceAtRef.current > 1500) setVoiceLevel(0);
    }, 100);
    return () => clearInterval(timer);
  }, []);

  // Poll /status every 2s
  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const res = await fetch(VOICE_STATUS_URL, { cache: 'no-store' });
        if (res.ok && mounted) setStatusData(await res.json());
      } catch (_) {}
    };
    tick();
    const timer = setInterval(tick, 2000);
    return () => { mounted = false; clearInterval(timer); };
  }, []);

  // Poll /models every 10s
  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const res = await fetch(VOICE_MODELS_URL, { cache: 'no-store' });
        if (res.ok && mounted) setModelsData(await res.json());
      } catch (_) {}
    };
    tick();
    const timer = setInterval(tick, 10000);
    return () => { mounted = false; clearInterval(timer); };
  }, []);

  /* ─────────────────────────────────────────────────────────────────────────
     Render
     ───────────────────────────────────────────────────────────────────────── */

  return (
    <div className="relative min-h-screen overflow-x-hidden font-display text-[#1e1f22]">
      {/* Background layers */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(900px_520px_at_80%_12%,rgba(198,168,109,0.18),transparent_60%),radial-gradient(900px_520px_at_18%_78%,rgba(140,150,170,0.16),transparent_60%),linear-gradient(180deg,#f2efe9_0%,#f7f4ee_60%,#ffffff_100%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(0,0,0,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(0,0,0,0.04)_1px,transparent_1px)] bg-[size:46px_46px] opacity-20" />
      <div className="pointer-events-none absolute -inset-32 bg-[radial-gradient(circle_at_50%_40%,transparent_30%,rgba(0,0,0,0.2)_75%)]" />

      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-[1600px] flex-col gap-6 px-4 py-6 md:px-8 lg:px-12">
        {/* Listening animation overlay */}
        <ListeningAnimation isListening={voiceMode === 'LISTENING'} voiceLevel={voiceLevel} />

        {/* ── Intro Overlay ──────────────────────────────────────────────── */}
        {!introDone && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/25 backdrop-blur-sm">
            <div className="w-[min(92vw,420px)] rounded-[28px] border border-[#e3ddd2] bg-white/95 p-8 text-center shadow-[0_24px_50px_rgba(20,18,14,0.12)]">
              <div className="text-[11px] uppercase tracking-[0.4em] text-[#7b8088]">
                CIAO CUCINE
              </div>
              <h2 className="mt-4 text-2xl font-semibold text-[#1e1f22]">
                Tap to Begin the Audio Experience
              </h2>
              <p className="mt-3 text-sm text-[#5b6068]">
                Say "<span className="font-semibold text-[#c6a86d]">{wakeHint}</span>" to activate,
                then speak your command.
              </p>
              <button
                type="button"
                className="mt-6 w-full rounded-full border border-[#c6a86d] bg-[#c6a86d] px-5 py-3 text-sm font-semibold uppercase tracking-[0.2em] text-white shadow-[0_12px_30px_rgba(198,168,109,0.35)] transition-transform active:scale-[0.97]"
                onClick={startExperience}
              >
                Start
              </button>
            </div>
          </div>
        )}

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <header className="flex flex-wrap items-start justify-between gap-6">
          <div>
            <div className="text-[11px] uppercase tracking-[0.4em] text-[#7b8088]">CIAO CUCINE</div>
            <div className="mt-2 text-2xl font-semibold tracking-[0.02em] md:text-3xl">
              Audio Experience Suite
            </div>
            <div className="mt-1 text-sm text-[#5b6068]">
              Architected soundscapes for modern living
            </div>
          </div>

          {/* Status badges */}
          <div className="flex flex-wrap justify-end gap-2">
            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[11px] uppercase tracking-[0.2em] bg-white/70 ${modeConfig.border} ${modeConfig.text}`}
            >
              <span
                className={`h-2 w-2 rounded-full ${modeConfig.dot} ${modeConfig.pulse ? 'animate-pulse' : ''}`}
              />
              {modeConfig.label}
            </div>
            <StatusBadge label={`Mic ${micStreamStatus}`} />
            <StatusBadge label={`Voice ${voiceStatus}`} />
          </div>
        </header>

        {/* ── Main Grid ──────────────────────────────────────────────────── */}
        <main className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-[minmax(320px,1.4fr)_minmax(280px,0.9fr)] xl:gap-8">
          {/* ── Left: Experience List ───────────────────────────────────── */}
          <section className="rounded-[26px] border border-[#e3ddd2] bg-white/90 p-5 shadow-[0_24px_50px_rgba(20,18,14,0.12)] backdrop-blur sm:p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-[11px] uppercase tracking-[0.28em] text-[#7b8088]">
                  SHOWROOM
                </div>
                <h2 className="mt-2 text-2xl font-semibold">The Art of Sound</h2>
              </div>
              <div className="text-xs text-[#5b6068]">Pick an experience or say it aloud</div>
            </div>

            <ul className="mt-6 space-y-3">
              {experiences.map((exp) => {
                const isActive = activeModal === exp.id;
                const ExperienceIcon = exp.icon;
                return (
                  <li
                    key={exp.id}
                    className={`group cursor-pointer rounded-2xl border px-5 py-4 transition-all duration-200 ${
                      isActive
                        ? 'border-[#c6a86d]/60 bg-[#fffaf0] shadow-[0_8px_24px_rgba(198,168,109,0.15)]'
                        : 'border-transparent bg-white hover:border-[#c6a86d]/40 hover:shadow-[0_16px_30px_rgba(20,18,14,0.08)]'
                    }`}
                    onClick={() => openModal(exp.id)}
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={`inline-flex h-9 w-9 items-center justify-center rounded-full border transition-colors ${
                          isActive
                            ? 'border-[#c6a86d]/50 bg-[#f7efde] text-[#c6a86d]'
                            : 'border-[#e3ddd2] bg-white text-[#9b9fa6] group-hover:border-[#c6a86d]/40 group-hover:text-[#c6a86d]/70'
                        }`}
                      >
                        <ExperienceIcon size={18} weight={isActive ? 'fill' : 'regular'} />
                      </span>
                      <div className="flex-1">
                        <div className="text-lg font-semibold text-[#1e1f22]">{exp.title}</div>
                        <div className="mt-0.5 text-xs text-[#9b9fa6]">{exp.description}</div>
                      </div>
                    </div>
                    <div
                      className={`mt-3 h-0.5 w-10 rounded-full transition-colors ${isActive ? 'bg-[#c6a86d]' : 'bg-[#e3ddd2] group-hover:bg-[#c6a86d]/60'}`}
                    />
                  </li>
                );
              })}
            </ul>
          </section>

          {/* ── Right: Sidebar ─────────────────────────────────────────── */}
          <aside className="flex flex-col gap-4 md:gap-5">
            {/* Voice Console */}
            <Card title="VOICE CONSOLE" heading="Command Center" aside={`Say "${wakeHint}"`}>
              <div className="space-y-4">
                <Row label="Mode">
                  <span className={`font-semibold ${modeConfig.text}`}>{modeConfig.label}</span>
                </Row>
                <Row label="Last Response">
                  <span className="font-semibold text-[#1e1f22]">
                    {actionLabel || 'Awaiting your command.'}
                  </span>
                </Row>
                <Row label="Heard">
                  <span className="text-[#5b6068]">{voiceTranscript || '--'}</span>
                  <span className="ml-2 text-xs text-[#9b9fa6]">
                    {rmsPct !== null
                      ? `Mic ${rmsPct}%`
                      : voiceLevel > 0
                        ? `Mic ${Math.round(voiceLevel * 100)}%`
                        : 'Mic idle'}
                  </span>
                </Row>
                <VoiceWave
                  isActive={voiceMode === 'LISTENING' || voiceLevel > 0.1}
                  amplitude={displayLevel}
                />
                {voiceDebug && (
                  <Row label="Debug">
                    <span className="text-[#9b5a5a]">{voiceDebug}</span>
                  </Row>
                )}
              </div>
            </Card>

            {/* Quick Commands */}
            <SmallCard title="Try saying">
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs uppercase tracking-[0.18em] text-[#5b6068]">
                {['Open cinema', 'Lounge', 'Option two', 'Volume up', 'Mute'].map((cmd) => (
                  <span
                    key={cmd}
                    className="rounded-full border border-[#e3ddd2] px-3 py-1 text-center"
                  >
                    {cmd}
                  </span>
                ))}
              </div>
            </SmallCard>

            {/* Voice Input Visualizer */}
            <SmallCard title="Voice Input">
              <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[#efe8dd]">
                <div
                  className="h-2 rounded-full bg-[#c6a86d] transition-[width] duration-150"
                  style={{ width: `${rmsPct ?? Math.round(voiceLevel * 100)}%` }}
                />
              </div>
              <div className="mt-2 text-[11px] uppercase tracking-[0.18em] text-[#7b8088]">
                {displayLevel > 0 ? 'Mic activity detected' : 'Waiting for voice'}
              </div>

              <div className="mt-4">
                <AudioSpectrum
                  voiceLevel={displayLevel}
                  isActive={voiceMode === 'LISTENING' || voiceLevel > 0.1}
                />
              </div>

              {/* Mic toggle button */}
              <button
                type="button"
                disabled={backendAudioSource === 'device'}
                className={`mt-5 w-full rounded-full border px-4 py-2.5 text-xs font-semibold uppercase tracking-[0.2em] transition-all active:scale-[0.97] ${
                  backendAudioSource === 'device'
                    ? 'cursor-not-allowed border-[#e3ddd2] bg-white/40 text-[#9aa0a8]'
                    : micStreamStatus === 'streaming'
                      ? 'border-[#c6a86d] bg-[#c6a86d] text-white shadow-[0_4px_16px_rgba(198,168,109,0.3)]'
                      : 'border-[#e3ddd2] text-[#5b6068] hover:border-[#c6a86d]/50'
                }`}
                onClick={() =>
                  micStreamStatus === 'streaming' ? stopBrowserMic() : startBrowserMic()
                }
              >
                {backendAudioSource === 'device'
                  ? 'Using Device Mic'
                  : micStreamStatus === 'streaming'
                    ? 'Stop Browser Mic'
                    : 'Enable Browser Mic'}
              </button>

              {micStreamError && (
                <div className="mt-3 text-xs text-[#9b5a5a]">{micStreamError}</div>
              )}
            </SmallCard>

            {/* Model & Package Check */}
            <SmallCard title="Model Check">
              {modelsData ? (
                <>
                  <div className="mt-3 space-y-2">
                    {Object.entries(modelsData.models).map(([key, m]) => (
                      <div key={key} className="flex items-start gap-2">
                        <Dot ok={m.exists} />
                        <div className="min-w-0">
                          <div
                            className={`text-xs font-semibold ${m.exists ? 'text-[#1e1f22]' : 'text-red-500'}`}
                          >
                            {m.label}
                          </div>
                          <div className="truncate text-[10px] text-[#9b9fa6]" title={m.path}>
                            {m.path || '--'}
                          </div>
                          {!m.exists && m.reason && (
                            <div className="text-[10px] text-red-400">{m.reason}</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  <Divider />
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[#7b8088]">
                    Packages
                  </div>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {Object.entries(modelsData.packages).map(([key, p]) => (
                      <PkgBadge key={key} label={p.label} installed={p.installed} />
                    ))}
                  </div>

                  <Divider />
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[#7b8088]">
                    Runtime
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                    <RuntimeItem
                      on={modelsData.settings.tts_enabled}
                      label={`TTS ${modelsData.settings.tts_enabled ? 'on' : 'off'}`}
                    />
                    <RuntimeItem
                      on={modelsData.settings.duck_enabled}
                      label={`Duck ${modelsData.settings.duck_enabled ? 'on' : 'off'}`}
                    />
                    <RuntimeItem
                      on={modelsData.settings.noise_reduction}
                      label={`Noise ${modelsData.settings.noise_reduction ? 'on' : 'off'}`}
                    />
                    <RuntimeItem
                      on={modelsData.runtime.noise_profile_ready}
                      amber={!modelsData.runtime.noise_profile_ready}
                      label={`Profile ${modelsData.runtime.noise_profile_ready ? 'ready' : 'collecting...'}`}
                    />
                    <div className="col-span-2 flex items-center gap-1.5">
                      <span className="text-[#9b9fa6]">Gate RMS:</span>
                      <span className="font-mono text-[#1e1f22]">
                        {modelsData.runtime.noise_gate_rms}
                      </span>
                      {modelsData.settings.noise_gate_auto && (
                        <span className="rounded-full bg-[#f0ebe2] px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-[#7b8088]">
                          auto
                        </span>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div className="mt-4 text-xs text-[#aaa]">Voice server offline</div>
              )}
            </SmallCard>

            {/* System Metrics */}
            <SmallCard title="System Status">
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-[#5b6068]">
                <Metric label="Voice Engine" value={statusData?.voice_ok ? 'Online' : 'Offline'} />
                <Metric
                  label="Wake Model"
                  value={
                    modelsData?.models?.openwakeword_model?.exists
                      ? 'OpenWakeWord'
                      : 'ASR fallback'
                  }
                />
                <Metric label="Clients" value={statusData?.clients ?? '--'} />
                <Metric
                  label="ASR"
                  value={modelsData?.models?.deepgram_api?.exists ? 'Deepgram OK' : 'Missing'}
                />
                <Metric label="Wakes" value={metrics.wakes ?? '--'} />
                <Metric
                  label="Intent"
                  value={modelsData?.settings?.asr_backend === 'deepgram' ? 'Deepgram' : 'Legacy'}
                />
                <Metric label="Commands" value={metrics.commands_executed ?? '--'} />
                <Metric label="ASR Errors" value={metrics.asr_errors ?? '--'} />
                <Metric label="TTS Errors" value={metrics.tts_errors ?? '--'} />
                <Metric label="Last Wake" value={fmtTime(metrics.last_wake_ts)} />
                <Metric label="Last Final" value={fmtTime(metrics.last_final_ts)} />
                <Metric label="Last TTS" value={fmtTime(metrics.last_tts_ts)} />
              </div>
            </SmallCard>
          </aside>
        </main>

        {/* ── Experience Modal ────────────────────────────────────────────── */}
        {activeExperience && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 p-4 backdrop-blur-sm"
            role="dialog"
            aria-modal="true"
          >
            <div className="w-full max-w-4xl rounded-[30px] border border-[#e3ddd2] bg-white/95 p-6 shadow-[0_30px_70px_rgba(20,18,14,0.2)]">
              <div className="flex items-center justify-between gap-4">
                <button
                  type="button"
                  className="rounded-full border border-[#e3ddd2] px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-[#5b6068] transition-colors hover:border-[#c6a86d]/50"
                  onClick={closeModal}
                >
                  {'< Back'}
                </button>
                <div className="text-xs uppercase tracking-[0.3em] text-[#7b8088]">Experience</div>
              </div>

              <h2 className="mt-4 text-2xl font-semibold">{activeExperience.title}</h2>
              <p className="mt-1 text-sm text-[#9b9fa6]">{activeExperience.description}</p>

              <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {activeExperience.options.map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    className={`rounded-2xl border px-4 py-3 text-sm font-semibold uppercase tracking-[0.2em] transition-all active:scale-[0.97] ${
                      activeOptionByModal[activeExperience.id] === opt.id
                        ? 'border-transparent bg-[#c6a86d] text-white shadow-[0_8px_20px_rgba(198,168,109,0.3)]'
                        : 'border-[#e3ddd2] text-[#5b6068] hover:border-[#c6a86d]/50 hover:text-[#1e1f22]'
                    }`}
                    onClick={() => activateOption(activeExperience.id, opt.id)}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>

              <div className="mt-6 space-y-4">
                {activeExperience.options.map((opt) => {
                  const isActive = activeOptionByModal[activeExperience.id] === opt.id;
                  if (!isActive || !opt.faderLabel) return null;
                  const value = volumes[volumeKey(activeExperience.id, opt.id)] || 0;
                  return (
                    <div
                      key={opt.id}
                      className="rounded-2xl border border-[#e3ddd2] bg-white px-4 py-3"
                    >
                      <p className="text-sm font-semibold text-[#1e1f22]">{opt.faderLabel}</p>
                      <input
                        type="range"
                        className="mt-3 w-full accent-[#c6a86d]"
                        min="0"
                        max={SLIDER_MAX}
                        step="0.01"
                        value={value}
                        onChange={(e) =>
                          updateVolume(activeExperience.id, opt.id, Number(e.target.value))
                        }
                      />
                      <div className="mt-1 text-right text-[10px] text-[#9b9fa6]">
                        {Math.round((value / SLIDER_MAX) * 100)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Reusable Sub-Components (keep in same file for simplicity)
   ═══════════════════════════════════════════════════════════════════════════ */

function Card({ title, heading, aside, children }) {
  return (
    <div className="rounded-[26px] border border-[#e3ddd2] bg-white/90 p-6 shadow-[0_24px_50px_rgba(20,18,14,0.12)] backdrop-blur">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.28em] text-[#7b8088]">{title}</div>
          <h2 className="mt-2 text-2xl font-semibold">{heading}</h2>
        </div>
        {aside && <div className="text-xs text-[#5b6068]">{aside}</div>}
      </div>
      <div className="mt-6">{children}</div>
    </div>
  );
}

function SmallCard({ title, children }) {
  return (
    <div className="rounded-[22px] border border-[#e3ddd2] bg-white/90 p-5 shadow-[0_20px_40px_rgba(20,18,14,0.1)]">
      <div className="text-[11px] uppercase tracking-[0.2em] text-[#7b8088]">{title}</div>
      {children}
    </div>
  );
}

function StatusBadge({ label }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-[#e3ddd2] bg-white/60 px-3 py-2 text-[11px] uppercase tracking-[0.2em] text-[#7b8088]">
      {label}
    </div>
  );
}

function Row({ label, children }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <div className="shrink-0 text-[#7b8088]">{label}</div>
      <div className="text-right text-sm">{children}</div>
    </div>
  );
}

function Dot({ ok }) {
  return (
    <span
      className={`mt-[3px] h-2 w-2 shrink-0 rounded-full ${ok ? 'bg-emerald-400' : 'bg-red-400'}`}
    />
  );
}

function PkgBadge({ label, installed }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${
        installed
          ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
          : 'border-red-200 bg-red-50 text-red-600'
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${installed ? 'bg-emerald-400' : 'bg-red-400'}`} />
      {label}
    </span>
  );
}

function RuntimeItem({ on, amber, label }) {
  let dotClass = on ? 'bg-emerald-400' : 'bg-[#ccc]';
  if (amber) dotClass = 'bg-amber-400';
  return (
    <div className="flex items-center gap-1.5">
      <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
      <span className="text-[#5b6068]">{label}</span>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.2em] text-[#7b8088]">{label}</div>
      <div className="mt-1 text-sm font-semibold text-[#1e1f22]">{value}</div>
    </div>
  );
}

function Divider() {
  return <div className="my-2 border-t border-[#ede8e0]" />;
}
