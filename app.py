# RCDJ228 SNIPER M3 - VERSION "FORTERESSE" - TRIAD ONLY
import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import io
import os
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from pydub import AudioSegment

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3 - ULTRA ROBUST", page_icon="🎯", layout="wide")

# --- PARAMÈTRES ET RÉFÉRENTIELS ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "sniper_triads": {
        "major": [1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0],
        "minor": [1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(16, 185, 129, 0.2); background: linear-gradient(145deg, #111827, #0b0e14);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 12px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ROBUSTES ---

def normalize_audio_rms(y, target_db=-20.0):
    """Normalisation pour traiter tous les fichiers au même niveau sonore"""
    rms = np.sqrt(np.mean(y**2))
    if rms == 0: return y
    curr_db = 20 * np.log10(rms + 1e-10)
    gain = 10**((target_db - curr_db) / 20)
    return y * gain

def apply_robust_filters(y, sr):
    """Suppression de la batterie et filtrage fréquentiel"""
    # 1. Séparation Harmonique/Percussive (HPSS) - Isole la mélodie de la batterie
    y_harm, _ = librosa.effects.hpss(y, margin=3.0)
    
    # 2. Filtre passe-bande (80Hz - 5000Hz)
    nyq = 0.5 * sr
    low, high = 80 / nyq, 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr):
    """Analyse focalisée sur les basses fréquences pour la fondamentale"""
    nyq = 0.5 * sr
    b, a = butter(2, 150/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def solve_key_sniper(chroma_vector, bass_vector):
    """Algorithme de décision par corrélation sur triade pure"""
    best_score = -1
    best_key = "Unknown"
    
    # Normalisation des vecteurs
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    p_data = PROFILES["sniper_triads"]
    for mode in ["major", "minor"]:
        for i in range(12):
            ref = np.roll(p_data[mode], i)
            score = np.corrcoef(cv, ref)[0, 1]
            
            if bv[i] > 0.7: score += 0.3  # Bonus Fondamentale en Basse
            if cv[(i + 7) % 12] > 0.6: score += 0.1 # Bonus Quinte stable
            
            if score > best_score:
                best_score = score
                best_key = f"{NOTES_LIST[i]} {mode}"
    return {"key": best_key, "score": best_score}

def process_audio_precision(file_obj, file_name, _progress_callback=None):
    try:
        # Chargement intelligent
        ext = file_name.split('.')[-1].lower()
        if ext == 'm4a':
            audio = AudioSegment.from_file(file_obj, format="m4a")
            y = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
            if audio.channels == 2: y = y.reshape((-1, 2)).mean(axis=1)
            sr = audio.frame_rate
        else:
            y, sr = librosa.load(file_obj, sr=22050, mono=True)
        
        # 1. Normalisation du volume
        y = normalize_audio_rms(y)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        # 2. Estimation précise du désaccordage (Tuning)
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        
        # 3. Filtrage "Forteresse" (HPSS + Bandpass)
        y_filt = apply_robust_filters(y, sr)
        
        duration = librosa.get_duration(y=y, sr=sr)
        step, timeline, votes = 2, [], Counter()
        segments = list(range(0, max(1, int(duration) - step), 1))
        
        for idx, start in enumerate(segments):
            if _progress_callback: _progress_callback(int((idx / len(segments)) * 100), f"Scan : {start}s")
            idx_start, idx_end = int(start * sr), int((start + step) * sr)
            seg = y_filt[idx_start:idx_end]
            
            if len(seg) < 2048 or np.max(np.abs(seg)) < 0.001: continue
            
            # Chroma CQT avec correction de tuning et haute résolution
            c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
            c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
            
            b_seg = get_bass_priority(y[idx_start:idx_end], sr)
            res = solve_key_sniper(c_avg, b_seg)
            
            # Pondération : on favorise l'intro et l'outro
            weight = 1.3 if (start < 15 or start > (duration - 15)) else 1.0
            votes[res['key']] += int(res['score'] * 100 * weight)
            timeline.append(res['score'])

        if not votes: return None

        final_key = votes.most_common(1)[0][0]
        final_conf = int(np.mean(timeline) * 100)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        res_obj = {
            "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
            "conf": min(final_conf, 99), "tempo": int(float(tempo)),
            "tuning": round(440 * (2**(tuning/12)), 1), "name": file_name
        }

        # Notification Telegram
        if st.secrets.get("TELEGRAM_TOKEN"):
            try:
                msg = f"🎯 *SNIPER ULTRA*\n📄 `{file_name}`\n🎹 `{final_key.upper()}`\n🎡 `{res_obj['camelot']}`\n✅ `{res_obj['conf']}%` | `{res_obj['tuning']}Hz`"
                requests.post(f"https://api.telegram.org/bot{st.secrets['TELEGRAM_TOKEN']}/sendMessage", 
                              data={'chat_id': st.secrets['CHAT_ID'], 'text': msg, 'parse_mode': 'Markdown'}, timeout=1)
            except: pass

        del y, y_filt; gc.collect()
        return res_obj
    except Exception as e:
        st.error(f"Erreur technique : {e}")
        return None

def get_chord_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'sine'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.2, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 1.5);
        }});
    }}; """

# --- INTERFACE UTILISATEUR ---
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

st.title("🎯 RCDJ228 SNIPER M3 - ULTRA ROBUST")
st.caption("Filtre HPSS (Anti-Batterie) + Normalisation RMS + Auto-Tuning")

uploaded_files = st.file_uploader("📂 Déposer fichiers audio", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if uploaded_files:
    total = len(uploaded_files)
    bar = st.progress(0)
    
    for idx, f in enumerate(uploaded_files):
        if f.name not in st.session_state.processed_files:
            with st.status(f"Analyse chirurgicale : `{f.name}`", expanded=False):
                inner = st.progress(0)
                data = process_audio_precision(f, f.name, _progress_callback=lambda v, m: inner.progress(v))
                if data: st.session_state.processed_files[f.name] = data
        bar.progress((idx + 1) / total)

    for i, (name, data) in enumerate(reversed(st.session_state.processed_files.items())):
        st.markdown(f"<div class='file-header'>📊 {data['name']}</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="report-card">
                <h1 style="font-size:5.5em; margin:0; color:#10b981;">{data['key'].upper()}</h1>
                <p style="font-size:1.5em; opacity:0.8;">CAMELOT: <b>{data['camelot']}</b> | CONFIANCE: <b>{data['conf']}%</b></p>
            </div> """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
        with c3:
            btn_id = f"play_{i}"
            components.html(f"""<button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #10b981, #059669); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🔊 VÉRIFIER LA TRIADE</button>
                            <script>{get_chord_js(btn_id, data['key'])}</script>""", height=110)

if st.sidebar.button("🗑️ Vider l'historique"):
    st.session_state.processed_files = {}
    st.rerun()
