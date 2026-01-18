import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3 - HYBRID", page_icon="🎯", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

SNIPER_TRIADS = {
    "major": [1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0],
    "minor": [1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.15); color: #f87171;
        padding: 15px; border-radius: 15px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    .sniper-badge { background: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.7em; }
    .fortress-badge { background: #b91c1c; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.7em; margin-left:5px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---

def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def get_bass_priority(y, sr):
    nyq = 0.5 * sr
    b, a = butter(2, 150/nyq, btype='low')
    y_bass = lfilter(b, a, y)
    chroma_bass = librosa.feature.chroma_cqt(y=y_bass, sr=sr, n_chroma=12)
    return np.mean(chroma_bass, axis=1)

def solve_key_engine(chroma_vector, bass_vector, profile_set):
    best_score = -1
    best_key = "Unknown"
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    for mode in ["major", "minor"]:
        for i in range(12):
            ref = np.roll(profile_set[mode], i)
            score = np.corrcoef(cv, ref)[0, 1]
            if bv[i] > 0.6: score += 0.2
            if cv[(i + 7) % 12] > 0.5: score += 0.1
            if score > best_score:
                best_score = score
                best_key = f"{NOTES_LIST[i]} {mode}"
    return {"key": best_key, "score": best_score}

def process_audio(audio_file, file_name, progress_placeholder):
    status_text = progress_placeholder.empty()
    progress_bar = progress_placeholder.progress(0)

    def update_prog(value, text):
        progress_bar.progress(value)
        status_text.markdown(f"**{text} | {value}%**")

    update_prog(10, f"Chargement de {file_name}")
    y, sr = librosa.load(audio_file, sr=22050, mono=True)
    
    update_prog(30, "Filtrage et Auto-Tuning")
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)

    update_prog(50, "Analyse Harmonique Multi-Profils")
    step, timeline, votes = 6, [], Counter()
    segments = range(0, max(1, int(duration) - step), 2)
    
    for i, start in enumerate(segments):
        idx_start, idx_end = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_start:idx_end]
        if len(seg) == 0 or np.max(np.abs(seg)) < 0.005: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        b_seg = get_bass_priority(y[idx_start:idx_end], sr)
        
        res = solve_key_engine(c_avg, b_seg, PROFILES["krumhansl"])
        weight = 2.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})
        update_prog(50 + int((i / max(1, len(segments))) * 40), "Calcul chirurgical")

    update_prog(95, "Synthèse et Arbitrage Forteresse")
    most_common = votes.most_common(2)
    
    # SÉCURITÉ : Si aucune note détectée
    if not most_common:
        return {"key": "C major", "camelot": "8B", "conf": 0, "tempo": 0, "tuning": 440, "timeline": [], "chroma": np.zeros(12), "modulation": False, "arbitration": False, "target_key": None, "target_camelot": None, "name": file_name}

    primary_key = most_common[0][0]
    total_votes = sum(votes.values())
    mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / total_votes) > 0.3
    
    chroma_global = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    bass_global = get_bass_priority(y, sr)
    
    if mod_detected:
        fortress_res = solve_key_engine(chroma_global, bass_global, SNIPER_TRIADS)
        final_key = fortress_res['key']
        arbitration = True
    else:
        final_key = primary_key
        arbitration = False

    # CORRECTION VALUEERROR : Calcul de confiance sécurisé
    relevant_confs = [t['Conf'] for t in timeline if t['Note'] == final_key]
    final_conf = int(np.mean(relevant_confs) * 100) if relevant_confs else 0

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    res_obj = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99), "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline if timeline else [{"Temps":0, "Note":final_key, "Conf":0}],
        "chroma": chroma_global, "modulation": mod_detected,
        "arbitration": arbitration,
        "target_key": most_common[1][0] if mod_detected else None,
        "target_camelot": CAMELOT_MAP.get(most_common[1][0], "??") if mod_detected else None,
        "name": file_name
    }
    
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            status = "🛡️ FORTERESSE" if arbitration else "✅ STABLE"
            msg = f"🎯 *SNIPER HYBRID*\n📂 `{file_name}`\n🎹 `{final_key.upper()}`\n🔥 CONF: `{res_obj['conf']}%` | {status}"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
        except: pass

    update_prog(100, "Terminé")
    status_text.empty(); progress_bar.empty()
    del y, y_filt; gc.collect()
    return res_obj

def get_chord_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 2.0);
        }});
    }}; """

# --- DASHBOARD ---
st.title("🎯 RCDJ228 SNIPER M3 - HYBRID")

uploaded_files = st.file_uploader("📥 Déposez vos fichiers", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if uploaded_files:
    progress_zone = st.container()
    for f in reversed(uploaded_files):
        data = process_audio(f, f.name, progress_zone)
        st.markdown(f"<div class='file-header'>📂 {data['name']}</div>", unsafe_allow_html=True)
        color = "linear-gradient(135deg, #065f46, #064e3b)" if not data['arbitration'] else "linear-gradient(135deg, #7f1d1d, #450a0a)"
        
        st.markdown(f"""
            <div class="report-card" style="background:{color};">
                <p style="letter-spacing:5px; opacity:0.8; font-size:0.8em;">SNIPER v5.0 { '<span class="fortress-badge">TRIAD JUDGE ACTIVE</span>' if data['arbitration'] else '' }</p>
                <h1 style="font-size:5.5em; margin:10px 0; font-weight:900;">{data['key'].upper()}</h1>
                <p style="font-size:1.5em; opacity:0.9;">CAMELOT: <b>{data['camelot']}</b> &nbsp; | &nbsp; CONFIANCE: <b>{data['conf']}%</b></p>
                {f"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE : {data['target_key'].upper()} ({data['target_camelot']})</div>" if data['modulation'] else ""}
            </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
        with m3:
            btn_id = f"play_{hash(data['name'])}"
            components.html(f"""<button id="{btn_id}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🎹 TESTER LA TRIADE</button>
                                <script>{get_chord_js(btn_id, data['key'])}</script>""", height=110)

        c1, c2 = st.columns([2, 1])
        with c1:
            if data['timeline']:
                fig_tl = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                fig_tl.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tl, use_container_width=True)
        with c2:
            fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_radar.update_layout(template="plotly_dark", height=300, margin=dict(l=40, r=40, t=30, b=20), polar=dict(radialaxis=dict(visible=False)), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("<hr style='border-color: #30363d; margin-bottom:40px;'>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Sniper Control")
    if st.button("🧹 Vider la file"):
        st.cache_data.clear()
        st.rerun()
