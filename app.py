import streamlit as st
import os
import tempfile
import time
import platform
import multiprocessing
from faster_whisper import WhisperModel

# Page configuration
st.set_page_config(
    page_title="Whispir - Trascrizione Locale",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dynamic Hardware Detection Functions
def get_cpu_info():
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line.lower():
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "CPU Generica"

def get_total_ram_gb():
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "memtotal" in line.lower():
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 1)
    except Exception:
        pass
    # Local fallback logic if not in Linux /proc
    import sys
    return 16.0  # Fallback guess

def get_cpu_cores():
    return multiprocessing.cpu_count()

# Detect specs dynamically
ram_gb = get_total_ram_gb()
cpu_cores = get_cpu_cores()
cpu_name = get_cpu_info()

# Recommend model based on RAM size
if ram_gb < 4.0:
    recommended_model = "tiny"
    rec_help = f"Rilevato sistema a basse risorse ({ram_gb} GB RAM). Consigliato: 'tiny' o 'base'."
elif ram_gb < 8.0:
    recommended_model = "small"
    rec_help = f"Rilevato sistema a risorse medie ({ram_gb} GB RAM). Consigliato: 'small'."
elif ram_gb < 16.0:
    recommended_model = "turbo"
    rec_help = f"Rilevato sistema performante ({ram_gb} GB RAM). Consigliato: 'turbo'."
else:
    recommended_model = "turbo"
    rec_help = f"Rilevato sistema ad alte prestazioni ({ram_gb} GB RAM). Consigliato: 'turbo' o 'large-v3' per massima precisione."

# Premium Custom CSS Injection for Glassmorphic & Modern Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Main font override */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Subtle background gradient */
.stApp {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #09090b 100%);
    color: #f4f4f5;
}

/* Custom glowing Title card */
.hero-title {
    background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.5rem;
    text-align: center;
}

.hero-subtitle {
    color: #a1a1aa;
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 2.5rem;
    font-weight: 300;
}

/* Glassmorphism containers */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* Style file uploader */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(167, 139, 250, 0.3) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}
section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #a78bfa !important;
    background: rgba(167, 139, 250, 0.03) !important;
}

/* Customized Status Box */
.status-box {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 8px;
    padding: 0.8rem;
    color: #93c5fd;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* Buttons style */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
}

/* Result box */
.transcript-box {
    background: rgba(9, 9, 11, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem;
    max-height: 400px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.95rem;
    white-space: pre-wrap;
    margin-top: 1rem;
    color: #e4e4e7;
}
</style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size):
    return WhisperModel(model_size, device="cpu", compute_type="int8")

# Helper function to cache local Italian translation model and tokenizer
@st.cache_resource(show_spinner=False)
def load_translator(src_lang, dest_lang="it"):
    supported = ["en", "es", "fr", "de"]
    model_lang = src_lang if src_lang in supported else "en"
    model_name = f"Helsinki-NLP/opus-mt-{model_lang}-{dest_lang}"
    
    # Lazy imports to speed up Streamlit startup
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Helper functions for subtitle formatting
def format_time(seconds: float, format_type: str = "srt") -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds >= 1000:
        milliseconds = 999
    
    if format_type == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    else:  # vtt
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

def segments_to_srt(segments) -> str:
    srt_content = []
    for i, seg in enumerate(segments, start=1):
        start_t = format_time(seg["start"], "srt")
        end_t = format_time(seg["end"], "srt")
        srt_content.append(f"{i}\n{start_t} --> {end_t}\n{seg['text']}\n")
    return "\n".join(srt_content)

def segments_to_vtt(segments) -> str:
    vtt_content = ["WEBVTT\n"]
    for i, seg in enumerate(segments, start=1):
        start_t = format_time(seg["start"], "vtt")
        end_t = format_time(seg["end"], "vtt")
        vtt_content.append(f"{start_t} --> {end_t}\n{seg['text']}\n")
    return "\n".join(vtt_content)

def segments_to_txt(segments) -> str:
    return "\n".join([seg["text"] for seg in segments])

# Header Section
st.markdown('<div class="hero-title">🎙️ Whispir</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Trascrizione e traduzione locale e privata di audio e video fino a 2h</div>', unsafe_allow_html=True)

# Main layout split into Sidebar controls and Main Panel
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Impostazioni Modello")
    
    model_options = ["tiny", "base", "small", "medium", "turbo", "large-v3"]
    default_index = model_options.index(recommended_model)
    
    model_size = st.selectbox(
        "Modello Whisper",
        options=model_options,
        index=default_index,
        help=f"{rec_help} Modelli più grandi richiedono più RAM ed elaborazione CPU."
    )
    
    task = st.selectbox(
        "Modalità (Task)",
        options=["transcribe", "translate", "translate_to_italian"],
        format_func=lambda x: {
            "transcribe": "Trascrivi (Lingua Originale)",
            "translate": "Traduci in Inglese (Nativo Whisper)",
            "translate_to_italian": "Traduci in Italiano (MarianMT locale)"
        }[x],
        index=0
    )
    
    languages = {
        "Auto (Rilevamento automatico)": "Auto",
        "Italiano": "it",
        "Inglese": "en",
        "Spagnolo": "es",
        "Francese": "fr",
        "Tedesco": "de",
        "Portoghese": "pt",
        "Cinese": "zh",
        "Giapponese": "ja",
        "Russo": "ru"
    }
    
    lang_label = st.selectbox(
        "Lingua Sorgente",
        options=list(languages.keys()),
        index=0
    )
    language_code = languages[lang_label]
    
    beam_size = st.slider(
        "Beam Size",
        min_value=1,
        max_value=10,
        value=5,
        help="Un valore più alto aumenta la precisione a scapito della velocità. Default: 5."
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dynamic Hardware Spec Box
    st.markdown('<div class="glass-card" style="padding: 1rem; border-color: rgba(167, 139, 250, 0.2);">', unsafe_allow_html=True)
    st.markdown("##### 💻 Specifiche Rilevate")
    st.markdown(f"**CPU**: `{cpu_name}`")
    st.markdown(f"**Core**: `{cpu_cores} vCPU`")
    st.markdown(f"**RAM Rilevata**: `{ram_gb} GB`")
    st.markdown(f"**Suggerimento**: `{recommended_model.upper()}`")
    st.markdown("</div>", unsafe_allow_html=True)

# Main Panel
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📤 Carica Audio o Video")
uploaded_file = st.file_uploader(
    "Trascina qui il tuo file (formati supportati: MP4, MKV, AVI, MOV, MP3, WAV, M4A, AAC, FLAC)",
    type=["mp4", "mkv", "avi", "mov", "mp3", "wav", "m4a", "aac", "flac"]
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_details = {
        "Nome File": uploaded_file.name,
        "Dimensione": f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**File selezionato:** {file_details['Nome File']}")
    with col2:
        st.write(f"**Dimensione:** {file_details['Dimensione']}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process Button
    if st.button("Avvia Elaborazione"):
        # Create temp file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            
        try:
            # 1. Load Whisper Model
            with st.spinner(f"Caricamento del modello Whisper '{model_size}' in memoria (offline)..."):
                model = load_whisper_model(model_size)
                
            st.markdown('<div class="status-box">Elaborazione del file multimediale in corso...</div>', unsafe_allow_html=True)
            
            # Setup indicators
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            st.write("### 📝 Preview Trascrizione / Traduzione in Tempo Reale")
            preview_container = st.empty()
            
            # Determine source language & load translator if translate_to_italian is selected
            whisper_task = "transcribe"
            if task == "translate":
                whisper_task = "translate"
            
            translator_tokenizer = None
            translator_model = None
            detected_lang = language_code
            
            if task == "translate_to_italian":
                # Quick lazy language detection
                if language_code == "Auto":
                    with st.spinner("Rilevamento automatico della lingua sorgente in corso..."):
                        # Lazy generator call: running transcribe only does language detection initially
                        _, detect_info = model.transcribe(temp_file_path, task="transcribe")
                        detected_lang = detect_info.language
                        st.info(f"Lingua rilevata: `{detected_lang.upper()}`")
                
                if detected_lang == "it":
                    whisper_task = "transcribe" # Already Italian, just transcribe
                elif detected_lang in ["en", "es", "fr", "de"]:
                    whisper_task = "transcribe"
                    with st.spinner(f"Caricamento del traduttore locale ({detected_lang.upper()} -> IT)..."):
                        translator_tokenizer, translator_model = load_translator(detected_lang, "it")
                else:
                    # Fallback pipeline: Translate to English first via Whisper, then to Italian via MarianMT
                    whisper_task = "translate"
                    with st.spinner("Caricamento del traduttore locale (EN -> IT)..."):
                        translator_tokenizer, translator_model = load_translator("en", "it")
            
            # Call transcription
            segments, info = model.transcribe(
                temp_file_path,
                beam_size=beam_size,
                language=None if language_code == "Auto" else language_code,
                task=whisper_task,
                vad_filter=True
            )
            
            # Real-time parsing loop
            all_segments = []
            transcribed_text = ""
            
            start_time = time.time()
            
            for segment in segments:
                text_to_show = segment.text.strip()
                
                # Apply local Italian translation if active
                if translator_tokenizer is not None and translator_model is not None:
                    try:
                        inputs = translator_tokenizer(text_to_show, return_tensors="pt", padding=True)
                        translated_tokens = translator_model.generate(**inputs, max_length=512)
                        text_to_show = translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True).strip()
                    except Exception as e:
                        # Fallback to original text in case of error
                        pass
                
                # Save processed segment
                custom_seg = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": text_to_show
                }
                all_segments.append(custom_seg)
                
                # Update progress based on timestamp vs total duration
                duration = info.duration
                if duration > 0:
                    percent = min(segment.end / duration, 1.0)
                    progress_bar.progress(percent)
                    elapsed = time.time() - start_time
                    eta = (elapsed / percent - elapsed) if percent > 0 else 0
                    status_text.text(f"Progresso: {percent*100:.1f}% | Segmento: {segment.start:.1f}s - {segment.end:.1f}s | Tempo trascorso: {int(elapsed)}s | Tempo rimasto stimato: {int(eta)}s")
                
                # Append to preview text with timestamps
                timestamp_str = f"[{format_time(segment.start)} --> {format_time(segment.end)}]"
                transcribed_text += f"{timestamp_str} {text_to_show}\n"
                
                # Render inside the scrollable container
                preview_container.markdown(f'<div class="transcript-box">{transcribed_text}</div>', unsafe_allow_html=True)
                
            # Completed status
            progress_bar.progress(1.0)
            elapsed_total = time.time() - start_time
            status_text.success(f"Elaborazione Completata con Successo in {int(elapsed_total)} secondi!")
            
            if not all_segments:
                st.warning("Non è stato possibile rilevare alcun parlato nel file multimediale fornito.")
            else:
                # 3. Create downloads
                txt_data = segments_to_txt(all_segments)
                srt_data = segments_to_srt(all_segments)
                vtt_data = segments_to_vtt(all_segments)
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📥 Scarica i Risultati")
                
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                with dl_col1:
                    st.download_button(
                        label="Scarica Testo Semplice (.txt)",
                        data=txt_data,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt",
                        mime="text/plain"
                    )
                with dl_col2:
                    st.download_button(
                        label="Scarica Sottotitoli SRT (.srt)",
                        data=srt_data,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}.srt",
                        mime="text/srt"
                    )
                with dl_col3:
                    st.download_button(
                        label="Scarica Sottotitoli VTT (.vtt)",
                        data=vtt_data,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}.vtt",
                        mime="text/vtt"
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Si è verificato un errore durante l'elaborazione del file: {str(e)}")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
