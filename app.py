import streamlit as st
import os
import tempfile
import time
import platform
import multiprocessing
from faster_whisper import WhisperModel

# Page configuration
st.set_page_config(
    page_title="Whispir - Local Transcription",
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

# Premium Custom CSS Injection for Glassmorphic & Modern Theme (WCAG AAA Compliant Colors)
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
    background: linear-gradient(135deg, #c084fc 0%, #6366f1 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.5rem;
    text-align: center;
}

.hero-subtitle {
    color: #e2e8f0; /* WCAG AAA compliant text color (slate-200) */
    font-size: 1.1rem;
    text-align: center;
    margin-bottom: 2.5rem;
    font-weight: 300;
}

/* Glassmorphism containers */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.12); /* Slightly higher visibility border */
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    color: #f8fafc; /* WCAG AAA compliant text color (slate-50) */
}

/* Style file uploader */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}
section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #a78bfa !important;
    background: rgba(167, 139, 250, 0.05) !important;
}

/* Customized Status Box - WCAG AA compliant contrast colors */
.status-box {
    background: rgba(30, 58, 138, 0.4);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-radius: 8px;
    padding: 0.8rem;
    color: #bfdbfe; /* WCAG AAA compliant light blue */
    font-weight: 500;
    margin-bottom: 1rem;
}

/* Buttons style - WCAG AAA compliant violet/blue gradient with white text */
div.stButton > button {
    background: linear-gradient(135deg, #5b21b6 0%, #1e40af 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(91, 33, 182, 0.3) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(91, 33, 182, 0.5) !important;
}

/* Result box - solid dark background for WCAG text contrast readability */
.transcript-box {
    background: rgba(15, 23, 42, 0.85); /* Solid dark slate background */
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 1.5rem;
    max-height: 400px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.95rem;
    white-space: pre-wrap;
    margin-top: 1rem;
    color: #ffffff; /* Maximum contrast white text */
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
st.markdown('<div class="hero-title">Whispir</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Private local transcription and translation of audio and video up to 2 hours</div>', unsafe_allow_html=True)

# Main layout split into Sidebar controls and Main Panel
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Model Settings")
    
    model_options = ["tiny", "base", "small", "medium", "turbo", "large-v3"]
    default_index = model_options.index(recommended_model)
    
    model_size = st.selectbox(
        "Whisper Model",
        options=model_options,
        index=default_index,
        help=f"{rec_help} Larger models require more RAM and CPU processing."
    )
    
    task = st.selectbox(
        "Mode (Task)",
        options=["transcribe", "transcribe_and_translate_en", "transcribe_and_translate_it"],
        format_func=lambda x: {
            "transcribe": "Transcription Only",
            "transcribe_and_translate_en": "Transcription + English Translation",
            "transcribe_and_translate_it": "Transcription + Italian Translation"
        }[x],
        index=0
    )
    
    languages = {
        "Auto (Detect Language)": "Auto",
        "Italian": "it",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Portuguese": "pt",
        "Chinese": "zh",
        "Japanese": "ja",
        "Russian": "ru"
    }
    
    lang_label = st.selectbox(
        "Source Language",
        options=list(languages.keys()),
        index=0
    )
    language_code = languages[lang_label]
    
    beam_size = st.slider(
        "Beam Size",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values increase accuracy at the cost of speed. Default is 5."
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dynamic Hardware Spec Box
    st.markdown('<div class="glass-card" style="padding: 1rem; border-color: rgba(167, 139, 250, 0.2);">', unsafe_allow_html=True)
    st.markdown("##### Detected Specs")
    st.markdown(f"**CPU**: `{cpu_name}`")
    st.markdown(f"**Cores**: `{cpu_cores} vCPU`")
    st.markdown(f"**Detected RAM**: `{ram_gb} GB`")
    st.markdown(f"**Recommended**: `{recommended_model.upper()}`")
    st.markdown("</div>", unsafe_allow_html=True)

# Main Panel
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### Upload Audio or Video")
uploaded_file = st.file_uploader(
    "Drag and drop your file here (supported formats: MP4, MKV, AVI, MOV, MP3, WAV, M4A, AAC, FLAC)",
    type=["mp4", "mkv", "avi", "mov", "mp3", "wav", "m4a", "aac", "flac"]
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    file_details = {
        "File Name": uploaded_file.name,
        "Size": f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Selected File:** {file_details['File Name']}")
    with col2:
        st.write(f"**Size:** {file_details['Size']}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process Button
    if st.button("Start Processing"):
        # Create temp file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            
        try:
            # 1. Load Whisper Model
            with st.spinner(f"Loading Whisper model '{model_size}' into memory (offline)..."):
                model = load_whisper_model(model_size)
                
            st.markdown('<div class="status-box">Processing media file...</div>', unsafe_allow_html=True)
            
            # Setup indicators
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            st.write("### Real-Time Preview")
            
            # Determine source language
            detected_lang = language_code
            if language_code == "Auto":
                with st.spinner("Detecting source language..."):
                    _, detect_info = model.transcribe(temp_file_path, beam_size=beam_size, vad_filter=True)
                    detected_lang = detect_info.language
                    st.info(f"Detected Language: `{detected_lang.upper()}`")
            
            # Determine pipeline configuration
            passes = 1
            has_translation = False
            translator_tokenizer = None
            translator_model = None

            if task == "transcribe_and_translate_en":
                has_translation = True
                if detected_lang != "en":
                    passes = 2
            elif task == "transcribe_and_translate_it":
                has_translation = True
                if detected_lang == "it":
                    passes = 1
                elif detected_lang in ["en", "es", "fr", "de"]:
                    passes = 1
                    with st.spinner(f"Loading local translator ({detected_lang.upper()} -> IT)..."):
                        translator_tokenizer, translator_model = load_translator(detected_lang, "it")
                else:
                    passes = 2
                    with st.spinner("Loading local translator (EN -> IT)..."):
                        translator_tokenizer, translator_model = load_translator("en", "it")

            original_segments = []
            translated_segments = []

            # Previews layout
            if has_translation:
                tab_orig, tab_trans = st.tabs(["Original Transcript", "Translation"])
                with tab_orig:
                    preview_orig = st.empty()
                with tab_trans:
                    preview_trans = st.empty()
            else:
                preview_orig = st.empty()
                preview_trans = None

            # Pass 1: Transcription
            with st.spinner("Running Pass 1/2: Transcribing original audio..." if passes == 2 else "Transcribing audio..."):
                segments, info = model.transcribe(
                    temp_file_path,
                    beam_size=beam_size,
                    language=None if language_code == "Auto" else language_code,
                    task="transcribe",
                    vad_filter=True
                )
                
                orig_text = ""
                trans_text = ""
                duration = info.duration
                start_time = time.time()
                
                for segment in segments:
                    seg_data = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    }
                    original_segments.append(seg_data)
                    
                    # Update progress
                    if duration > 0:
                        percent = min(segment.end / duration, 1.0)
                        scaled_percent = percent * 0.5 if passes == 2 else percent
                        progress_bar.progress(scaled_percent)
                        elapsed = time.time() - start_time
                        pass_label = "Pass 1/2: Transcribing" if passes == 2 else "Transcribing"
                        status_text.text(f"{pass_label} | Progress: {percent*100:.1f}% | Segment: {segment.start:.1f}s - {segment.end:.1f}s | Elapsed: {int(elapsed)}s")
                    
                    # Render original preview
                    timestamp_str = f"[{format_time(segment.start)} --> {format_time(segment.end)}]"
                    orig_text += f"{timestamp_str} {seg_data['text']}\n"
                    preview_orig.markdown(f'<div class="transcript-box">{orig_text}</div>', unsafe_allow_html=True)
                    
                    # If passes == 1 and we have translation, it's either identical or we translate via MarianMT here
                    if passes == 1 and has_translation:
                        if detected_lang == "en" or detected_lang == "it":
                            # Identical
                            translated_text = seg_data['text']
                        else:
                            # MarianMT translation
                            translated_text = seg_data['text']
                            if translator_tokenizer is not None and translator_model is not None:
                                try:
                                    inputs = translator_tokenizer(seg_data['text'], return_tensors="pt", padding=True)
                                    translated_tokens = translator_model.generate(**inputs, max_length=512)
                                    translated_text = translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True).strip()
                                except Exception:
                                    pass
                        
                        translated_segments.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": translated_text
                        })
                        trans_text += f"{timestamp_str} {translated_text}\n"
                        if preview_trans is not None:
                            preview_trans.markdown(f'<div class="transcript-box">{trans_text}</div>', unsafe_allow_html=True)

            # Pass 2: Translate via Whisper if needed
            if passes == 2:
                with st.spinner("Running Pass 2/2: Translating to English..." if task == "transcribe_and_translate_en" else "Running Pass 2/2: Translating to English & Italian..."):
                    segments, info = model.transcribe(
                        temp_file_path,
                        beam_size=beam_size,
                        language=None if language_code == "Auto" else language_code,
                        task="translate",
                        vad_filter=True
                    )
                    
                    trans_text = ""
                    duration = info.duration
                    start_time = time.time()
                    
                    for segment in segments:
                        eng_text = segment.text.strip()
                        
                        if task == "transcribe_and_translate_en":
                            translated_text = eng_text
                        else: # transcribe_and_translate_it
                            translated_text = eng_text
                            if translator_tokenizer is not None and translator_model is not None:
                                try:
                                    inputs = translator_tokenizer(eng_text, return_tensors="pt", padding=True)
                                    translated_tokens = translator_model.generate(**inputs, max_length=512)
                                    translated_text = translator_tokenizer.decode(translated_tokens[0], skip_special_tokens=True).strip()
                                except Exception:
                                    pass
                        
                        translated_segments.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": translated_text
                        })
                        
                        # Update progress (from 50% to 100%)
                        if duration > 0:
                            percent = min(segment.end / duration, 1.0)
                            scaled_percent = 0.5 + (percent * 0.5)
                            progress_bar.progress(scaled_percent)
                            elapsed = time.time() - start_time
                            status_text.text(f"Pass 2/2: Translating | Progress: {percent*100:.1f}% | Segment: {segment.start:.1f}s - {segment.end:.1f}s | Elapsed: {int(elapsed)}s")
                        
                        # Update translation preview
                        timestamp_str = f"[{format_time(segment.start)} --> {format_time(segment.end)}]"
                        trans_text += f"{timestamp_str} {translated_text}\n"
                        if preview_trans is not None:
                            preview_trans.markdown(f'<div class="transcript-box">{trans_text}</div>', unsafe_allow_html=True)

            # Completed status
            progress_bar.progress(1.0)
            elapsed_total = time.time() - start_time
            status_text.success(f"Processing Completed Successfully in {int(elapsed_total)} seconds!")
            
            if not original_segments:
                st.warning("Could not detect any speech in the provided media file.")
            else:
                # 3. Create downloads
                txt_orig = segments_to_txt(original_segments)
                srt_orig = segments_to_srt(original_segments)
                vtt_orig = segments_to_vtt(original_segments)
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### Download Results")
                
                if has_translation:
                    txt_trans = segments_to_txt(translated_segments)
                    srt_trans = segments_to_srt(translated_segments)
                    vtt_trans = segments_to_vtt(translated_segments)
                    
                    col_orig, col_trans = st.columns(2)
                    
                    with col_orig:
                        st.markdown("#### Original Transcript")
                        st.download_button(
                            label="Download Plain Text (.txt)",
                            data=txt_orig,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_original.txt",
                            mime="text/plain",
                            key="dl_txt_orig"
                        )
                        st.download_button(
                            label="Download Subtitles SRT (.srt)",
                            data=srt_orig,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_original.srt",
                            mime="text/srt",
                            key="dl_srt_orig"
                        )
                        st.download_button(
                            label="Download Subtitles VTT (.vtt)",
                            data=vtt_orig,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_original.vtt",
                            mime="text/vtt",
                            key="dl_vtt_orig"
                        )
                        
                    with col_trans:
                        trans_lang_label = "Italian" if task == "transcribe_and_translate_it" else "English"
                        st.markdown(f"#### Translated Text ({trans_lang_label})")
                        st.download_button(
                            label="Download Plain Text (.txt)",
                            data=txt_trans,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_translated.txt",
                            mime="text/plain",
                            key="dl_txt_trans"
                        )
                        st.download_button(
                            label="Download Subtitles SRT (.srt)",
                            data=srt_trans,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_translated.srt",
                            mime="text/srt",
                            key="dl_srt_trans"
                        )
                        st.download_button(
                            label="Download Subtitles VTT (.vtt)",
                            data=vtt_trans,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_translated.vtt",
                            mime="text/vtt",
                            key="dl_vtt_trans"
                        )
                else:
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    with dl_col1:
                        st.download_button(
                            label="Download Plain Text (.txt)",
                            data=txt_orig,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt",
                            mime="text/plain",
                            key="dl_txt_single"
                        )
                    with dl_col2:
                        st.download_button(
                            label="Download Subtitles SRT (.srt)",
                            data=srt_orig,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.srt",
                            mime="text/srt",
                            key="dl_srt_single"
                        )
                    with dl_col3:
                        st.download_button(
                            label="Download Subtitles VTT (.vtt)",
                            data=vtt_orig,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.vtt",
                            mime="text/vtt",
                            key="dl_vtt_single"
                        )
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during file processing: {str(e)}")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
