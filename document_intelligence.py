import os
from PIL import Image
import streamlit as st
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText

# Caching markitdown initialization
@st.cache_resource(show_spinner=False)
def get_markitdown_client():
    from markitdown import MarkItDown
    return MarkItDown()

# Caching the OpenAI Privacy Filter token-classification pipeline
@st.cache_resource(show_spinner=False)
def load_privacy_filter(model_id="openai/privacy-filter"):
    # Using trust_remote_code=True to handle custom model weights/architectures safely
    return pipeline(
        task="token-classification",
        model=model_id,
        aggregation_strategy="simple",
        trust_remote_code=True,
        device="cpu"  # Force CPU for local execution inside the Docker environment
    )

# Caching the GLM-OCR model and processor
@st.cache_resource(show_spinner=False)
def load_glm_ocr(model_id="zai-org/GLM-OCR"):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cpu",  # Run locally on CPU in the slim container
        trust_remote_code=True
    )
    return processor, model

# Document conversion to Markdown using markitdown
def convert_document(file_path: str) -> str:
    try:
        md = get_markitdown_client()
        result = md.convert(file_path)
        return result.text_content
    except Exception as e:
        return f"Error during document conversion: {str(e)}"

# PII Redaction using local openai/privacy-filter
def redact_text(text: str) -> str:
    if not text.strip():
        return ""
        
    try:
        classifier = load_privacy_filter()
        # Since text can be very large, process it in safe chunks if it exceeds common limits
        # However, privacy-filter supports 128k context window, so we can process paragraphs
        paragraphs = text.split("\n\n")
        redacted_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                redacted_paragraphs.append("")
                continue
                
            results = classifier(para)
            
            # Sort by start index to merge contiguous/overlapping entities of same type
            sorted_entities = sorted(results, key=lambda x: x["start"])
            merged_entities = []
            
            for ent in sorted_entities:
                if not merged_entities:
                    merged_entities.append(dict(ent))
                    continue
                    
                prev = merged_entities[-1]
                # Merge if same group and contiguous or overlapping (within 1 char)
                if ent["entity_group"] == prev["entity_group"] and ent["start"] <= prev["end"] + 1:
                    prev["end"] = max(prev["end"], ent["end"])
                else:
                    merged_entities.append(dict(ent))
            
            # Trim boundary whitespace for each merged entity to preserve natural spacing
            trimmed_entities = []
            for ent in merged_entities:
                start = ent["start"]
                end = ent["end"]
                while start < end and para[start].isspace():
                    start += 1
                while end > start and para[end-1].isspace():
                    end -= 1
                if start < end:
                    trimmed_entities.append({
                        "start": start,
                        "end": end,
                        "entity_group": ent["entity_group"]
                    })
            
            # Sort in reverse order of start positions to avoid index shift errors during replacement
            final_entities = sorted(trimmed_entities, key=lambda x: x["start"], reverse=True)
            redacted_para = para
            for ent in final_entities:
                start = ent["start"]
                end = ent["end"]
                label = ent["entity_group"]
                placeholder = f"[{label.upper()}]"
                redacted_para = redacted_para[:start] + placeholder + redacted_para[end:]
            redacted_paragraphs.append(redacted_para)
            
        return "\n\n".join(redacted_paragraphs)
    except Exception as e:
        return f"Error during redaction: {str(e)}"

# Visual Document OCR using GLM-OCR
def ocr_document(image_path: str) -> str:
    try:
        processor, model = load_glm_ocr()
        abs_path = os.path.abspath(image_path)
        
        # Structure the request as per GLM-OCR official HF model card
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": abs_path},
                    {"type": "text", "text": "Text Recognition:"}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        return output_text.strip()
    except Exception as e:
        return f"Error during visual OCR: {str(e)}"

#耦合 translation with markdown (splitting by paragraph to fit MarianMT token limits)
def translate_markdown(text: str, src_lang: str, dest_lang: str = "it", load_translator_fn=None) -> str:
    if not text.strip() or load_translator_fn is None:
        return text
        
    try:
        # Load local MarianMT models via the main app's helper
        tokenizer, model = load_translator_fn(src_lang, dest_lang)
        
        # Split text into paragraphs to avoid model max length constraints
        paragraphs = text.split("\n")
        translated_paragraphs = []
        
        for para in paragraphs:
            # Maintain formatting elements like list markers, headers, and empty lines
            stripped = para.strip()
            if not stripped:
                translated_paragraphs.append("")
                continue
                
            # If it's a markdown header or image syntax, we might want to translate only the text
            # Here we do a simple line-by-line translation for robust structural matching
            try:
                inputs = tokenizer(para, return_tensors="pt", padding=True)
                translated_tokens = model.generate(**inputs, max_length=512)
                translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True).strip()
                translated_paragraphs.append(translated_text)
            except Exception:
                translated_paragraphs.append(para) # Fallback to original line on translation failure
                
        return "\n".join(translated_paragraphs)
    except Exception as e:
        return f"Error during translation: {str(e)}"
