from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
import torch
from transformers import pipeline
from googletrans import Translator
from PIL import Image
import io
app = FastAPI(title="Sanskriti")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

whisper_model = whisper.load_model("medium", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device != "cpu" else -1)
translator = Translator()
enhancer = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x4-64", device=0 if device != "cpu" else -1)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with open("temp_audio", "wb") as f:
            f.write(audio_bytes)

        result = whisper_model.transcribe("temp_audio")

        return {
            "success": True,
            "language": result.get("language", None),
            "text": result.get("text", "").strip()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/summarize")
async def summarize_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        with open("temp_audio", "wb") as f:
            f.write(audio_bytes)
        transcription = whisper_model.transcribe("temp_audio")

        text = transcription.get("text", "").strip()
        lang = transcription.get("language", "en")

        if lang not in ["en", "english"]:
            translated = translator.translate(text, dest="en")
            english_text = translated.text
        else:
            english_text = text

        # Step 3: Summarize
        summary = summarizer(english_text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]

        # Step 4: Return
        return {
            "success": True,
            "detected_language": lang,
            "original_text": text,
            "summary": summary
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/enhance-image")
async def enhance_image(file: UploadFile = File(...)):
    try:
        # Read input image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Enhance using HF model
        enhanced = enhancer(img)

        # Save to temp output
        output_path = "enhanced_output.png"
        enhanced.save(output_path)

        return FileResponse(output_path, media_type="image/png", filename="enhanced_image.png")

    except Exception as e:
        return {"success": False, "error": str(e)}
