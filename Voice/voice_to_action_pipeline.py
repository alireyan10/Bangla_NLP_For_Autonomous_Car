"""

1. Audio conversion: Any audio (.mp4, .m4a, .mp3) -> 16kHz Mono .wav via FFmpeg.
2. Speech-to-Text: Transcribes Bangla audio via Google Speech Recognition API (bn-BD).
3. Domain Noise Cleaner: Auto-corrects STT errors using robot vocabulary 
4. Action parsing: Multi-Head BanglaBERT (ultra-fast) OR Qwen2.5-3B (complex reasoning LLM).


# 1. Run all voices with BanglaBERT (default fast edge model):
python Voice/voice_to_action_pipeline.py

# 2. Run with Qwen2.5-3B LLM (high-accuracy reasoning for complex commands):
python Voice/voice_to_action_pipeline.py --model qwen

# 3. Process a single file:
python Voice/voice_to_action_pipeline.py --input "Voice/4.mp4" --model qwen

# 4. Live Microphone input with Qwen:
python Voice/voice_to_action_pipeline.py --mic --model qwen

5. for BanglaBERT:
python Voice/voice_to_action_pipeline.py --input "Voice/4.mp4" --model banglabert

6.Activate folder :
cd "C:\Users\Reyan\Downloads\Case_Study"
.\bangla_robot_action_dataset\.venv\Scripts\Activate.ps1

"""

import os
import sys
import json
import argparse
import subprocess
import io
import difflib

# Set UTF-8 encoding for Windows terminal safely without closing streams
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
NON_LLM_DIR = os.path.join(PROJECT_ROOT, "training", "non_llm_banglabert_base")
QWEN_EXP3_DIR = os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_3_optimized_lora")

for d in [PROJECT_ROOT, NON_LLM_DIR, QWEN_EXP3_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)


# ===========================================================================
# 1. Domain-Aware Fuzzy STT Noise Auto-Corrector
# ===========================================================================
DOMAIN_DIRECTIONS = ["ডানে", "ডানদিকে", "বামে", "বামদিকে", "সামনে", "সোজা", "পেছনে", "পিছনে"]
DOMAIN_ACTIONS = ["যাও", "ঘুরো", "থামো", "দাঁড়াও", "চলো", "আস্তে", "ধীরে", "দ্রুত", "গতি"]
DOMAIN_TARGETS = [
    "টেবিল", "টেবিলের", "সোফা", "সোফার", "বিছানা", "বিছানার", "দরজা", "দরজার",
    "চেয়ার", "চেয়ারের", "ডাস্টবিন", "টিভি", "ফ্রিজ", "স্টেশন", "চার্জার"
]
DOMAIN_UNITS = ["মিটার", "ইঞ্চি", "কদম", "ডিগ্রি", "এক", "দুই", "তিন", "চার", "পাঁচ", "দশ"]
COMMON_SAFE_WORDS = {"রোবট", "তুমি", "তারপর", "পরে", "আর", "একটু", "কাছে", "দিকে", "থেকে", "গিয়ে", "করে"}

ALL_DOMAIN_KEYWORDS = DOMAIN_DIRECTIONS + DOMAIN_ACTIONS + DOMAIN_TARGETS + DOMAIN_UNITS

# Common phonetic STT confusions for Bangla speech recognition
PHONETIC_REPLACEMENTS = {
    "কানে": "ডানে",
    "টানে": "ডানে",
    "দানে": "ডানে",
    "বানে": "বামে",
    "বায়ে": "বামে",
    "শামনে": "সামনে",
    "সজা": "সোজা",
    "ছুফা": "সোফা",
    "টিবিল": "টেবিল",
    "দাড়াও": "দাঁড়াও",
}

def clean_and_correct_stt_text(raw_text: str) -> str:
    """
    Cleans STT text by mapping unwanted acoustic noise words (e.g. 'কানে' -> 'ডানে')
    to domain-relevant robotics vocabulary using phonetic dictionary and fuzzy matching.
    """
    words = raw_text.split()
    corrected_words = []
    corrections_made = []

    for word in words:
        clean_word = word.strip("।,?! ")

        # 1. Direct phonetic STT error dictionary
        if clean_word in PHONETIC_REPLACEMENTS:
            fixed = PHONETIC_REPLACEMENTS[clean_word]
            corrected_words.append(fixed)
            corrections_made.append(f"'{clean_word}' -> '{fixed}'")
            continue

        # 2. Known domain word or safe common particle
        if clean_word in ALL_DOMAIN_KEYWORDS or clean_word in COMMON_SAFE_WORDS:
            corrected_words.append(clean_word)
            continue

        # 3. Fuzzy matching with domain lexicon (edit distance threshold 0.70)
        fuzzy_matches = difflib.get_close_matches(clean_word, ALL_DOMAIN_KEYWORDS, n=1, cutoff=0.70)
        if fuzzy_matches:
            fixed = fuzzy_matches[0]
            corrected_words.append(fixed)
            corrections_made.append(f"'{clean_word}' -> '{fixed}' (fuzzy)")
        else:
            corrected_words.append(clean_word)

    final_text = " ".join(corrected_words)
    if corrections_made:
        print(f"[Auto-Correct] Replaced STT noise: {', '.join(corrections_made)}")
        print(f"[Auto-Corrected Text]: \"{final_text}\"")

    return final_text


# ===========================================================================
# 2. Audio Format Conversion (FFmpeg)
# ===========================================================================
def convert_to_wav(input_path: str, output_path: str = None) -> str:
    """Converts audio/video files to 16kHz Mono WAV using FFmpeg."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.wav"

    if input_path.lower().endswith(".wav") and os.path.exists(output_path):
        return output_path

    print(f"[FFmpeg] Converting '{os.path.basename(input_path)}' -> 16kHz Mono WAV...")
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",           # Mono channel
        "-ar", "16000",       # 16kHz standard sample rate
        output_path,
        "-y"                  # Overwrite
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    print(f"[FFmpeg] Output saved: {output_path}")
    return output_path


# ===========================================================================
# 3. Speech-to-Text (Google API)
# ===========================================================================
def speech_to_bangla_text(wav_path: str) -> str:
    """Transcribes WAV audio to Bengali text using Google Speech API."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("[Error] 'SpeechRecognition' library is missing. Install via: pip install SpeechRecognition")
        return None

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)

    try:
        print("[STT] Transcribing Bangla speech...")
        raw_text = recognizer.recognize_google(audio_data, language="bn-BD")
        print(f"[Raw STT Heard]: \"{raw_text}\"")
        clean_text = clean_and_correct_stt_text(raw_text)
        return clean_text
    except sr.UnknownValueError:
        print("[STT] Audio was unclear or unintelligible.")
        return None
    except sr.RequestError as e:
        print(f"[STT] API request failed: {e}")
        return None


def listen_from_microphone(model_wrapper=None):
    """Listens live from microphone and predicts robot action JSON."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("[Error] 'SpeechRecognition' library is missing. Install via: pip install SpeechRecognition pyaudio")
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("\n" + "=" * 65)
            print("MICROPHONE ACTIVE: Speak your Bangla command...")
            print("   (Example: 'রোবট ডানে যাও পরে বামে যাও এক মিটার সামনে যাও তারপর থামো')")
            print("=" * 65)
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
    except Exception as e:
        print(f"[Mic Error] Could not access microphone: {e}")
        print("Note: Ensure your microphone is connected and 'pyaudio' is installed (pip install pyaudio).")
        return None

    try:
        print("[STT] Recognizing speech...")
        raw_text = recognizer.recognize_google(audio, language="bn-BD")
        print(f"[Raw STT Heard]: \"{raw_text}\"")
        bangla_text = clean_and_correct_stt_text(raw_text)
        print(f"\n[Final Clean Command]: \"{bangla_text}\"")

        if model_wrapper is not None:
            raw_json = model_wrapper.predict(bangla_text)
            print("\n[Generated Autonomous Car Action JSON]:")
            print(raw_json)
            return raw_json
    except sr.UnknownValueError:
        print("[STT] Speech unclear or no sound detected.")
    except Exception as e:
        print(f"[STT Error]: {e}")
    return None


# ===========================================================================
# 4. Model Wrapper (BanglaBERT or Qwen2.5-3B LLM)
# ===========================================================================
class ModelPipelineWrapper:
    def __init__(self, model_type: str = "banglabert"):
        self.model_type = model_type.lower()
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        if self.model_type == "qwen":
            self._init_qwen()
        else:
            self._init_banglabert()

    def _init_banglabert(self):
        try:
            from training.non_llm_banglabert_base.inference import BanglaBERTBaseInferencePipeline
            print("[Model Init] Loading BanglaBERT Multi-Head (Fast Edge Model)...")
            self.model = BanglaBERTBaseInferencePipeline()
            print("[Model Init] BanglaBERT ready (2ms latency).")
        except Exception as e:
            print(f"[Model Warning] Could not load BanglaBERT: {e}")

    def _init_qwen(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        model_name = "Qwen/Qwen2.5-3B"
        adapter_candidates = [
            os.path.join(SCRIPT_DIR, "adapter"),
            os.path.join(PROJECT_ROOT, "adapter"),
            os.path.join(QWEN_EXP3_DIR, "outputs", "checkpoints", "checkpoint-1050"),
            os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_3_optimized_lora", "outputs", "checkpoints", "checkpoint-1050"),
            os.path.join(PROJECT_ROOT, "training", "qwen2.5_3b", "experiment_1_raw", "outputs", "adapter"),
        ]
        adapter_dir = next((c for c in adapter_candidates if os.path.exists(c)), None)
        if adapter_dir is None:
            adapter_dir = os.path.join(QWEN_EXP3_DIR, "outputs", "checkpoints", "checkpoint-1050")

        print(f"[Model Init] Loading Qwen2.5-3B LLM (4-bit QLoRA) from: {adapter_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        print("[Model Init] Qwen2.5-3B LLM ready (High-Accuracy Reasoning).")

    def predict(self, text: str) -> str:
        if self.model_type == "qwen":
            return self._predict_qwen(text)
        else:
            return self._predict_banglabert(text)

    def _predict_banglabert(self, text: str) -> str:
        if self.model is None:
            return "{\"error\": \"BanglaBERT model not loaded\"}"
        raw_json, _, _ = self.model.predict(text)
        return raw_json

    def _predict_qwen(self, text: str) -> str:
        if self.model is None:
            return "{\"error\": \"Qwen model not loaded\"}"
        import torch
        system_prompt = (
            "You are a robot command mapping assistant. "
            "Convert the given Bangla robot command into a structured JSON action plan "
            "with exactly these fields: intent, target, speed_category, linear_speed, "
            "angular_speed, action_sequence, step_count. Return only valid JSON."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean JSON markdown fences if present
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            raw_output = raw_output.split("```")[0].strip()
            
        return raw_output


def run_pipeline(input_audio_path: str, model_wrapper=None):
    """Executes full pipeline: Audio -> WAV -> Bangla Text -> Robot Action JSON."""
    print("\n" + "=" * 65)
    print(f"Input: {input_audio_path}")
    print("=" * 65)

    # 1. Convert to WAV
    wav_path = input_audio_path if input_audio_path.lower().endswith(".wav") else convert_to_wav(input_audio_path)

    # 2. Speech to text (with automatic domain noise cleanup)
    bangla_text = speech_to_bangla_text(wav_path)
    if bangla_text is None:
        return None

    print(f"[Final Processed Command]: \"{bangla_text}\"")

    # 3. Model prediction
    if model_wrapper is not None:
        print(f"[Model] Parsing action slots using '{model_wrapper.model_type}'...")
        raw_json = model_wrapper.predict(bangla_text)
        print("\n[Action JSON Output]:")
        print(raw_json)
        return raw_json

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice to Robot Action Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to input audio file")
    parser.add_argument("--mic", action="store_true", help="Record live from microphone")
    parser.add_argument("--model", type=str, default="banglabert", choices=["banglabert", "qwen"], help="Model backend: 'banglabert' (fast edge) or 'qwen' (high accuracy)")
    args = parser.parse_args()

    # Load model wrapper
    model_wrapper = ModelPipelineWrapper(model_type=args.model)

    voice_folder = os.path.join(PROJECT_ROOT, "Voice")

    if args.mic:
        # Direct Live Microphone Input
        listen_from_microphone(model_wrapper)
    elif args.input:
        # Single Audio File
        run_pipeline(args.input, model_wrapper)
    else:
        # Batch process all unique audio files in Voice/
        audio_files = sorted([f for f in os.listdir(voice_folder) if f.endswith((".mp4", ".wav", ".m4a", ".mp3")) and not f.endswith("_converted.wav")])
        bases = set()
        unique_files = []
        for f in audio_files:
            b, ext = os.path.splitext(f)
            if b not in bases:
                bases.add(b)
                unique_files.append(f)

        if not unique_files:
            print("[Warning] No audio files found in Voice/ directory.")
        for filename in unique_files:
            run_pipeline(os.path.join(voice_folder, filename), model_wrapper)
