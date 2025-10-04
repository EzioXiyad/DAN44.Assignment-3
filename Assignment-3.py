"""
hf_model_gui.py

Run:
  # Normal GUI (will use stubs if dependencies missing):
  python hf_model_gui.py

  # Run internal self-tests (no heavy deps required):
  python hf_model_gui.py --selftest

Requirements for real-model mode :
  pip install gradio transformers torch torchvision torchaudio pillow
  # ffmpeg is required for whisper-based ASR models

"""

from typing import Any, Dict, List, Optional
import time
import sys
import random

# -------------------------
# Try imports and detect capability
# -------------------------
_missing = {}

# Gradio is required for the GUI. If it's missing, fail early with instruction.
try:
    import gradio as gr
except Exception as e:
    raise RuntimeError(
        "Missing required package 'gradio'. Please install it: pip install gradio"
    )

# Optional imports (we can run in stub mode if these are missing)
try:
    from PIL import Image
except Exception as e:
    Image = None
    _missing['PIL'] = e

try:
    import torch
except Exception as e:
    torch = None
    _missing['torch'] = e

try:
    from transformers import pipeline as hf_pipeline
except Exception as e:
    hf_pipeline = None
    _missing['transformers'] = e

# Determine if we can run *real* HF pipelines
CAN_RUN_REAL_PIPELINES = (hf_pipeline is not None) and (torch is not None)

# Helpful summary message (used in the UI)
def _dependency_summary() -> str:
    if CAN_RUN_REAL_PIPELINES:
        return "All required packages detected; real Hugging Face pipelines are available."
    else:
        miss = ", ".join(sorted(list(_missing.keys()))) if _missing else "unknown"
        install_cmd = (
            "pip install gradio transformers torch torchvision torchaudio pillow"
        )
        return (
            "⚠️ Some heavy dependencies are missing: " + miss + "\n\n"
            "This app will run in STUB (demo) mode so you can test the UI without "
            "downloading large models or installing PyTorch.\n\n"
            "To enable real Hugging Face models, install the packages: \n"
            f"    {install_cmd}\n\n"
            "Note: installing these will enable real model inference but may require "
            "significant disk space and GPU/CPU resources."
        )

# -------------------------
# Base model wrapper and two implementations (real / stub)
# -------------------------
class BaseModelWrapper:
    """Common interface for our model wrappers.

    Subclasses should implement .load() and .run().
    """

    def __init__(self, model_id: str, description: str, use_stub: bool = False):
        self.model_id = model_id
        self.description = description
        self.pipeline = None
        self.last_loaded: Optional[float] = None
        self.use_stub = use_stub

    def load(self):
        raise NotImplementedError

    def run(self, input_data: Any) -> Dict[str, Any]:
        raise NotImplementedError


# Real image classifier wrapper (uses HF pipeline if available)
class ImageClassifierReal(BaseModelWrapper):
    def __init__(self, model_id: str = "google/vit-base-patch16-224"):
        super().__init__(model_id, "Image classification (ViT)")

    def load(self):
        if self.pipeline is None:
            device = 0 if (torch is not None and torch.cuda.is_available()) else -1
            self.pipeline = hf_pipeline("image-classification", model=self.model_id, device=device)
            self.last_loaded = time.time()

    def run(self, input_data: Any) -> Dict[str, Any]:
        self.load()
        # transformers' image-classification returns a list of dicts
        out = self.pipeline(input_data, top_k=5)
        return {"predictions": out, "model_id": self.model_id, "stub": False}


# Real ASR wrapper (uses HF pipeline if available)
class ASRReal(BaseModelWrapper):
    def __init__(self, model_id: str = "openai/whisper-small"):
        super().__init__(model_id, "Automatic speech recognition (Whisper)")

    def load(self):
        if self.pipeline is None:
            device = 0 if (torch is not None and torch.cuda.is_available()) else -1
            self.pipeline = hf_pipeline("automatic-speech-recognition", model=self.model_id, device=device)
            self.last_loaded = time.time()

    def run(self, input_data: Any) -> Dict[str, Any]:
        self.load()
        out = self.pipeline(input_data)
        return {"transcription": out.get("text", ""), "raw": out, "model_id": self.model_id, "stub": False}


# -------------------------
# Stub (demo) wrappers used when heavy deps are missing
# -------------------------
class ImageClassifierStub(BaseModelWrapper):
    def __init__(self, model_id: str = "stub/vit"):
        super().__init__(model_id, "Stub image classifier (demo)", use_stub=True)

    def load(self):
        # Nothing to load in stub
        self.last_loaded = time.time()

    def run(self, input_data: Any) -> Dict[str, Any]:
        # Return some deterministic-ish demo predictions so the UI can render them
        labels = ["tabby cat", "golden retriever", "sports car", "person", "bicycle"]
        picks = random.sample(labels, 3)
        preds = [{"label": picks[i], "score": round(random.uniform(0.35, 0.99), 3)} for i in range(3)]
        return {"predictions": preds, "model_id": self.model_id, "stub": True}


class ASRStub(BaseModelWrapper):
    def __init__(self, model_id: str = "stub/whisper"):
        super().__init__(model_id, "Stub ASR (demo)", use_stub=True)

    def load(self):
        self.last_loaded = time.time()

    def run(self, input_data: Any) -> Dict[str, Any]:
        demo_transcripts = [
            "Hello, this is a demo transcription.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing one two three, this is a stubbed ASR output."
        ]
        text = random.choice(demo_transcripts)
        return {"transcription": text, "raw": {"demo": True}, "model_id": self.model_id, "stub": True}


# -------------------------
# GUI app
# -------------------------
class HFGuiApp:
    """Main application class that wires the UI and model wrappers.

    OOP notes are provided in the UI: encapsulation, polymorphism, single responsibility.
    """

    def __init__(self):
        # Choose wrapper implementations depending on environment capability
        if CAN_RUN_REAL_PIPELINES:
            ImageWrapper = ImageClassifierReal
            ASRWrapper = ASRReal
            self._mode_info = "REAL MODE: using Hugging Face pipelines"
        else:
            ImageWrapper = ImageClassifierStub
            ASRWrapper = ASRStub
            self._mode_info = "STUB MODE: missing heavy deps; using demo models"

        # Register the two models (different categories)
        self.models: Dict[str, BaseModelWrapper] = {
            "Image Classification (ViT)": ImageWrapper("google/vit-base-patch16-224"),
            "Speech-to-Text (Whisper-small)": ASRWrapper("openai/whisper-small"),
        }

        self.model_info: Dict[str, tuple] = {
            "Image Classification (ViT)": (
                "google/vit-base-patch16-224",
                "A Vision Transformer checkpoint suitable for general-purpose image classification."
            ),
            "Speech-to-Text (Whisper-small)": (
                "openai/whisper-small",
                "OpenAI Whisper small model for transcription; good balance of speed & accuracy."
            ),
        }

        # OOP explanation text
        self.oop_explanation = self._build_oop_explanation()

        # Build GUI
        self.app = self._build_ui()

    def _build_oop_explanation(self) -> str:
        return (
            "OOP concepts used:\n"
            "1) Encapsulation: Model loading and running are encapsulated in wrapper classes.\n"
            "2) Polymorphism: All wrappers provide a common .run() API so the GUI can call them uniformly.\n"
            "3) Single Responsibility: Wrappers manage model I/O; HFGuiApp manages UI and wiring.\n"
            "4) Lazy loading: Real pipelines are only loaded when first used (resource friendly).\n\n"
            "Why used: This structure lets you add more model types (text, image, audio) easily and keeps code testable and modular."
        )

    def _build_ui(self):
        with gr.Blocks(title="Hugging Face Model Integrator — Robust Version") as demo:
            gr.Markdown("# Hugging Face — Model Integrator (Image classification & ASR)")

            # Dependency status / instructions
            with gr.Box():
                gr.Markdown(_dependency_summary())
                gr.Markdown(f"**Run mode:** {self._mode_info}")

            with gr.Row():
                with gr.Column(scale=2):
                    model_dropdown = gr.Dropdown(list(self.models.keys()), label="Select model", value=list(self.models.keys())[0])
                    input_type = gr.Dropdown(["image", "audio", "text"], value="image", label="Input data type")

                    image_in = gr.Image(type="pil", label="Upload image", visible=True)
                    audio_in = gr.Audio(source="upload", type="filepath", label="Upload audio", visible=False)
                    text_in = gr.Textbox(lines=3, label="Enter text (for models that accept text)", visible=False)

                    run_btn = gr.Button("Run model")

                    output_box = gr.JSON(label="Model output (raw)")
                    pretty_out = gr.Textbox(label="Readable output (summary)", interactive=False)

                    # Quick self-test button (runs light internal tests)
                    test_btn = gr.Button("Run quick self-tests")
                    test_results = gr.Textbox(label="Self-test results", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Model information")
                    model_info_box = gr.Textbox(label="Model info", interactive=False, lines=6)

                    gr.Markdown("### Explanations & OOP notes")
                    oop_box = gr.Textbox(value=self.oop_explanation, interactive=False, lines=12)

            # Toggle input visibility based on input_type
            def toggle_inputs(choice):
                return {image_in: gr.update(visible=(choice=="image")),
                        audio_in: gr.update(visible=(choice=="audio")),
                        text_in: gr.update(visible=(choice=="text"))}

            input_type.change(toggle_inputs, inputs=[input_type], outputs=[image_in, audio_in, text_in])

            # Update model info when model dropdown changes
            def update_model_info(model_name: str):
                mid, desc = self.model_info.get(model_name, ("unknown", "No description available."))
                info = f"Model name: {model_name}\nHF id: {mid}\nDescription: {desc}"
                # If running in stub mode, mention it
                if not CAN_RUN_REAL_PIPELINES:
                    info += "\n\nNOTE: Running in stub/demo mode (missing torch/transformers)."
                return info

            model_dropdown.change(update_model_info, inputs=[model_dropdown], outputs=[model_info_box])

            # Main run logic
            def run_model(selected_model_name, input_type_val, image_obj, audio_path, text_val):
                wrapper = self.models[selected_model_name]

                # Pick forwarded input based on selected input type
                forwarded_input = None
                if input_type_val == "image":
                    if image_obj is None:
                        return {output_box: {"error": "No image provided."}, pretty_out: "Please upload an image."}
                    forwarded_input = image_obj
                elif input_type_val == "audio":
                    if audio_path is None:
                        return {output_box: {"error": "No audio provided."}, pretty_out: "Please upload audio."}
                    forwarded_input = audio_path
                elif input_type_val == "text":
                    if not text_val:
                        return {output_box: {"error": "No text provided."}, pretty_out: "Please enter text."}
                    forwarded_input = text_val

                try:
                    result = wrapper.run(forwarded_input)
                except Exception as e:
                    return {output_box: {"error": str(e)}, pretty_out: f"Model run failed: {e}"}

                # Build human readable summary
                if isinstance(wrapper, (ImageClassifierReal, ImageClassifierStub)):
                    preds = result.get("predictions", [])
                    try:
                        pretty = "\n".join([f"{p.get('label', p.get('label', ''))}: {p.get('score', 0):.3f}" for p in preds])
                    except Exception:
                        # Fallback formatting
                        pretty = str(preds)
                elif isinstance(wrapper, (ASRReal, ASRStub)):
                    pretty = result.get("transcription", "(no transcription returned)")
                else:
                    pretty = str(result)

                return {output_box: result, pretty_out: pretty}

            run_btn.click(
                run_model,
                inputs=[model_dropdown, input_type, image_in, audio_in, text_in],
                outputs=[output_box, pretty_out]
            )

            # wire up quick self-test to button
            def quick_self_tests():
                """A few very light checks that exercise the stub wrappers.

                This function is safe to run in any environment (it doesn't load heavy models).
                """
                try:
                    # Test image stub
                    img_wrapper = ImageClassifierStub() if not CAN_RUN_REAL_PIPELINES else ImageClassifierReal()
                    asr_wrapper = ASRStub() if not CAN_RUN_REAL_PIPELINES else ASRReal()

                    ic_res = img_wrapper.run(None)
                    as_res = asr_wrapper.run(None)

                    ic_ok = isinstance(ic_res.get("predictions"), list)
                    as_ok = isinstance(as_res.get("transcription"), str)

                    return (
                        f"Image wrapper returned predictions: {ic_ok} (sample len={len(ic_res.get('predictions',[]))})\n"
                        f"ASR wrapper returned transcription: {as_ok} (sample text='{as_res.get('transcription')[:60]}')\n"
                        f"Mode: {'REAL' if CAN_RUN_REAL_PIPELINES else 'STUB'}"
                    )
                except Exception as e:
                    return f"Self-test failed: {e}"

            test_btn.click(quick_self_tests, inputs=[], outputs=[test_results])

            # Initialize model info with default selection
            demo.load(lambda: update_model_info(list(self.models.keys())[0]), outputs=[model_info_box])

        return demo

    def launch(self):
        self.app.launch()


# -------------------------
# Minimal internal self-test runner for CLI usage
# -------------------------

def _run_internal_tests_cli() -> int:
    """Run a few quick checks and exit with non-zero code on failure.

    This function does not require heavy dependencies because it uses stub classes when
    real pipelines are not available. Use: python hf_model_gui.py --selftest
    """
    try:
        # Image stub
        img_stub = ImageClassifierStub()
        res_ic = img_stub.run(None)
        assert isinstance(res_ic.get("predictions"), list) and len(res_ic["predictions"]) >= 1

        # ASR stub
        asr_stub = ASRStub()
        res_asr = asr_stub.run(None)
        assert isinstance(res_asr.get("transcription"), str)

        print("Internal self-tests passed (stub mode).")
        return 0
    except AssertionError as ae:
        print("Self-test assertions failed:", ae)
        return 2
    except Exception as e:
        print("Self-test encountered an error:", e)
        return 3


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="Run internal self-tests and exit (safe, no heavy deps required)")
    args = parser.parse_args()

    if args.selftest:
        code = _run_internal_tests_cli()
        sys.exit(code)

    app = HFGuiApp()
    app.launch()
