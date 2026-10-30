# Parakeet Lite Desktop Assistant

Parakeet Lite is a local-first Windows desktop assistant. It records microphone audio, transcribes English speech with faster-whisper, and streams answers from a locally running Ollama model.

## Requirements

- Python 3.10+
- Tkinter (included with the standard Windows Python installer)
- [Ollama](https://ollama.com/) running locally
- Optional: NVIDIA drivers/CUDA, PyAudio, and faster-whisper for live recording and transcription

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
uv sync --extra test --extra audio --extra transcription
ollama pull qwen2.5:3b
uv run python -m parakeet_lite.main
```

The first launch creates `config.json`. It can be edited to change model names, Ollama URL, recording duration, and hotkeys. `Ctrl+R` starts recording, `Ctrl+S` stops it, and `Ctrl+A` requests an answer.

If PyAudio or faster-whisper is not installed, the application still starts and reports the missing capability instead of failing during import. CUDA is optional; the model manager falls back to CPU mode.

## Testing

```powershell
pytest
```

If `uv` is not on `PATH`, use its installed Windows path:

```powershell
C:\Python314\Scripts\uv.exe sync --extra test --extra audio --extra transcription
C:\Python314\Scripts\uv.exe run python -m parakeet_lite.main
```

Alternatively, run directly with the already-created environment:

```powershell
.\.venv\Scripts\python.exe -m parakeet_lite.main
```

## Packaging

Install PyInstaller and build a Windows executable:

```powershell
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name ParakeetLite parakeet_lite\main.py
```
