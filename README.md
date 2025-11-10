RF-DETR POTATO — AI VIDEO INFERENCE (PYTORCH + OPENVINO)

Prosty projekt do detekcji/śledzenia ziemniaków w wideo. Działa na CPU lub GPU (CUDA).

WYMAGANIA

Python 3.10+

pip + virtualenv

FFmpeg (zalecany)

INSTALACJA (Windows, PowerShell)

Sklonuj repo i wejdź do folderu:
git clone https://github.com/Nygus193Pro/rf-detr-potato.git

cd rf-detr-potato

Utwórz i aktywuj wirtualne środowisko:
python -m venv .venv
..venv\Scripts\Activate.ps1
pip install --upgrade pip

Zainstaluj PyTorch:
CPU (działa u każdego):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

GPU (NVIDIA + CUDA 12.x):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Zainstaluj pozostałe pakiety:
pip install -r requirements.txt

URUCHOMIENIE
CPU:
python infer_video_pytorch.py --input demo.mp4 --output demo_results_simple --device cpu
GPU:
python infer_video_pytorch.py --input demo.mp4 --output demo_results_simple --device cuda

STRUKTURA
infer_video_pytorch.py
export_to_onnx.py
requirements.txt
demo.mp4
demo_results_simple/
README.md

UWAGI

demo.mp4 jest w repozytorium.

Wyniki i modele (demo_results_simple/, output_model/, *.pt, *.onnx, *.xml, *.bin) są ignorowane w .gitignore.

LICENCJA
MIT