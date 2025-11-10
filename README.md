AI Video Inference Demo

Prosty projekt do uruchamiania inferencji wideo w PyTorch.

Wymagania

Python 3.10+

pip + virtualenv

FFmpeg (zalecany)

Instalacja
git clone https://github.com/Nygus193Pro/rf-detr-potato.git
cd rf-detr-potato
python -m venv .venv
.venv\Scripts\activate     # Windows
# lub
source .venv/bin/activate  # Linux/macOS

PyTorch

CPU (zalecane):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


GPU (CUDA 12.x):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Reszta pakietów
pip install -r requirements.txt

Uruchomienie
python infer_video_pytorch.py --input demo.mp4 --output demo_results_simple --device cpu


GPU: --device cuda

Struktura
├── infer_video_pytorch.py
├── requirements.txt
├── demo.mp4
├── demo_results_simple/
└── README.md

Licencja

MIT