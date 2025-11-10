# 🥔 RF-DETR Potato — AI Video Inference (PyTorch + OpenVINO)

Prosty projekt do detekcji i śledzenia ziemniaków w wideo.  
Działa na **CPU** lub **GPU (CUDA)**.

---

## ⚙️ Wymagania
- Python **3.10+**  
- pip + virtualenv  
- FFmpeg *(zalecany)*

---

## 💻 Instalacja (Windows, PowerShell)

### 1️⃣ Sklonuj repozytorium
```powershell
git clone https://github.com/Nygus193Pro/rf-detr-potato.git
cd rf-detr-potato
2️⃣ Utwórz i aktywuj środowisko
powershell
Skopiuj kod
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
3️⃣ Zainstaluj PyTorch
CPU (działa u każdego):

powershell
Skopiuj kod
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
GPU (NVIDIA + CUDA 12.x):

powershell
Skopiuj kod
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
4️⃣ Zainstaluj pozostałe pakiety
powershell
Skopiuj kod
pip install -r requirements.txt
▶️ Uruchomienie
CPU:

powershell
Skopiuj kod
python infer_video_pytorch.py --input demo.mp4 --output demo_results_simple --device cpu
GPU:

powershell
Skopiuj kod
python infer_video_pytorch.py --input demo.mp4 --output demo_results_simple --device cuda
📂 Struktura projektu
Skopiuj kod
├── infer_video_pytorch.py
├── export_to_onnx.py
├── requirements.txt
├── demo.mp4
├── demo_results_simple/
└── README.md
🧠 Uwagi
🎬 demo.mp4 znajduje się w repozytorium.

📁 Wyniki są zapisywane lokalnie po uruchomieniu w folderze demo_results_simple/.

📦 Modele i eksporty (output_model/, *.pt, *.onnx, *.xml, *.bin) są ignorowane w .gitignore.

📜 Licencja
MIT