# AI Video Inference Demo

Prosty projekt do uruchamiania inferencji wideo na modelu PyTorch.

---

## 🔧 Wymagania

* Python 3.10+
* pip + virtualenv
* PyTorch (CPU lub CUDA)
* FFmpeg (zalecany)

---

## 🚀 Szybki start

### 1️⃣ Klonowanie repo

```bash
git clone https://github.com/<twoj-nick>/<nazwa-projektu>.git
cd <nazwa-projektu>
```

### 2️⃣ Tworzenie środowiska

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# lub
source .venv/bin/activate  # Linux/macOS
```

### 3️⃣ Instalacja zależności

```bash
pip install -r requirements.txt
```

### 4️⃣ Uruchom demo

```bash
python infer_video_pytorch.py --input demo.mp4 --output demo_results_simple --device cpu
```

> Dla GPU: zamień `--device cpu` na `--device cuda`

---

## 📁 Struktura projektu

```
├── infer_video_pytorch.py
├── export_to_onnx.py
├── requirements.txt
├── demo.mp4
├── demo_results_simple/ (wyniki – ignorowane w git)
├── .gitignore
└── README.md
```

---

## 🧠 Uwagi

* `demo.mp4` jest w repozytorium.
* Wyniki i modele są ignorowane (`demo_results_simple/`, `output_model/`, `*.pt`, `*.onnx`, ...).

---

## 📜 Licencja

MIT 
