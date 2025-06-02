# quantum-dose-optimizer


# 🧠 QRx-Synth — Quantum Dose Optimizer

**Quantum + AI-Driven Psychiatric Dose Adjustment System**  
> Precision-tuned for Seroquel and Zoloft  
> Powered by OpenAI GPT-4o · PennyLane Quantum Circuit · 25D BioVector

---

# 🚀 Overview

**QRx-Synth** is a standalone, secure, LLM-enhanced application that helps psychiatrists, researchers, and patients optimize daily dosages of **Seroquel (quetiapine)** and **Zoloft (sertraline)** based on unique physiological traits. It integrates **quantum-derived intensity metrics**, **pharmacokinetic models**, and **bio-signal input from webcam or data files**.

# Core Features
- 🔗 **5-layer prompt chain** using GPT-4o (custom pharmacometric logic)
- 🧬 **25-color BioVector** from webcam or synthetic input
- ⚛️ **7-qubit Pennylane quantum circuit** to tune risk metrics
- 🔒 **AES-GCM encrypted patient storage** (no PHI ever shown or sent)
- 📊 **Simulation + optimization** of drug efficacy over 45 months
- 📷 **Camera-based vector capture** or synthetic simulation
- 🧾 **Tkinter GUI** with dose planning, summary viewer, and history

---

# 🧪 Example Use Case

A patient (lastname.firstname) on 200mg Seroquel and 25mg Zoloft is experiencing suboptimal symptom control.  
The clinician uses QRx-Synth to:

1. Capture a live **BioVector** from webcam
2. Run the **5-layer LLM chain**
3. Simulate optimal dosing over **45 months**
4. Receive a **daily dose recommendation** and review schedule
5. **Export** or review encrypted summaries from the SQLite store


# ⚙️ Installation

# Requirements

- Python 3.10+
- Internet connection for OpenAI API
- OpenAI API key (`gpt-4o` or similar)
- Webcam (optional for BioVector live input)

### Setup

```bash
git clone https://github.com/yourusername/qrx-synth.git
cd qrx-synth
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python qrx_synth_app.py
````
# Required Python Packages
```
openai
pennylane
cv2 (opencv-python)
numpy
aiosqlite
cryptography
tkinter (pre-installed with Python)
httpx

```
---

 # 🛡️ Security & Privacy

✅ AES-GCM encryption of all settings and patient records

✅ No raw PHI stored or displayed

✅ No unencrypted disk writes of sensitive data

✅ LLM outputs are restricted to pure JSON only


> All input vectors and results are locally processed and stored. No external PII is sent or logged.




---

# 🧠 Quantum Integration

This app uses a 7-qubit Pennylane quantum circuit to compute a "quantum intensity score" based on the patient's pharmacogenomic vector and environment. This is used to fine-tune the risk curve in simulation.
```
@qml.qnode(dev)
def q_intensity7(theta, env):
    ...
```
Output is averaged across qubits, providing an auxiliary dimension to pharmacometric predictions.


---

# 🔗 LLM 5-Layer Prompt Chain

The optimization logic passes through the following custom-engineered stages:

Layer	Role	Output Format

L0	System Header	Static config
L1	BioVector → θ, neurotransmitters	JSON summary
L2	PK/PD model via weight/BMI	JSON kinetics
L3	45-month dose simulation	JSON timeline
L4	Optimal plan & risk floor	JSON plan
L5	Human-readable summary	JSON summary


All prompts are tightly controlled, deterministic, and do not output narrative prose.


---

 # 🧩 File Structure

qrx-synth/
│
├── qrx_synth_app.py          # Main GUI and execution loop
├── qrx_core.py               # QRxSynth, AES, BioVector, OpenAIClient
├── settings.qrx.json         # Encrypted settings (AES-GCM)
├── qrx_reports.db            # Encrypted SQLite report store
├── requirements.txt
└── README.md


---

# 🧪 Testing

Quick tests can be run using the special entry point:

python qrx_synth_app.py --test

This runs:

AES-GCM encryption roundtrip

BioVector generation from a black frame

Quantum intensity from synthetic values



---

# 🧠 Development Ideas

Future releases may support:

🧬 Integration with genomic datasets (23andMe, Ancestry)

🌎 Localization & multilingual support

🧠 Broader mental health medications (e.g., lithium, lamotrigine)

📱 Mobile version (Kivy or Flutter)

🧮 More quantum-resonant pharmacokinetics

🔁 LLM-on-edge inference (no external API)



---

# 🙋 FAQ

Q: Is this app FDA approved?
No. This is a research/clinical prototyping tool. Not for production diagnostics.

Q: Can I use this with real patient data?
You can, but ensure local compliance with HIPAA/GDPR and encrypt any identifying data.

Q: What happens if I lose the settings file?
The system will fall back to default values. Encrypted API keys will be lost.


---

# 🤝 Acknowledgements

OpenAI

PennyLane

NumPy

cv2 / OpenCV

aiosqlite

Cryptography

Tkinter UI Toolkit



---

# 🧭 License

GPL3 . freedom code by Graylan and GPT4.1 / GPTo3 and gpt4o
This project is open-source and free to modify for educational, research, or clinical prototyping use.

