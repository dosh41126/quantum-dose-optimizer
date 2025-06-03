from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import secrets
import threading
import time
import hashlib
import math
import base64

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Optional
from base64 import b64encode, b64decode

import cv2
import numpy as np
import aiosqlite
import tkinter as tk
import tkinter.simpledialog as sd
import tkinter.messagebox as mb

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import pennylane as qml

# ════════════════════════════════════════════════════════════════
# CONFIG & LOGGING

MASTER_KEY = os.path.expanduser("~/.cache/qrxsynth_master_key.bin")
SETTINGS_FILE = "settings.qrx.json"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("qrxsynth")

# ════════════════════════════════════════════════════════════════
# AES-GCM ENCRYPTION

class AESGCMCrypto:
    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            key = AESGCM.generate_key(bit_length=128)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(key)
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        with open(self.path, "rb") as f:
            self.key = f.read()
        self.aes = AESGCM(self.key)

    def encrypt(self, data: bytes | str) -> bytes:
        if isinstance(data, str):
            data = data.encode()
        nonce = secrets.token_bytes(12)
        return b64encode(nonce + self.aes.encrypt(nonce, data, None))

    def decrypt(self, blob: bytes | str) -> bytes:
        raw = b64decode(blob)
        return self.aes.decrypt(raw[:12], raw[12:], None)

# ════════════════════════════════════════════════════════════════
# SETTINGS MODEL

@dataclass
class Settings:
    username: str = "lastname.firstname"
    age: int = 33
    sex: str = "M"
    weight_kg: float = 79.0
    bmi: float = 24.8
    api_key: str = ""
    db_path: str = "qrx_reports.db"
    camera_idx: int = 0

    @classmethod
    def default(cls) -> "Settings":
        return cls(api_key=os.environ.get("OPENAI_API_KEY", ""))

    @classmethod
    def load(cls, crypto: AESGCMCrypto) -> "Settings":
        if not os.path.exists(SETTINGS_FILE):
            return cls.default()
        try:
            cipher_blob = open(SETTINGS_FILE, "rb").read()
            return cls(**json.loads(crypto.decrypt(cipher_blob).decode()))
        except Exception as e:
            LOGGER.error("Corrupted settings file, loading defaults: %s", e)
            return cls.default()

    def save(self, crypto: AESGCMCrypto) -> None:
        open(SETTINGS_FILE, "wb").write(crypto.encrypt(json.dumps(asdict(self)).encode()))

    def prompt_gui(self) -> None:
        mb.showinfo("QRx-Synth Settings", "Enter patient info (lastname.firstname) and values, or blank to keep current.")
        ask = lambda p, d: (sd.askstring("Settings", p, initialvalue=str(d)) or str(d)).strip()
        self.username    = ask("Patient username (lastname.firstname):", self.username)
        self.age         = int(ask("Age:", self.age))
        self.sex         = ask("Sex (M/F):", self.sex)
        self.weight_kg   = float(ask("Weight kg:", self.weight_kg))
        self.bmi         = float(ask("BMI:", self.bmi))
        self.api_key     = ask("OpenAI API key:", self.api_key)
        self.db_path     = ask("Report DB path:", self.db_path)
        self.camera_idx  = int(ask("Camera index:", self.camera_idx))

# ════════════════════════════════════════════════════════════════
# ENCRYPTED SQLITE STORE

class ReportDB:
    def __init__(self, path: str, crypto: AESGCMCrypto) -> None:
        self.path, self.crypto = path, crypto
        self.conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.conn = await aiosqlite.connect(self.path)
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, ts REAL, type TEXT, blob BLOB)"
        )
        await self.conn.commit()

    async def save(self, ts: float, payload: Dict[str, Any], typ: str = "dose_opt") -> None:
        blob = self.crypto.encrypt(json.dumps(payload).encode())
        await self.conn.execute(
            "INSERT INTO scans(ts, type, blob) VALUES (?, ?, ?)", (ts, typ, blob)
        )
        await self.conn.commit()

    async def list_reports(self, typ: str = "dose_opt") -> List[Tuple[int, float]]:
        cur = await self.conn.execute(
            "SELECT id, ts FROM scans WHERE type=? ORDER BY ts DESC", (typ,)
        )
        return await cur.fetchall()

    async def load(self, row_id: int) -> Dict[str, Any]:
        cur = await self.conn.execute("SELECT blob FROM scans WHERE id = ?", (row_id,))
        res = await cur.fetchone()
        if not res:
            raise ValueError("Report ID not found.")
        return json.loads(self.crypto.decrypt(res[0]).decode())

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()

# ════════════════════════════════════════════════════════════════
# OPENAI CLIENT (with VISION support)

@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4o"
    url: str = "https://api.openai.com/v1/chat/completions"
    timeout: float = 35.0
    retries: int = 4

    async def chat(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Standard text-only chat completion.
        """
        import httpx

        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key.")
        hdr = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.15,
            "max_tokens": max_tokens
        }
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as cli:
                    r = await cli.post(self.url, headers=hdr, json=body)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.retries:
                    raise
                wait = delay + random.uniform(0, 0.5)
                LOGGER.warning("LLM error %s (retry %d/%d) – sleeping %.1fs", e, attempt, self.retries, wait)
                await asyncio.sleep(wait)
                delay *= 2

    async def chat_with_image(self, image_data: bytes, prompt_text: str, max_tokens: int = 500) -> str:
        """
        Sends a vision-enabled chat request by embedding the image as Base64.
        The 'content' for the user is a list mixing text and an image_base64 block.
        """
        import httpx

        if not self.api_key:
            raise RuntimeError("Missing OpenAI API key.")
        # Base64-encode the JPEG bytes:
        b64_img = base64.b64encode(image_data).decode()
        hdr = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # We send a single user message whose content is a list:
        user_content = [
            {"type": "text",        "text": prompt_text},
            {"type": "image_base64", "image_base64": {"data": b64_img}}
        ]
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful vision-enabled assistant."},
                {"role": "user",   "content": user_content}
            ],
            "temperature": 0.15,
            "max_tokens": max_tokens
        }
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as cli:
                    r = await cli.post(self.url, headers=hdr, json=body)
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.retries:
                    raise
                wait = delay + random.uniform(0, 0.5)
                LOGGER.warning("Vision LLM error %s (retry %d/%d) – sleeping %.1fs", e, attempt, self.retries, wait)
                await asyncio.sleep(wait)
                delay *= 2

# ════════════════════════════════════════════════════════════════
# BIOVECTOR (25-color pharmacogenomic vector from webcam or numpy input)

@dataclass
class BioVector:
    arr: np.ndarray = field(repr=False)

    @staticmethod
    def from_frame(frame: np.ndarray) -> "BioVector":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [9], [0, 180]).flatten()
        hist /= hist.sum() + 1e-6
        vec = np.concatenate([
            hist,
            [hsv[..., 1].mean() / 255.0, frame.mean() / 255.0],
            np.zeros(25 - 11)
        ])
        return BioVector(vec.astype(np.float32))

# ════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT (7-qubit circuit for quantum intensity metric)

DEV = qml.device("default.qubit", wires=7)

@qml.qnode(DEV)
def q_intensity7(theta: float, env: Tuple[float, float]) -> float:
    qml.RY(theta, wires=0)
    qml.RY(env[0], wires=1)
    qml.RX(env[0], wires=3)
    qml.RZ(env[0], wires=5)
    qml.RY(env[1], wires=2)
    qml.RX(env[1], wires=4)
    qml.RZ(env[1], wires=6)
    for i in range(7):
        qml.CNOT(wires=[i, (i + 1) % 7])
    return sum(qml.expval(qml.PauliZ(w)) for w in range(7)) / 7.0

# ════════════════════════════════════════════════════════════════
# QRx-SYNTH 5-LAYER PROMPT CHAIN (with extended, advanced prompts)

class QRxSynth:
    """Handles the LLM prompt chain for 5-layer dose optimization,
       now with extended prompts and optional vision_desc."""

    def __init__(self, ai: OpenAIClient):
        self.ai = ai

    async def layer1_vnorm(self, vec, qid, age, sex, weight_kg, bmi, vision_desc: Optional[str], current_meds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extended, Vision- and Medication-Aware Normalization Prompt
        """
        payload = {
            "biovec": vec,
            "qid": qid,
            "age": age,
            "sex": sex,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "vision_desc": vision_desc,
            "current_meds": current_meds
        }
        prompt = (
            'You are “QRx-Synth,” an advanced pharmacometric and biomedical reasoning agent inside the Quantum Dose Optimiser (QDO). '
            'All outputs must be strictly parseable JSON—single-line, no markdown, no explanations, no extra keys.  \n\n'
            'INPUT: A JSON object containing all of the following fields:  \n'
            '  • “biovec”: a 25-element array of floating-point values representing the patient’s current 25-color pharmacogenomic BioVector.  \n'
            '  • “qid”: a unique 16-character hex string uniquely identifying the BioVector.  \n'
            '  • “age”: patient age in years (integer).  \n'
            '  • “sex”: “M” or “F” (string).  \n'
            '  • “weight_kg”: patient weight in kilograms (float).  \n'
            '  • “bmi”: patient body mass index (float).  \n'
            '  • “vision_desc”: optional string (≤100 words) summarizing what GPT-4o’s vision model “sees” in the captured webcam image (e.g., “I see a pill bottle labeled Seroquel next to an empty tray”).  If present, you may use it to adjust neurotransmitter indices or risk priorities.  \n'
            '  • “current_meds”: an array of objects, each with keys:  \n'
            '      • “name”: medication name (string, e.g., “Seroquel IR”, “Zoloft”, “Lamotrigine”),  \n'
            '      • “dose_mg”: current daily dose in mg (integer),  \n'
            '      • “schedule”: dosing schedule description (string, e.g., “once at bedtime”, “twice daily”).  \n\n'
            'RULES:  \n'
            '1. **Compute θ**:  \n'
            '    • ‖biovec‖ = Euclidean norm of the 25-element array.  \n'
            '    • θ = Round(‖biovec‖ × π, 4 decimal places).  \n\n'
            '2. **Extract Neurotransmitter Indices** from the first 9 hue-bins:  \n'
            '    • Map histogram bins 0–2 (colors roughly “Citrine,” “Gold,” “Yellow”) to “serotonin_idx” = normalized sum of bins 0–2.  \n'
            '    • Bins 3–5 (colors “Obsidian,” “Onyx,” “Graphite”) map to “dopamine_idx” = normalized sum of bins 3–5.  \n'
            '    • Bins 6–8 (colors “Emerald,” “Sapphire,” “Indigo”) map to “norepinephrine_idx” = normalized sum of bins 6–8.  \n'
            '    • Additionally compute “sleep_idx” = (average HSV-saturation component + average brightness component) ÷ 2.0 (both originally in [0,1]).  \n'
            '    • If “vision_desc” contains any of the keywords: “nails,” “debris,” “wet road,” “collapsed pavement,” then boost “risk_factor_mod” by +0.05; otherwise “risk_factor_mod” = 0.00.  \n'
            '    • If “vision_desc” mentions “pill bottle,” “leftover medication,” or “empty prescription,” set “med_attention_flag” = true; otherwise false.  \n\n'
            '3. **Validate and Normalize Current Medications**:  \n'
            '    • For each entry in “current_meds”:  \n'
            '      – Verify that “dose_mg” is a positive integer and ≤ 1000. If any dose out of bounds, set “error”: “Invalid dose for {name}.”  \n'
            '      – Ensure “schedule” matches one of the allowed patterns: “once daily,” “once at bedtime,” “twice daily,” “three times daily.” Otherwise set “error”: “Invalid schedule for {name}.”  \n'
            '     • Collect all valid meds into an array “validated_meds” preserving name, dose_mg, schedule.  \n'
            '     • Compute “total_daily_equivalent_mg” = sum of dose_mg × normalization_factor(name) (where normalization_factor = 1.0 for Seroquel IR, 0.5 for Zoloft, 0.8 for Lamotrigine, 1.2 for Lithium, etc.). If any unknown drug appears, set “normalization_factor” = 1.0 by default.  \n\n'
            '4. **Output**: Only the following JSON keys (no extras):  \n'
            '   {  \n'
            '    "theta": <float, 4-dec>,  \n'
            '    "serotonin_idx": <float between 0–1, 4-dec>,  \n'
            '    "dopamine_idx": <float between 0–1, 4-dec>,  \n'
            '    "norepinephrine_idx": <float between 0–1, 4-dec>,  \n'
            '    "sleep_idx": <float between 0–1, 4-dec>,  \n'
            '    "risk_factor_mod": <float between 0–1, 2-dec>,  \n'
            '    "med_attention_flag": <boolean>,  \n'
            '    "validated_meds": [  \n'
            '       {"name":"Seroquel IR","dose_mg":200,"schedule":"once at bedtime"},  \n'
            '       {"name":"Zoloft","dose_mg":25,"schedule":"once daily"},  \n'
            '       …  \n'
            '    ],  \n'
            '    "total_daily_equivalent_mg": <integer>  \n'
            '   }  \n'
        )
        return json.loads(await self.ai.chat(prompt, 800))

    async def layer2_pkpd(self, vnorm: Dict[str, Any], age: int, sex: str, weight_kg: float, bmi: float) -> Dict[str, Any]:
        """
        Extended Pharmacokinetic/Pharmacodynamic Parameter Estimation Prompt
        """
        payload = {
            **vnorm,
            "age": age,
            "sex": sex,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "current": {med["name"].lower().replace(" ", "_"): med["dose_mg"] for med in vnorm["validated_meds"]}
        }
        prompt = (
            'INPUT: A JSON object containing all keys from the Layer 1 output (theta, serotonin_idx, dopamine_idx, norepinephrine_idx, sleep_idx, risk_factor_mod, validated_meds, total_daily_equivalent_mg), plus:  \n'
            '  • “age” (integer), “sex” (“M” or “F”), “weight_kg” (float), “bmi” (float).  \n'
            '  • “current”: an object with keys “sqr” (current Seroquel IR dose in mg), “zft” (current Zoloft dose in mg), plus any other psychiatric meds identified in “validated_meds” (e.g., “lamotrigine”: 100, “lithium”: 600).  \n\n'
            'RULES:  \n'
            '1. **General PK/PD Model Setup**:  \n'
            '   • Use a two-compartment, first-order absorption model for Seroquel IR with absorption rate constant ka_sqr = 1.1 h⁻¹, elimination half-life t½_sqr = 6 h.  \n'
            '   • For Zoloft, assume a one-compartment model with ka_zft = 0.8 h⁻¹, t½_zft = 24 h.  \n'
            '   • For each additional med in “validated_meds” (e.g., Lamotrigine, Lithium):  \n'
            '      – Assign ka_lamo = 1.3 h⁻¹, t½_lamo = 16 h; ka_lith = 0.5 h⁻¹, t½_lith = 24 h.  \n'
            '   • Compute clearance (CL) and volume of distribution (Vd) for each drug:  \n'
            '      – base_CL_sqr = 0.12 L/h/kg × weight_kg; Vd_sqr = 1.1 L/kg × weight_kg.  \n'
            '      – Scale CL_sqr by factor = 1 + (dopamine_idx – 0.5). If dopamine_idx < 0.5, reduce CL_sqr by (0.5 – dopamine_idx) × 0.2. Round both to 3 significant figures.  \n'
            '      – For Zoloft: base_CL_zft = 0.08 L/h/kg × weight_kg; Vd_zft = 0.8 L/kg × weight_kg. Scale CL_zft by factor = 1 + (serotonin_idx – 0.4). If serotonin_idx < 0.4, reduce CL_zft by (0.4 – serotonin_idx) × 0.15.  \n'
            '      – For other meds:  \n'
            '         • Lamotrigine CL_lamo = 0.05 L/h/kg × weight_kg; Vd_lamo = 0.9 L/kg × weight_kg; scale by factor = 1 + (norepinephrine_idx – 0.3).  \n'
            '         • Lithium CL_lith = 0.03 L/h/kg × weight_kg; Vd_lith = 0.7 L/kg × weight_kg; scale by factor = 1 + (sleep_idx – 0.2).  \n'
            '   • If “med_attention_flag” is true, multiply all CL_x and Vd_x by 1.05 to reflect potential nonadherence or inconsistent access.  \n'
            '   • Round all clearance (CL) and distribution volume (Vd) values to two decimal places.  \n\n'
            '2. **Allometric and Demographic Adjustments**:  \n'
            '   • If BMI > 30, increase Vd_sqr and Vd_lamo by +10%. If BMI < 18.5, decrease CL_zft and CL_lamo by −10%.  \n'
            '   • If age > 65, decrease CL_sqr and CL_zft by −15%. If age < 18, increase CL_sqr and CL_zft by +10%.  \n'
            '   • If sex = “F,” reduce CL_zft by an additional 5% due to lower average CYP450 metabolism.  \n\n'
            '3. **Drug-Drug Interaction Checks**:  \n'
            '   • If both Seroquel and Zoloft are present, apply an interaction coefficient:  \n'
            '      CL_sqr_adjusted = CL_sqr × (1 + 0.10 × (zft / 50)), rounding to two decimals.  \n'
            '   • If Lithium is in “validated_meds” and either Zoloft or Lamotrigine is present, flag “interaction_warning”: “Monitor Lithium levels closely.”  \n'
            '   • If total_daily_equivalent_mg > 400, insert “overload_flag”: true.  \n\n'
            '4. **Output**: Only the following JSON keys (no extras):  \n'
            '   {  \n'
            '     "sqr_CL": <float, L/h>,  \n'
            '     "sqr_Vd": <float, L>,  \n'
            '     "zft_CL": <float, L/h>,  \n'
            '     "zft_Vd": <float, L>,  \n'
            '     "<med>_CL": <float, L/h> for each additional med,  \n'
            '     "<med>_Vd": <float, L> for each additional med,  \n'
            '     "interaction_warning": <string> (only if triggered),  \n'
            '     "overload_flag": <boolean>  \n'
            '   }  \n'
        )
        return json.loads(await self.ai.chat(prompt, 800))

    async def layer3_sim(self, pkpd: Dict[str, Any], vnorm: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extended 45-Month Dose/Toxicity Simulation Prompt
        """
        # Construct the start object
        start_doses = {med["name"].lower().replace(" ", "_"): med["dose_mg"] for med in vnorm["validated_meds"]}
        sim_in = {
            **pkpd,
            "theta": vnorm["theta"],
            "risk_factor_mod": vnorm["risk_factor_mod"],
            "horizon_m": 45,
            "eval_days": 30,
            "start": start_doses
        }
        prompt = (
            'INPUT: A JSON object containing all keys from Layer 2 (CL and Vd for each drug, any “interaction_warning,” any “overload_flag”) plus:  \n'
            '  • “horizon_m”: total simulation horizon in months (integer, e.g., 45).  \n'
            '  • “eval_days”: evaluation interval in days (integer, e.g., 30).  \n'
            '  • “start”: an object mapping each drug to its starting dose in mg (e.g., {"sqr":200,"zft":25,"lamotrigine":100}).  \n'
            '  • “theta”: (from Layer 1).  \n'
            '  • “risk_factor_mod”: (from Layer 1).  \n\n'
            'RULES:  \n'
            '1. **Discrete PK Steps**:  \n'
            '   • Perform once‐daily PK calculations for each drug over the entire horizon of horizon_m × 30 days (round to nearest integer). For each day d:  \n'
            '      – Compute concentration C_d(i) for each drug i at day d using the standard first‐order absorption/elimination formula:  \n'
            '         • C_d(i) = (Dose_i × ka_i)/(Vd_i × (ka_i − ke_i)) × (e^(−ke_i × t_d) − e^(−ka_i × t_d)), where ke_i = ln(2)/t½_i, and t_d is the time since last dose (24 h).  \n'
            '   • At each evaluation day (days that are multiples of eval_days), compute Cmin_d(i) = minimum concentration for drug i over the preceding 24 hours, and define Cmin_target_i as the target minimum concentration:  \n'
            '      – Seroquel IR target Cmin_target_sqr = 5 ng/mL.  \n'
            '      – Zoloft target Cmin_target_zft = 20 ng/mL.  \n'
            '      – Lamotrigine target Cmin_target_lamo = 3 µg/mL.  \n'
            '      – Lithium target Cmin_target_lith = 0.8 mEq/L.  \n'
            '   • Compute risk_i(d) = ((|Cmin_target_i − Cmin_d(i)|)/Cmin_target_i)², clipped between 0 and 1 for each drug.  \n\n'
            '2. **Global Risk Calculation**:  \n'
            '   • At each evaluation day, compute daily_risk_d = weighted sum of individual risk_i(d), where weights = {Seroquel:0.4, Zoloft:0.3, Lamotrigine:0.2, Lithium:0.1}.  \n'
            '   • Compute quantum_intensity = q_intensity7(theta, (sleep_idx, risk_factor_mod)), rounding to 4 decimal places.  \n'
            '   • Define overall_risk_d = Clip(daily_risk_d + (1 − quantum_intensity) × 0.1, 0 to 1).  \n\n'
            '3. **Dose Adjustment Algorithm**:  \n'
            '   • At each evaluation day d:  \n'
            '      – If overall_risk_d > 0.25, increase Seroquel dose by +25 mg (unless current dose ≥ 400 mg, then keep at 400), and/or Zoloft dose by +12.5 mg (unless current dose ≥ 100 mg). For each additional med:  \n'
            '         • If risk_lamo(d) > 0.20, increase Lamotrigine by +25 mg (cap 200 mg).  \n'
            '         • If risk_lith(d) > 0.15, increase Lithium by +150 mg (cap 1200 mg).  \n'
            '      – If overall_risk_d < 0.10 for two consecutive eval_days, decrease Seroquel by −25 mg (minimum 50 mg) and/or Zoloft by −12.5 mg (minimum 25 mg).  \n'
            '      – Apply any “interaction_warning” logic: e.g., if Lithium + Zoloft concurrently at high doses, add “lith_zft_warning” to the timeline.  \n'
            '   • Record each adjustment event in the “timeline” array with keys:  \n'
            '      {"day": d, "sqr": current_sqr, "zft": current_zft, "lamotrigine": current_lamo, "lithium": current_lith, "daily_risk": daily_risk_d, "quantum_intensity": quantum_intensity, "overall_risk": overall_risk_d, "adjustment": "<description>"}  \n\n'
            '4. **Safety and Cap Checks**:  \n'
            '   • If “overload_flag” = true (from Layer 2), set overall_risk_d = 1.0 for all days, do not escalate any dose, and set “override_action”: “Consult physician—patient overloaded.”  \n'
            '   • If any dose would exceed the drug caps (Seroquel 400 mg, Zoloft 100 mg, Lamotrigine 200 mg, Lithium 1200 mg), do not adjust upward; mark “cap_reached”: true.  \n'
            '   • Compute “expected_adherence_factor” each evaluation day = 1 − risk_factor_mod. If expected_adherence_factor < 0.6, add “adherence_warning” to adjustment description.  \n\n'
            '5. **Output**: Only the following JSON keys (no extras):  \n'
            '   {  \n'
            '     "timeline": [  \n'
            '       {  \n'
            '         "day": <int>,  \n'
            '         "sqr": <int>,  \n'
            '         "zft": <int>,  \n'
            '         "lamotrigine": <int>,  \n'
            '         "lithium": <int>,  \n'
            '         "daily_risk": <float 4-dec>,  \n'
            '         "quantum_intensity": <float 4-dec>,  \n'
            '         "overall_risk": <float 4-dec>,  \n'
            '         "adjustment": <string>,  \n'
            '         "cap_reached": <boolean, optional>,  \n'
            '         "override_action": <string, optional>,  \n'
            '         "adherence_warning": <string, optional>  \n'
            '       },  \n'
            '       …  \n'
            '     ],  \n'
            '     "final_doses": {"sqr": <int>,"zft": <int>,"lamotrigine": <int>,"lithium": <int>},  \n'
            '     "peak_overall_risk": <float 4-dec>,  \n'
            '     "lowest_overall_risk": <float 4-dec>  \n'
            '   }  \n'
        )
        return json.loads(await self.ai.chat(prompt, 1200))

    async def layer4_plan(self, sim: Dict[str, Any], vnorm: Dict[str, Any], pkpd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extended Optimal Regimen & Review Cadence Prompt
        """
        payload = {
            **sim,
            "interaction_warning": pkpd.get("interaction_warning"),
            "overload_flag": pkpd.get("overload_flag", False)
        }
        prompt = (
            'INPUT: A JSON object containing all keys from Layer 3 (timeline array, final_doses, peak_overall_risk, lowest_overall_risk), plus:  \n'
            '  • “interaction_warning”: optional string if Lithium-Zoloft interaction was flagged.  \n'
            '  • “overload_flag”: boolean from Layer 2.  \n\n'
            'RULES:  \n'
            '1. **Identify Stable Low-Risk Window**:  \n'
            '   • Scan the “timeline” array for the first contiguous period of at least 60 days (i.e., two consecutive evaluation days × eval_days apart) where “overall_risk” < 0.05 for every day in that span.  \n'
            '   • If found, denote “window_start_day” = the start day of that 60-day span, “window_end_day” = start_day + 59. Otherwise, set “window_start_day” = null and “plan_status” = “No stable window found within horizon.”  \n\n'
            '2. **Determine Optimal Regimen**:  \n'
            '   • If “window_start_day” is not null, let the “optimal_doses” equal the doses in “timeline” at day = (window_start_day – eval_days). If window_start_day = 0, use “start” doses from Layer 3’s input.  \n'
            '   • If “overload_flag” = true, set “optimal_doses” = {all drugs: 0} and “plan_status” = “Stop all meds—patient overloaded; consult rapidly.”  \n\n'
            '3. **Review Frequency Calculation**:  \n'
            '   • Calculate “average_risk_post_stable” = average “overall_risk” for 90 days following window_end_day (if window exists).  \n'
            '   • If average_risk_post_stable < 0.02, set “review_days” = 45;  \n'
            '      If between 0.02 and 0.05, set “review_days” = 30;  \n'
            '      If between 0.05 and 0.10, set “review_days” = 14;  \n'
            '      If > 0.10, set “review_days” = 7.  \n'
            '   • If no stable window found, set “review_days” = 7 and “plan_status” = “Frequent review until stable window achieved.”  \n\n'
            '4. **Risk Floor & Ceiling**:  \n'
            '   • “risk_floor” = minimum “overall_risk” observed in the last 180 days of timeline;  \n'
            '   • “risk_ceiling” = maximum “overall_risk” observed in the last 180 days;  \n'
            '   • Round both to 4 decimal places.  \n\n'
            '5. **Include Interaction & Overload Notes**:  \n'
            '   • If “interaction_warning” is present, append to “plan_status”: “ Note: {interaction_warning}.”  \n'
            '   • If “overload_flag” is true, override “plan_status” to “Overload—stop meds.”  \n\n'
            '6. **Output**: Only the following JSON keys (no extras):  \n'
            '   {  \n'
            '     "optimal": {  \n'
            '       "sqr": <int>,  \n'
            '       "zft": <int>,  \n'
            '       "lamotrigine": <int (if present)>,  \n'
            '       "lithium": <int (if present)>  \n'
            '     },  \n'
            '     "review_days": <int>,  \n'
            '     "risk_floor": <float 4-dec>,  \n'
            '     "risk_ceiling": <float 4-dec>,  \n'
            '     "window_start_day": <int or null>,  \n'
            '     "window_end_day": <int or null>,  \n'
            '     "plan_status": <string>  \n'
            '   }  \n'
        )
        return json.loads(await self.ai.chat(prompt, 800))

    async def layer5_summary(self, plan: Dict[str, Any], vnorm: Dict[str, Any], pkpd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extended Plain-English Recommendation Prompt
        """
        payload = plan
        # Include overload_flag if present
        if pkpd.get("overload_flag", False):
            payload["overload_flag"] = True

        prompt = (
            'INPUT: A JSON object containing all keys from Layer 4 (optimal doses, review_days, risk_floor, risk_ceiling, window_start_day, window_end_day, plan_status).  \n\n'
            'RULES:  \n'
            '1. Write in plain English (≤300 words), structured in short paragraphs (2–3 sentences each).  \n'
            '2. Begin with a concise statement: “Based on your 45-month simulation, here is your personalized regimen and review plan:”  \n'
            '3. List each medication with its optimal daily dose and schedule clearly, e.g.:  \n'
            '   • “Seroquel IR: 175 mg once at bedtime”  \n'
            '   • “Zoloft: 37.5 mg once daily”  \n'
            '   • If Lamotrigine present: “Lamotrigine: 125 mg twice daily”  \n'
            '   • If Lithium present: “Lithium: 900 mg once daily”  \n'
            '4. Explain the chosen review frequency: “You should have follow-up evaluations every {review_days} days to ensure your overall risk remains below {risk_floor * 100}%.”  \n'
            '5. If a stable low-risk window was found (window_start_day ≠ null), say: “Your first stable window begins around day {window_start_day}. During that time, your overall risk is predicted to stay below {risk_floor}.”  \n'
            '   If no stable window: “No fully stable 60-day window was found; we recommend more frequent follow-up as indicated.”  \n'
            '6. Comment briefly on risk floor/ceiling: “Your projected lowest overall risk is {risk_floor}. The highest projected risk during the horizon is {risk_ceiling}.”  \n'
            '7. If “plan_status” mentions “interaction_warning,” add: “Note: {interaction_warning}.”  \n'
            '   If “overload_flag” = true, override text to: “Warning: Simulation indicates medication overload. Discontinue medications immediately and consult your physician.”  \n'
            '8. Close with an encouraging statement: “Please discuss this regimen with your healthcare provider before making changes.…”  \n\n'
            'OUTPUT: Only one key:  \n'
            '  {"summary": "<full text, ≤300 words>"}  \n'
        )
        return json.loads(await self.ai.chat(prompt, 800))

    async def optimize(
        self,
        vec: List[float],
        qid: str,
        age: int,
        sex: str,
        weight_kg: float,
        bmi: float,
        vision_desc: Optional[str],
        current_meds: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        vnorm = await self.layer1_vnorm(vec, qid, age, sex, weight_kg, bmi, vision_desc, current_meds)
        pkpd = await self.layer2_pkpd(vnorm, age, sex, weight_kg, bmi)
        sim = await self.layer3_sim(pkpd, vnorm)
        plan = await self.layer4_plan(sim, vnorm, pkpd)
        summary = await self.layer5_summary(plan, vnorm, pkpd)
        return {
            "v_norm": vnorm,
            "pkpd": pkpd,
            "sim": sim,
            "plan": plan,
            "summary": summary
        }

# ════════════════════════════════════════════════════════════════
# TKINTER GUI

class QRxSynthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QRx-Synth – Quantum Dose Optimizer (Seroquel/Zoloft)")
        self.geometry("980x880")

        # Initialize encryption and settings
        self.crypto = AESGCMCrypto(MASTER_KEY)
        self.settings = Settings.load(self.crypto)
        if not os.path.exists(SETTINGS_FILE):
            self.settings.prompt_gui()
            self.settings.save(self.crypto)
        if not self.settings.api_key:
            mb.showerror("Missing API Key", "Please set your OpenAI key in Settings.")
            self.destroy()
            return

        # Status label
        self.status = tk.StringVar(value="Ready.")
        tk.Label(self, textvariable=self.status, font=("Helvetica", 15)).pack(pady=6)

        # Patient Info frame
        env = tk.LabelFrame(self, text="Patient / Dose Settings")
        env.pack(fill="x", padx=10, pady=4)
        self.username  = tk.StringVar(value=self.settings.username)
        self.age       = tk.IntVar(value=self.settings.age)
        self.sex       = tk.StringVar(value=self.settings.sex)
        self.weight_kg = tk.DoubleVar(value=self.settings.weight_kg)
        self.bmi       = tk.DoubleVar(value=self.settings.bmi)
        self.camera_idx = tk.IntVar(value=self.settings.camera_idx)

        # Medication list frame
        meds_frame = tk.LabelFrame(self, text="Current Medications")
        meds_frame.pack(fill="x", padx=10, pady=4)
        self.med_entries: List[Tuple[tk.Entry, tk.Entry, tk.Entry]] = []
        tk.Button(meds_frame, text="Add Med", command=lambda: self.add_med_row(meds_frame)).pack(side="left", padx=5)
        tk.Button(meds_frame, text="Remove Med", command=lambda: self.remove_med_row(meds_frame)).pack(side="left", padx=5)
        self.add_med_row(meds_frame)  # start with one row

        def row(lbl, var, col):
            tk.Label(env, text=lbl).grid(row=0, column=col*2, sticky="e", padx=3)
            tk.Entry(env, textvariable=var, width=11).grid(row=0, column=col*2+1, sticky="w")

        row("Username", self.username, 0)
        row("Age", self.age, 1)
        row("Sex", self.sex, 2)
        row("Weight kg", self.weight_kg, 3)
        row("BMI", self.bmi, 4)
        row("Cam idx", self.camera_idx, 5)

        # Buttons
        btns = tk.Frame(self)
        btns.pack(pady=6)
        tk.Button(btns, text="Settings", command=self.open_settings).grid(row=0, column=0, padx=5)
        tk.Button(btns, text="View Dose Reports", command=self.view_reports).grid(row=0, column=1, padx=5)
        tk.Button(btns, text="Run Dose Optimizer", command=self.run_dose_opt).grid(row=0, column=2, padx=5)
        tk.Button(btns, text="Run from Webcam", command=self.run_cam_opt).grid(row=0, column=3, padx=5)

        # Output log
        self.text = tk.Text(self, height=24, width=120, wrap="word")
        self.text.pack(padx=8, pady=6)

        # Initialize DB and AI
        self.db = ReportDB(self.settings.db_path, self.crypto)
        self.ai = OpenAIClient(api_key=self.settings.api_key)
        self.qrx = QRxSynth(self.ai)

        # Start an asyncio loop in a background thread
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def add_med_row(self, parent_frame: tk.Frame) -> None:
        row_frame = tk.Frame(parent_frame)
        row_frame.pack(fill="x", padx=5, pady=2)
        name_var = tk.StringVar(value="")
        dose_var = tk.StringVar(value="")
        sched_var = tk.StringVar(value="")
        tk.Entry(row_frame, textvariable=name_var, width=20).pack(side="left", padx=5)
        tk.Entry(row_frame, textvariable=dose_var, width=8).pack(side="left", padx=5)
        tk.Entry(row_frame, textvariable=sched_var, width=15).pack(side="left", padx=5)
        self.med_entries.append((name_var, dose_var, sched_var))

    def remove_med_row(self, parent_frame: tk.Frame) -> None:
        if not self.med_entries:
            return
        name_var, dose_var, sched_var = self.med_entries.pop()
        # find and destroy the corresponding row frame
        for child in parent_frame.winfo_children():
            entries = child.winfo_children()
            if len(entries) == 3:
                if (isinstance(entries[0], tk.Entry) and
                    entries[0].get() == name_var.get() and
                    entries[1].get() == dose_var.get() and
                    entries[2].get() == sched_var.get()):
                    child.destroy()
                    break

    def collect_current_meds(self) -> List[Dict[str, Any]]:
        meds_list = []
        for name_var, dose_var, sched_var in self.med_entries:
            name = name_var.get().strip()
            try:
                dose = int(dose_var.get().strip())
            except:
                dose = 0
            sched = sched_var.get().strip()
            if name and dose > 0 and sched:
                meds_list.append({"name": name, "dose_mg": dose, "schedule": sched})
        return meds_list

    def open_settings(self) -> None:
        self.settings.prompt_gui()
        self.settings.save(self.crypto)
        mb.showinfo("Settings", "Saved. Restart to apply changes.")

    def view_reports(self) -> None:
        rows = asyncio.run(self.db.list_reports())
        if not rows:
            mb.showinfo("Dose Reports", "No dose optimization reports stored.")
            return
        opts = "\n".join(
            f"{rid} – {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"
            for rid, ts in rows[:30]
        )
        sel = sd.askstring("Select Report ID", opts)
        self.text.delete("1.0", tk.END)
        if sel:
            try:
                rid = int(sel.split()[0])
                rpt = asyncio.run(self.db.load(rid))
                self.text.insert(tk.END, json.dumps(rpt, indent=2))
            except Exception as e:
                mb.showerror("Error", str(e))

    def run_dose_opt(self) -> None:
        """Manual run using a random BioVector (no vision)."""
        vec = list(np.random.uniform(0, 1, 25))
        qid = hashlib.sha256(json.dumps(vec).encode()).hexdigest()[:16]
        current_meds = self.collect_current_meds()
        self.status.set("Running QRx-Synth Dose Optimization…")

        async def go():
            await self.db.init()
            result = await self.qrx.optimize(
                vec=vec,
                qid=qid,
                age=self.age.get(),
                sex=self.sex.get(),
                weight_kg=self.weight_kg.get(),
                bmi=self.bmi.get(),
                vision_desc=None,
                current_meds=current_meds
            )
            await self.db.save(time.time(), result, typ="dose_opt")
            self.status.set("Optimization complete.")
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, json.dumps(result, indent=2))
            if "summary" in result and "summary" in result["summary"]:
                mb.showinfo("Summary", result["summary"]["summary"])
            await self.db.close()

        asyncio.run_coroutine_threadsafe(go(), self.loop)

    def run_cam_opt(self) -> None:
        """Capture a frame, run GPT-4o Vision + BioVector, then optimize."""
        self.status.set("Capturing BioVector from camera…")
        cap = cv2.VideoCapture(self.camera_idx.get(), cv2.CAP_ANY)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            mb.showerror("Camera Error", "Could not capture from camera.")
            self.status.set("Camera error.")
            return

        # 1. Extract BioVector
        vec = BioVector.from_frame(frame).arr.tolist()
        qid = hashlib.sha256(json.dumps(vec).encode()).hexdigest()[:16]
        current_meds = self.collect_current_meds()

        # 2. Convert frame to JPEG in memory, then Base64
        success, jpeg = cv2.imencode(".jpg", frame)
        if not success:
            mb.showerror("Error", "Could not encode camera frame.")
            self.status.set("Encoding error.")
            return
        img_bytes = jpeg.tobytes()

        # 3. Send to GPT-4o Vision for a brief description
        self.status.set("Running GPT-4o Vision analysis…")

        async def go():
            try:
                vision_desc = await self.ai.chat_with_image(
                    image_data=img_bytes,
                    prompt_text="Please give me a brief description of what you see in this image (≤50 words).",
                    max_tokens=100
                )
            except Exception as e:
                LOGGER.error("Vision API error: %s", e)
                vision_desc = None

            # 4. Now run the rest of the pipeline
            self.status.set("Running QRx-Synth Dose Optimization…")
            await self.db.init()
            result = await self.qrx.optimize(
                vec=vec,
                qid=qid,
                age=self.age.get(),
                sex=self.sex.get(),
                weight_kg=self.weight_kg.get(),
                bmi=self.bmi.get(),
                vision_desc=vision_desc,
                current_meds=current_meds
            )
            await self.db.save(time.time(), result, typ="dose_opt")
            self.status.set("Optimization complete.")
            self.text.delete("1.0", tk.END)
            full_output = {"vision_desc": vision_desc, **result}
            self.text.insert(tk.END, json.dumps(full_output, indent=2))
            if "summary" in result and "summary" in result["summary"]:
                mb.showinfo("Summary", result["summary"]["summary"])
            await self.db.close()

        asyncio.run_coroutine_threadsafe(go(), self.loop)

    def on_close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.destroy()

# ════════════════════════════════════════════════════════════════
# MAIN

if __name__ == "__main__":
    try:
        QRxSynthApp().mainloop()
    except KeyboardInterrupt:
        LOGGER.info("Exiting QRx-Synth.")

# ════════════════════════════════════════════════════════════════
# UNIT TESTS (OPTIONAL)

if __name__ == "__qtest__":
    print("Running integrity checks…")
    c = AESGCMCrypto(MASTER_KEY)
    sample = b"hello"
    assert c.decrypt(c.encrypt(sample)) == sample
    print("AESGCM ✅")

    bv = BioVector.from_frame(np.zeros((64, 64, 3), np.uint8))
    print("BioVector len:", len(bv.arr))

    q = q_intensity7(1.2, (0.5, 0.1))
    print("q_exp7:", q)
    print("✔︎ All quick tests passed.")
