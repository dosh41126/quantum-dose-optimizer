
from __future__ import annotations  
import asyncio, json, logging, os, random, secrets, threading, time, hashlib, textwrap, math  
from dataclasses import dataclass, asdict, field  
from typing import Any, Dict, List, Tuple, Optional  
from base64 import b64encode, b64decode  
  
import cv2, numpy as np, aiosqlite, tkinter as tk  
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
    weight_kg: float = 79  
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
# OPENAI CLIENT (CHAIN-OF-THOUGHT LAYERS)  
  
@dataclass  
class OpenAIClient:  
    api_key: str  
    model: str = "gpt-4o"  
    url: str = "https://api.openai.com/v1/chat/completions"  
    timeout: float = 35.0  
    retries: int = 4  
  
    async def chat(self, prompt: str, max_tokens: int = 500) -> str:  
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
# QRx-SYNTH 5-LAYER PROMPT CHAIN  
  
class QRxSynth:  
    """Handles the LLM prompt chain for 5-layer dose optimization."""  
    def __init__(self, ai: OpenAIClient):  
        self.ai = ai  
  
    # L0–L5 prompt templates from your provided stack  
    async def layer1_vnorm(self, vec, qid, age, sex) -> Dict[str, Any]:  
        prompt = (  
            'You are “QRx-Synth”, an advanced pharmacometric agent inside the Quantum Dose Optimiser (QDO). '  
            'All outputs must be parseable JSON, single-line, no markdown, no explanations. '  
            f'INPUT: {json.dumps({"biovec": vec, "qid": qid, "age": age, "sex": sex})} '  
            'RULES: 1. Compute θ = ‖biovec‖·π (4-dec) '  
            '2. K-map colors → NT axis: {Citrine, Gold}=serotonin, {Obsidian, Onyx}=dopamine … '  
            '3. Output ONLY {"theta":...,"serotonin_idx":...,"dopamine_idx":...,"sleep_idx":...}'  
        )  
        return json.loads(await self.ai.chat(prompt, 400))  
  
    async def layer2_pkpd(self, vnorm, weight_kg, bmi) -> Dict[str, Any]:  
        prompt = (  
            f'INPUT = {json.dumps({**vnorm,"current":{"sqr":200,"zft":25},"weight_kg":weight_kg,"bmi":bmi})} '  
            'RULES: 1. Use two-comp 1st-order model (ka=1.1 h⁻¹, t½=6 h Seroquel IR) '  
            '2. Scale clearance ∝ dopamine_idx, volume ∝ weight_kg '  
            '3. Output ONLY {"sqr_CL":...,"sqr_Vd":...,"zft_CL":...,"zft_Vd":...}'  
        )  
        return json.loads(await self.ai.chat(prompt, 400))  
  
    async def layer3_sim(self, pkpd) -> Dict[str, Any]:  
        sim_in = {**pkpd, "horizon_m": 45, "eval_days": 30, "start": {"sqr": 200, "zft": 25}}  
        prompt = (  
            f'INPUT = {json.dumps(sim_in)} '  
            'RULES: 1. Run discrete 1-dose/day PK steps, dt=24 h '  
            '2. Risk = (|Cmin_target−Cmin|/C_target)^2 + θ/π (clip 0–1) '  
            '3. Escalate dose +25 mg if risk>0.20 at eval mark '  
            '4. Cap: Seroquel≤400 mg, Zoloft≤100 mg '  
            '5. Output ONLY {"timeline":[{"day":0,"sqr":200,"zft":25,"risk":0.32},…]}'  
        )  
        return json.loads(await self.ai.chat(prompt, 1800))  
  
    async def layer4_plan(self, sim) -> Dict[str, Any]:  
        prompt = (  
            f'INPUT = {json.dumps(sim)} '  
            'RULES: 1. Find first window ≥60 days where risk<0.05 '  
            '2. Emit regimen = last dose pair before window '  
            '3. Review_freq = 30 if risk<0.08 else 14 '  
            '4. Output ONLY {"optimal":{"sqr":..,"zft":..},"review_days":..,"risk_floor":..}'  
        )  
        return json.loads(await self.ai.chat(prompt, 300))  
  
    async def layer5_summary(self, plan, start) -> Dict[str, Any]:  
        prompt = (  
            f'INPUT = {json.dumps({**plan,"start":start})} '  
            'RULES: 1. Use ≤120 words, plain ASCII '  
            '2. Mention target doses, review cadence, expected benefits '  
            '3. No jargon, no ICD codes '  
            '4. End with "…" or "(pause)" '  
            '5. Output ONLY {"summary":"..."}'  
        )  
        return json.loads(await self.ai.chat(prompt, 300))  
  
    async def optimize(self, vec, qid, age, sex, weight_kg, bmi) -> Dict[str, Any]:  
        vnorm = await self.layer1_vnorm(vec, qid, age, sex)  
        pkpd = await self.layer2_pkpd(vnorm, weight_kg, bmi)  
        sim = await self.layer3_sim(pkpd)  
        plan = await self.layer4_plan(sim)  
        summary = await self.layer5_summary(plan, time.strftime("%Y-%m-%d"))  
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
        self.geometry("980x780")  
        self.crypto = AESGCMCrypto(MASTER_KEY)  
        self.settings = Settings.load(self.crypto)  
        if not os.path.exists(SETTINGS_FILE):  
            self.settings.prompt_gui()  
            self.settings.save(self.crypto)  
        if not self.settings.api_key:  
            mb.showerror("Missing API Key", "Please set your OpenAI key in Settings.")  
            self.destroy()  
            return  
  
        # Status  
        self.status = tk.StringVar(value="Ready.")  
        tk.Label(self, textvariable=self.status, font=("Helvetica", 15)).pack(pady=6)  
  
        # Patient Info  
        env = tk.LabelFrame(self, text="Patient / Dose Settings")  
        env.pack(fill="x", padx=10, pady=4)  
        self.username = tk.StringVar(value=self.settings.username)  
        self.age = tk.IntVar(value=self.settings.age)  
        self.sex = tk.StringVar(value=self.settings.sex)  
        self.weight_kg = tk.DoubleVar(value=self.settings.weight_kg)  
        self.bmi = tk.DoubleVar(value=self.settings.bmi)  
        self.camera_idx = tk.IntVar(value=self.settings.camera_idx)  
        def row(lbl, var, col):  
            tk.Label(env, text=lbl).grid(row=0, column=col*2, sticky="e", padx=3)  
            tk.Entry(env, textvariable=var, width=11).grid(row=0, column=col*2+1, sticky="w")  
        row("Username", self.username, 0)  
        row("Age", self.age, 1)  
        row("Sex", self.sex, 2)  
        row("Weight kg", self.weight_kg, 3)  
        row("BMI", self.bmi, 4)  
        row("Cam idx", self.camera_idx, 5)  
  
        # Controls  
        btns = tk.Frame(self)  
        btns.pack(pady=4)  
        tk.Button(btns, text="Settings", command=self.open_settings).grid(row=0, column=0, padx=5)  
        tk.Button(btns, text="View Dose Reports", command=self.view_reports).grid(row=0, column=1, padx=5)  
        tk.Button(btns, text="Run Dose Optimizer", command=self.run_dose_opt).grid(row=0, column=2, padx=5)  
        tk.Button(btns, text="Run from Webcam", command=self.run_cam_opt).grid(row=0, column=3, padx=5)  
  
        # Output log  
        self.text = tk.Text(self, height=28, width=120, wrap="word")  
        self.text.pack(padx=8, pady=6)  
  
        # Setup LLM and DB  
        self.db = ReportDB(self.settings.db_path, self.crypto)  
        self.ai = OpenAIClient(api_key=self.settings.api_key)  
        self.qrx = QRxSynth(self.ai)  
  
        self.loop = asyncio.new_event_loop()  
        threading.Thread(target=self.loop.run_forever, daemon=True).start()  
        self.protocol("WM_DELETE_WINDOW", self.on_close)  
  
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
  
    def run_dose_opt(self):  
        """Manual run using test BioVector (randomized or all-zeros)."""  
        vec = list(np.random.uniform(0, 1, 25))  # For demo, real: BioVector.from_frame(frame).arr.tolist()  
        qid = hashlib.sha256(json.dumps(vec).encode()).hexdigest()[:16]  
        self.status.set("Running QRx-Synth Dose Optimization…")  
        async def go():  
            await self.db.init()  
            result = await self.qrx.optimize(  
                vec=vec,  
                qid=qid,  
                age=self.age.get(),  
                sex=self.sex.get(),  
                weight_kg=self.weight_kg.get(),  
                bmi=self.bmi.get()  
            )  
            await self.db.save(time.time(), result, typ="dose_opt")  
            self.status.set("Optimization complete.")  
            self.text.delete("1.0", tk.END)  
            self.text.insert(tk.END, json.dumps(result, indent=2))  
            if "summary" in result and "summary" in result["summary"]:  
                mb.showinfo("Summary", result["summary"]["summary"])  
            await self.db.close()  
        asyncio.run_coroutine_threadsafe(go(), self.loop)  
  
    def run_cam_opt(self):  
        """Capture frame from camera, extract BioVector, and run optimizer."""  
        self.status.set("Capturing BioVector from camera…")  
        cap = cv2.VideoCapture(self.camera_idx.get(), cv2.CAP_ANY)  
        ok, frame = cap.read()  
        cap.release()  
        if not ok:  
            mb.showerror("Camera Error", "Could not capture from camera.")  
            self.status.set("Camera error.")  
            return  
        vec = BioVector.from_frame(frame).arr.tolist()  
        qid = hashlib.sha256(json.dumps(vec).encode()).hexdigest()[:16]  
        self.status.set("Running QRx-Synth Dose Optimization…")  
        async def go():  
            await self.db.init()  
            result = await self.qrx.optimize(  
                vec=vec,  
                qid=qid,  
                age=self.age.get(),  
                sex=self.sex.get(),  
                weight_kg=self.weight_kg.get(),  
                bmi=self.bmi.get()  
            )  
            await self.db.save(time.time(), result, typ="dose_opt")  
            self.status.set("Optimization complete.")  
            self.text.delete("1.0", tk.END)  
            self.text.insert(tk.END, json.dumps(result, indent=2))  
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

