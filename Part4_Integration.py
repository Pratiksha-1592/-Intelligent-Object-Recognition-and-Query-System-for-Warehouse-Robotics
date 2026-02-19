# ============================================================
# PART 4: FULL INTEGRATION — CV + ML + RAG Pipeline
# AI Research Intern - Technical Assessment
# Compatible with: Google Colab (CPU/GPU)
#
# Workflow:
#   Image → OpenCV Detection → ML Classification → RAG Answer
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Install All Dependencies
# ─────────────────────────────────────────────────────────────

!pip install opencv-python-headless torch torchvision \
             sentence-transformers faiss-cpu \
             scikit-learn matplotlib seaborn numpy Pillow -q

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw
import faiss, json, os, re, time, textwrap
from collections import defaultdict

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSES       = ['FRAGILE', 'HEAVY', 'HAZARDOUS']
CLASS_COLORS  = {'FRAGILE': '#E74C3C', 'HEAVY': '#3498DB', 'HAZARDOUS': '#F39C12'}
CV_COLORS     = {'FRAGILE': (50, 50, 220), 'HEAVY': (200, 130, 0), 'HAZARDOUS': (0, 180, 220)}

print(f"✅ All libraries loaded  |  Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# CELL 2: Re-build CV Module (from Part 1)
# ─────────────────────────────────────────────────────────────

class WarehouseObjectDetector:
    """
    Warehouse object detector using edge detection + contour analysis.
    Returns bounding boxes, dimensions, and center coordinates.
    """
    def __init__(self, min_contour_area=1500):
        self.min_contour_area = min_contour_area

    def preprocess(self, img):
        blurred  = cv2.GaussianBlur(img, (5, 5), 0)
        gray     = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return blurred, gray, enhanced

    def detect(self, img):
        _, gray, enhanced = self.preprocess(img)
        thresh, _  = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges      = cv2.Canny(enhanced, thresh * 0.5, thresh)
        kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed     = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated    = cv2.dilate(closed, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid      = sorted([c for c in contours if cv2.contourArea(c) > self.min_contour_area],
                            key=cv2.contourArea, reverse=True)[:6]
        detections = []
        for c in valid:
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2
            detections.append({
                'bbox':   (x, y, w, h),
                'center': (cx, cy),
                'area':   int(cv2.contourArea(c)),
                'roi':    img[max(0,y):y+h, max(0,x):x+w],
            })
        return detections, edges

print("✅ WarehouseObjectDetector ready")


# ─────────────────────────────────────────────────────────────
# CELL 3: Re-build ML Classifier (from Part 2)
# ─────────────────────────────────────────────────────────────

class WarehouseClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        backbone       = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone  = backbone.features
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256,   64), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.gap(self.backbone(x)))


class MLClassifier:
    """
    Wrapper around WarehouseClassifier for single-image inference.
    Falls back to colour-heuristic if model weights unavailable.
    """
    def __init__(self, model_path=None):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.model = WarehouseClassifier(num_classes=3).to(DEVICE)

        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"   ✅ Loaded weights from {model_path}")
        else:
            # Train a quick version for integration demo
            print("   ⚡ No saved weights found — training quick demo model...")
            self._quick_train()

        self.model.eval()

    def _quick_train(self):
        """15-epoch quick train on synthetic data for integration demo."""
        from torchvision import transforms as T
        from torch.utils.data import Dataset, DataLoader

        class QuickDS(Dataset):
            def __init__(self, n=600):
                self.rng = np.random.RandomState(42)
                self.data, self.labels = [], []
                for cls_idx in range(3):
                    for i in range(n // 3):
                        img = self._make(cls_idx, i)
                        self.data.append(img)
                        self.labels.append(cls_idx)
            def _make(self, cls, idx):
                c = [(220,210,210),(60,55,55),(45,45,45)][cls]
                img = Image.new('RGB',(64,64), c)
                d   = ImageDraw.Draw(img)
                r   = self.rng
                if cls==0:
                    d.rectangle([10,10,54,54], fill=(195,160,115), outline=(140,100,65), width=2)
                    d.rectangle([14,22,50,35], fill=(210,40,40))
                elif cls==1:
                    d.rectangle([6,8,58,56],  fill=(90,60,30),  outline=(55,35,15), width=3)
                    for y in range(18,56,10): d.line([6,y,58,y],fill=(70,45,20),width=2)
                else:
                    cx=32
                    d.ellipse([cx-16,6,cx+16,20],  fill=(210,175,0), outline=(160,130,0), width=2)
                    d.rectangle([cx-16,13,cx+16,52],fill=(210,175,0),outline=(160,130,0),width=2)
                    d.rectangle([cx-10,22,cx+10,40],fill=(0,0,0))
                    d.text((cx-4,24),"!",fill=(255,220,0))
                arr = np.array(img).astype(np.float32)
                arr += np.random.normal(0,6,arr.shape)
                return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
            def __len__(self): return len(self.data)
            def __getitem__(self, i):
                tf = T.Compose([T.Resize((64,64)),T.ToTensor(),
                                T.Normalize([.485,.456,.406],[.229,.224,.225])])
                return tf(self.data[i]), self.labels[i]

        ds     = QuickDS(n=900)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        opt    = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        crit   = nn.CrossEntropyLoss()
        self.model.train()
        for ep in range(15):
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                loss = crit(self.model(imgs), labels)
                loss.backward()
                opt.step()
            if (ep+1) % 5 == 0:
                print(f"      Epoch {ep+1}/15  loss={loss.item():.3f}")
        self.model.eval()
        print("   ✅ Quick training complete")

    @torch.no_grad()
    def predict(self, roi_bgr):
        """Predict class from a BGR numpy ROI."""
        if roi_bgr is None or roi_bgr.size == 0:
            return 'FRAGILE', 0.33, np.array([0.33,0.33,0.34])
        pil_img = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        tensor  = self.transform(pil_img).unsqueeze(0).to(DEVICE)
        logits  = self.model(tensor)
        probs   = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred    = int(np.argmax(probs))
        return CLASSES[pred], float(probs[pred]), probs

print("🔧 Building ML Classifier...")
ml_classifier = MLClassifier(model_path='/content/results/warehouse_classifier.pth')
print("✅ ML Classifier ready")


# ─────────────────────────────────────────────────────────────
# CELL 4: Re-build RAG System (from Part 3)
# ─────────────────────────────────────────────────────────────

# ── Compact knowledge base (key entries for integration) ──
KB_COMPACT = [
    {"id":"DOC-001","title":"Fragile Item Handling",   "category":"Handling",
     "content":"""Fragile Item Handling Protocol.
Use soft-padded foam gripper GRP-SOFT-200. Set maximum grip force to 15 N.
Enable tactile feedback mode with threshold 0.2 N/mm².
Approach speed: maximum 0.05 m/s during contact. Lift at 0.08 m/s maximum.
Do NOT stack fragile items. Robot travel speed must not exceed 5 km/h.
Place items at 0.03 m/s — never drop or release abruptly.
Always confirm surface stability before releasing grip.
Log every fragile handling event in the WMS."""},

    {"id":"DOC-002","title":"Heavy Item Handling",     "category":"Handling",
     "content":"""Heavy Item Handling — Weight Limits.
Standard Gripper GRP-STD-100: maximum 25 kg.
Reinforced Gripper GRP-REF-300: maximum 80 kg.
Fork Attachment FRK-HVY-500: maximum 150 kg.
Items over 150 kg require manual forklift — robot must NOT attempt lift.
Travel speed limited to 3 km/h for items over 30 kg.
Heaviest items must always be placed on the bottom of any stack.
Never carry heavy items above conveyor height of 1.0 m."""},

    {"id":"DOC-003","title":"Hazardous Material Handling","category":"Handling",
     "content":"""Hazardous Material Handling Protocol.
Confirm HAZMAT_CLEAR signal in WMS before starting.
Verify containment integrity: no visible leaks, dents, or label damage.
Activate chemical-resistant gripper coating using spray nozzle A3.
Switch zone ventilation to maximum fan speed HIGH.
Alert human supervisor via Red flashing LED beacon.
Maximum speed: 2 km/h for Class 1 and 3; 1 km/h for Class 2 corrosives.
Keep item perfectly level — tilt sensors must stay below 5 degrees.
After handling: decontaminate gripper for 30 seconds with solvent nozzle B1.
Log hazmat event with timestamp, material class, and route."""},

    {"id":"DOC-005","title":"Pre-Operation Safety Checklist","category":"Safety",
     "content":"""Pre-Operation Safety Checklist — must complete before every shift.
Inspect all joint seals for leaks or cracking.
Test emergency stop button — must halt arm within 0.3 seconds.
LiDAR self-test: range accuracy must be within 2 cm at 5 m distance.
Camera calibration: run checkerboard test, verify under 1 px reprojection error.
Confirm WMS connection with ping under 50 ms.
Run diagnostic routine DIAG-001: all indicators must be green.
Ensure floor is clear of obstacles and spills in robot travel zone.
Supervisor must digitally confirm checklist before robot activation."""},

    {"id":"DOC-006","title":"Emergency Stop Procedures","category":"Safety",
     "content":"""Emergency Stop and Incident Response.
E-STOP triggers automatically when: human within 50 cm, joint torque exceeds 110%,
WMS communication lost for over 2 seconds, or battery below 8%.
Recovery: resolve trigger, clear personnel, press physical RESET button,
run abbreviated diagnostic DIAG-002 taking 45 seconds, log in WMS.
Supervisor must authorise restart after any human-proximity E-STOP.
During fire alarm: robot moves automatically to safe park position SP-01."""},

    {"id":"DOC-009","title":"Gripper Arm Specifications","category":"Equipment",
     "content":"""Gripper Arm Specifications — Model ARM-6DOF-PRO.
6 degrees of freedom: 3 positional and 3 rotational.
Maximum reach: 1200 mm. Positioning accuracy: plus or minus 0.5 mm.
Maximum end-effector speed: 2.0 m/s in unrestricted zones.
Standard gripper maximum payload: 25 kg.
Reinforced gripper maximum payload: 80 kg.
Fork attachment maximum payload: 150 kg.
Maximum grip force: 200 N standard, 500 N reinforced.
Tool change time under 30 seconds for all attachments."""},

    {"id":"DOC-013","title":"Gripper Troubleshooting","category":"Troubleshooting",
     "content":"""Gripper Failure Troubleshooting.
Gripper not closing: run CMD GRIPPER_RESET, then GRIPPER_TEST_001.
Check grip force parameter is not set to zero in WMS settings.
If test fails: replace actuator module GRP-ACT-01 (Level 2 tech required).
Dropping items: verify grip force is at least 3 times item weight in Newtons.
Recalibrate tactile sensors with CMD CAL_TACTILE (takes 2 minutes).
Check jaw wear — replace if gap deviation exceeds 1 mm.
Gripper overheating above 70 C: stop immediately, allow 20 min cool-down."""},
]

class CompactRAG:
    """
    Lightweight RAG system for integration — no chunking needed
    since documents are already concise.
    """
    def __init__(self, kb):
        print("⬇️  Loading embedding model...")
        self.sbert  = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb     = kb
        texts       = [d['content'] for d in kb]
        embs        = self.sbert.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False).astype(np.float32)
        self.index  = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        print(f"✅ RAG index built: {len(kb)} documents")

    def query(self, question, obj_class=None, top_k=2):
        """Retrieve + generate answer, optionally biased to obj_class."""
        enhanced_q = f"{obj_class} {question}" if obj_class else question
        q_emb      = self.sbert.encode([enhanced_q], normalize_embeddings=True,
                                       show_progress_bar=False).astype(np.float32)
        scores, idxs = self.index.search(q_emb, top_k)
        results      = [(self.kb[i], float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

        if not results:
            return "No relevant documentation found.", []

        answer_parts = [
            f"📋 Handling guidance for {obj_class or 'this item'}:\n"
        ]
        for doc, score in results:
            key_lines = [l.strip() for l in doc['content'].split('\n')
                         if len(l.strip()) > 30][:4]
            answer_parts.append(f"[{doc['id']}] {doc['title']}:")
            for line in key_lines:
                answer_parts.append(f"  • {line}")
            answer_parts.append("")

        citations = [f"{d['id']}: {d['title']}" for d, _ in results]
        return '\n'.join(answer_parts), citations

print("🔧 Building RAG system...")
rag = CompactRAG(KB_COMPACT)


# ─────────────────────────────────────────────────────────────
# CELL 5: Scene Generator (Integration Test Images)
# ─────────────────────────────────────────────────────────────

def make_integration_scene(scene_type='fragile'):
    """
    Generate a labelled warehouse scene for integration testing.
    scene_type: 'fragile' | 'heavy' | 'hazardous' | 'mixed'
    """
    img = np.full((480, 640, 3), (75, 72, 68), dtype=np.uint8)

    # floor shading
    for i in range(480):
        img[i] = np.clip(np.array([75,72,68]) * (0.65 + 0.35 * i/480), 0, 255).astype(np.uint8)

    if scene_type == 'fragile':
        # Cardboard box with FRAGILE sticker
        cv2.rectangle(img, (140,140),(360,380),(35,85,145),-1)
        cv2.rectangle(img, (360,118),(390,358),(22,60,110),-1)
        cv2.rectangle(img, (140,118),(390,142),(45,105,175),-1)
        cv2.rectangle(img, (140,140),(360,380),(15,48,82),2)
        cv2.line(img,(250,140),(250,380),(55,125,195),3)
        cv2.line(img,(140,260),(360,260),(55,125,195),3)
        cv2.rectangle(img,(160,210),(340,310),(40,40,195),-1)
        cv2.putText(img,"FRAGILE",(170,270),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3)
        cv2.putText(img,"HANDLE WITH CARE",(155,305),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,255),1)
        cv2.rectangle(img,(30,380),(610,415),(18,45,75),-1)

    elif scene_type == 'heavy':
        # Dark wooden crate — large
        cv2.rectangle(img,(80,100),(440,390),(90,60,30),-1)
        cv2.rectangle(img,(440,80),(475,368),(60,38,18),-1)
        cv2.rectangle(img,(80,80),(475,105),(110,78,40),-1)
        cv2.rectangle(img,(80,100),(440,390),(45,28,10),3)
        for y in range(120,390,30):
            cv2.line(img,(80,y),(440,y),(72,46,22),2)
        for corner in [(80,100),(440,100),(80,390),(440,390)]:
            cv2.rectangle(img,(corner[0]-6,corner[1]-6),(corner[0]+6,corner[1]+6),(160,160,160),-1)
        cv2.rectangle(img,(160,200),(360,290),(25,18,10),-1)
        cv2.putText(img,"HEAVY",(170,258),cv2.FONT_HERSHEY_SIMPLEX,1.0,(180,180,200),3)
        cv2.putText(img,"85 kg",(195,290),cv2.FONT_HERSHEY_SIMPLEX,0.65,(140,140,165),2)
        cv2.rectangle(img,(30,390),(610,425),(15,40,68),-1)

    elif scene_type == 'hazardous':
        # Yellow hazmat drum
        cx = 300
        cv2.ellipse(img,(cx,145),(80,28),0,0,360,(0,175,215),-1)
        cv2.rectangle(img,(cx-80,142),(cx+80,395),(0,160,200),-1)
        cv2.ellipse(img,(cx,395),(80,28),0,0,360,(0,130,168),-1)
        for y in [195,255,315,365]:
            cv2.line(img,(cx-80,y),(cx+80,y),(0,195,240),3)
        cv2.rectangle(img,(cx-55,215),(cx+55,355),(0,0,0),-1)
        cv2.putText(img,"!",(cx-18,320),cv2.FONT_HERSHEY_SIMPLEX,3.2,(0,215,255),5)
        cv2.putText(img,"HAZARDOUS",(cx-72,200),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
        cv2.putText(img,"CORROSIVE",(cx-68,370),cv2.FONT_HERSHEY_SIMPLEX,0.58,(200,240,255),1)
        cv2.rectangle(img,(30,398),(610,428),(15,40,68),-1)

    elif scene_type == 'mixed':
        # Three objects: fragile box left, heavy crate centre, hazmat drum right
        cv2.rectangle(img,(20,200),(175,385),(35,82,142),-1)
        cv2.rectangle(img,(175,185),(198,368),(22,58,108),-1)
        cv2.rectangle(img,(20,185),(198,205),(45,102,172),-1)
        cv2.rectangle(img,(38,265),(158,315),(40,40,195),-1)
        cv2.putText(img,"FRAGILE",(42,298),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2)

        cv2.rectangle(img,(210,165),(420,385),(88,58,28),-1)
        cv2.rectangle(img,(420,145),(450,365),(60,36,16),-1)
        cv2.rectangle(img,(210,145),(450,170),(108,76,38),-1)
        for y in range(185,385,28): cv2.line(img,(210,y),(420,y),(70,44,20),2)
        cv2.rectangle(img,(240,245),(390,315),(25,16,8),-1)
        cv2.putText(img,"HEAVY",(248,290),cv2.FONT_HERSHEY_SIMPLEX,0.75,(180,180,200),2)

        cx2 = 545
        cv2.ellipse(img,(cx2,195),(65,22),0,0,360,(0,172,212),-1)
        cv2.rectangle(img,(cx2-65,192),(cx2+65,385),(0,158,198),-1)
        cv2.ellipse(img,(cx2,385),(65,22),0,0,360,(0,128,165),-1)
        for y in [228,278,328,368]: cv2.line(img,(cx2-65,y),(cx2+65,y),(0,192,238),2)
        cv2.rectangle(img,(cx2-42,240),(cx2+42,355),(0,0,0),-1)
        cv2.putText(img,"!",(cx2-14,325),cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,212,252),4)

        cv2.rectangle(img,(15,388),(625,418),(16,38,65),-1)

    noise = np.random.normal(0,5,img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)

# Preview all scene types
scenes = {k: make_integration_scene(k) for k in ['fragile','heavy','hazardous','mixed']}
fig, axes = plt.subplots(1,4,figsize=(20,5))
fig.suptitle("Integration Test Scenes",fontsize=14,fontweight='bold')
for ax, (name, sc) in zip(axes, scenes.items()):
    ax.imshow(cv2.cvtColor(sc, cv2.COLOR_BGR2RGB))
    ax.set_title(name.upper(),fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.show()
print("✅ Integration test scenes generated!")


# ─────────────────────────────────────────────────────────────
# CELL 6: The Integrated Pipeline Class
# ─────────────────────────────────────────────────────────────

class WarehouseRobotSystem:
    """
    ╔══════════════════════════════════════════════════════════╗
    ║  INTEGRATED WAREHOUSE ROBOT INTELLIGENCE SYSTEM          ║
    ║  Stage 1 → OpenCV:  Detect objects + bounding boxes      ║
    ║  Stage 2 → ML:      Classify each ROI (FRAGILE/HEAVY/    ║
    ║                      HAZARDOUS)                          ║
    ║  Stage 3 → RAG:     Answer handling questions from docs  ║
    ╚══════════════════════════════════════════════════════════╝
    """

    RISK_CONFIG = {
        'FRAGILE':   {'color':(50,50,220),  'emoji':'🔴', 'risk':'MEDIUM', 'speed':'5 km/h max'},
        'HEAVY':     {'color':(200,130,0),  'emoji':'🔵', 'risk':'MEDIUM', 'speed':'3 km/h max'},
        'HAZARDOUS': {'color':(0,180,220),  'emoji':'🟡', 'risk':'HIGH',   'speed':'2 km/h max'},
    }

    DEFAULT_QUESTIONS = {
        'FRAGILE':   "How should the robot handle fragile items safely?",
        'HEAVY':     "What are the weight limits and heavy item handling rules?",
        'HAZARDOUS': "What safety checks are required before moving hazardous materials?",
    }

    def __init__(self, cv_detector, ml_clf, rag_system):
        self.cv  = cv_detector
        self.ml  = ml_clf
        self.rag = rag_system

    def run(self, image_bgr, user_question=None):
        """
        Full pipeline execution on a single image.

        Returns:
            result_dict with keys:
              - annotated_img  : BGR image with all annotations
              - detections     : list of per-object dicts
              - rag_answers    : list of (question, answer, citations)
              - pipeline_time  : total ms
        """
        t_start = time.time()

        # ══ STAGE 1: COMPUTER VISION ══════════════════════
        t1 = time.time()
        raw_detections, edges = self.cv.detect(image_bgr)
        t_cv = (time.time() - t1) * 1000

        # ══ STAGE 2: ML CLASSIFICATION ════════════════════
        t2 = time.time()
        classified = []
        for det in raw_detections:
            cls, conf, probs = self.ml.predict(det['roi'])
            classified.append({**det, 'class': cls, 'confidence': conf, 'probs': probs})
        t_ml = (time.time() - t2) * 1000

        # ══ STAGE 3: RAG RETRIEVAL ═════════════════════════
        t3 = time.time()
        rag_answers  = []
        seen_classes = set()

        for obj in classified:
            cls = obj['class']
            if cls in seen_classes:
                continue
            seen_classes.add(cls)

            q = user_question if user_question else self.DEFAULT_QUESTIONS[cls]
            answer, citations = self.rag.query(q, obj_class=cls, top_k=2)
            rag_answers.append({'class': cls, 'question': q,
                                 'answer': answer, 'citations': citations})
        t_rag = (time.time() - t3) * 1000

        total_ms = (time.time() - t_start) * 1000

        # ══ ANNOTATE IMAGE ════════════════════════════════
        annotated = self._annotate(image_bgr.copy(), classified, t_cv, t_ml, t_rag)

        return {
            'annotated_img':  annotated,
            'edges':          edges,
            'detections':     classified,
            'rag_answers':    rag_answers,
            'timing':         {'cv_ms': t_cv, 'ml_ms': t_ml,
                               'rag_ms': t_rag, 'total_ms': total_ms},
        }

    def _annotate(self, img, detections, t_cv, t_ml, t_rag):
        """Draw bounding boxes, labels, confidence bars on image."""
        h, w = img.shape[:2]

        # Top status bar
        cv2.rectangle(img,(0,0),(w,45),(20,20,20),-1)
        cv2.putText(img,"WAREHOUSE ROBOT INTELLIGENCE SYSTEM",
                    (8,18),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,220,160),1)
        cv2.putText(img,f"CV:{t_cv:.0f}ms  ML:{t_ml:.0f}ms  RAG:{t_rag:.0f}ms",
                    (8,38),cv2.FONT_HERSHEY_SIMPLEX,0.42,(160,200,255),1)

        for i, det in enumerate(detections):
            x, y, bw, bh = det['bbox']
            cx, cy       = det['center']
            cls          = det['class']
            conf         = det['confidence']
            cfg          = self.RISK_CONFIG[cls]
            color        = cfg['color']

            # Bounding box (thick)
            cv2.rectangle(img,(x,y),(x+bw,y+bh),color,2)
            # Corner accents
            L = 16
            for (px,py,dx,dy) in [(x,y,1,1),(x+bw,y,-1,1),(x,y+bh,1,-1),(x+bw,y+bh,-1,-1)]:
                cv2.line(img,(px,py),(px+dx*L,py),color,3)
                cv2.line(img,(px,py),(px,py+dy*L),color,3)

            # Center cross
            cv2.drawMarker(img,(cx,cy),(255,255,255),cv2.MARKER_CROSS,14,2)

            # Label banner
            banner = f"#{i+1} {cls}  {conf*100:.0f}%"
            (tw,th),_ = cv2.getTextSize(banner,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
            by2 = max(y-1, th+8)
            cv2.rectangle(img,(x, by2-th-6),(x+tw+8, by2+2),color,-1)
            cv2.putText(img,banner,(x+4,by2-2),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

            # Dimension text
            cv2.putText(img,f"{bw}×{bh}px | ({cx},{cy})",
                        (x, y+bh+15),cv2.FONT_HERSHEY_SIMPLEX,0.36,(200,200,200),1)

            # Mini confidence bar (right side of bbox)
            bar_x = x + bw + 5
            if bar_x + 55 < w:
                for ci, (cn, cp) in enumerate(zip(CLASSES, det['probs'])):
                    bary  = y + 8 + ci*18
                    bar_w = int(cp * 50)
                    bc    = CV_COLORS[cn]
                    cv2.rectangle(img,(bar_x, bary),(bar_x+50, bary+12),(40,40,40),-1)
                    cv2.rectangle(img,(bar_x, bary),(bar_x+bar_w, bary+12),bc,-1)
                    cv2.putText(img,f"{cn[:3]}",(bar_x+52, bary+10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.3,(200,200,200),1)

        # Risk legend
        lx, ly = 8, h-65
        cv2.rectangle(img,(lx-3,ly-15),(lx+195,h-4),(25,25,25),-1)
        cv2.putText(img,"RISK:",(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
        for j, (k, v) in enumerate(self.RISK_CONFIG.items()):
            cv2.rectangle(img,(lx+40+j*65, ly-12),(lx+55+j*65, ly-1),v['color'],-1)
            cv2.putText(img,k[:3],(lx+57+j*65,ly-2),cv2.FONT_HERSHEY_SIMPLEX,0.32,(200,200,200),1)

        return img

    def print_full_report(self, result):
        """Print a structured terminal report."""
        print("\n" + "╔" + "═"*63 + "╗")
        print("║  INTEGRATED PIPELINE REPORT" + " "*35 + "║")
        print("╚" + "═"*63 + "╝")

        t = result['timing']
        print(f"\n  ⏱  Timing:  CV={t['cv_ms']:.0f}ms  |  "
              f"ML={t['ml_ms']:.0f}ms  |  RAG={t['rag_ms']:.0f}ms  |  "
              f"Total={t['total_ms']:.0f}ms")

        print(f"\n  📦 Detections: {len(result['detections'])} objects\n")
        for i, d in enumerate(result['detections'], 1):
            x, y, bw, bh = d['bbox']
            print(f"  Object #{i}:")
            print(f"    Class      : {d['class']}  ({d['confidence']*100:.1f}% confidence)")
            print(f"    Bbox       : x={x} y={y} w={bw}px h={bh}px")
            print(f"    Center     : ({d['center'][0]}, {d['center'][1]})")
            probs_str = "  ".join(f"{c}:{p*100:.0f}%" for c,p in zip(CLASSES,d['probs']))
            print(f"    Probs      : {probs_str}")
            print()

        print("  " + "─"*61)
        print("  🔍 RAG ANSWERS\n")
        for qa in result['rag_answers']:
            print(f"  Object class : {qa['class']}")
            print(f"  Question     : {qa['question']}")
            print(f"  Answer:\n")
            for line in qa['answer'].split('\n')[:12]:
                print(f"    {line}")
            print(f"\n  Sources: {' | '.join(qa['citations'])}")
            print("  " + "─"*61)

print("✅ WarehouseRobotSystem ready!")


# ─────────────────────────────────────────────────────────────
# CELL 7: Run Integration Pipeline on All Scenes
# ─────────────────────────────────────────────────────────────

cv_detector = WarehouseObjectDetector(min_contour_area=1200)
robot_system = WarehouseRobotSystem(cv_detector, ml_classifier, rag)

results = {}
for scene_name, scene_img in scenes.items():
    print(f"\n{'='*55}")
    print(f"  🤖 Running pipeline on: {scene_name.upper()} scene")
    print(f"{'='*55}")
    result = robot_system.run(scene_img)
    results[scene_name] = result
    robot_system.print_full_report(result)


# ─────────────────────────────────────────────────────────────
# CELL 8: Main Visualisation Dashboard
# ─────────────────────────────────────────────────────────────

def plot_integration_dashboard(scene_name, result, scene_img):
    """
    Full dashboard for one scene:
    Row 1: Original | Edge Map | Annotated Output
    Row 2: ML confidence bars per object | RAG answer panel
    """
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#1a1a2e')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28)

    title = f"Integrated Pipeline — {scene_name.upper()} Scene"
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#00E5CC', y=0.97)

    # ── Row 1 ──────────────────────────────────────────────

    # Original
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB))
    ax0.set_title("① Input Image", color='white', fontweight='bold')
    ax0.axis('off')
    ax0.set_facecolor('#1a1a2e')

    # Edge map
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(result['edges'], cmap='Blues')
    ax1.set_title("② Edge Detection (OpenCV)", color='white', fontweight='bold')
    ax1.axis('off')
    ax1.set_facecolor('#1a1a2e')

    # Annotated
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(cv2.cvtColor(result['annotated_img'], cv2.COLOR_BGR2RGB))
    ax2.set_title("③ Full Pipeline Output", color='white', fontweight='bold')
    ax2.axis('off')
    ax2.set_facecolor('#1a1a2e')

    # ── Row 2 ──────────────────────────────────────────────

    # ML confidence bars
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#16213e')
    detections = result['detections']

    if detections:
        n_obj    = len(detections)
        bar_w    = 0.22
        x_pos    = np.arange(len(CLASSES))
        palette  = ['#E74C3C','#3498DB','#F39C12']

        for i, det in enumerate(detections[:3]):
            offset = (i - n_obj/2 + 0.5) * bar_w
            bars   = ax3.bar(x_pos + offset, det['probs']*100,
                              bar_w, label=f"Obj #{i+1}: {det['class']}",
                              color=palette[i % 3], alpha=0.85,
                              edgecolor='white', linewidth=0.8)
            for bar, v in zip(bars, det['probs']*100):
                if v > 5:
                    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                             f"{v:.0f}%", ha='center', fontsize=7.5, color='white')

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(CLASSES, color='white', fontsize=10)
        ax3.set_ylabel('Confidence (%)', color='white')
        ax3.set_ylim(0, 115)
        ax3.set_title("② ML Classification Confidence", color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.spines[:].set_color('#555')
        ax3.legend(loc='upper right', fontsize=8,
                   facecolor='#16213e', labelcolor='white')
        ax3.grid(axis='y', alpha=0.2, color='grey')
    else:
        ax3.text(0.5, 0.5, 'No objects detected', color='white',
                 ha='center', va='center', transform=ax3.transAxes)

    # RAG answer panel (spans 2 columns)
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_facecolor('#0f3460')
    ax4.axis('off')

    if result['rag_answers']:
        qa  = result['rag_answers'][0]
        cls = qa['class']
        q   = qa['question']
        ans = qa['answer']

        # Header
        ax4.text(0.02, 0.96, "③ RAG Documentation Retrieval",
                 transform=ax4.transAxes, color='#00E5CC',
                 fontsize=12, fontweight='bold', va='top')
        ax4.text(0.02, 0.88, f"Object Class: {cls}",
                 transform=ax4.transAxes,
                 color=CLASS_COLORS.get(cls,'white'), fontsize=10, va='top')
        ax4.text(0.02, 0.80, f"Q: {q}",
                 transform=ax4.transAxes, color='#ECF0F1',
                 fontsize=9, va='top', style='italic',
                 wrap=True)

        # Answer lines
        lines    = [l for l in ans.split('\n') if l.strip()][:10]
        y_start  = 0.70
        for line in lines:
            shortened = line if len(line) < 90 else line[:87] + '…'
            ax4.text(0.02, y_start, shortened,
                     transform=ax4.transAxes,
                     color='#BDC3C7', fontsize=8, va='top',
                     fontfamily='monospace')
            y_start -= 0.065

        # Citations
        cit_str = "  |  ".join(qa['citations'])
        ax4.text(0.02, 0.04, f"📎 {cit_str}",
                 transform=ax4.transAxes, color='#7FB3D3',
                 fontsize=7.5, va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No RAG answer generated',
                 color='white', ha='center', va='center',
                 transform=ax4.transAxes)

    # Timing footer
    t = result['timing']
    fig.text(0.5, 0.01,
             f"Pipeline Timing — CV: {t['cv_ms']:.0f}ms  |  "
             f"ML: {t['ml_ms']:.0f}ms  |  RAG: {t['rag_ms']:.0f}ms  |  "
             f"Total: {t['total_ms']:.0f}ms",
             ha='center', color='#888', fontsize=9)

    plt.show()


# Plot dashboard for all scenes
for scene_name, result in results.items():
    plot_integration_dashboard(scene_name, result, scenes[scene_name])


# ─────────────────────────────────────────────────────────────
# CELL 9: Custom Query Interface
# Ask your own question about a detected object
# ─────────────────────────────────────────────────────────────

def ask_about_scene(scene_type, user_question):
    """
    Run the full pipeline on a scene and answer a custom question.

    Args:
        scene_type:     'fragile' | 'heavy' | 'hazardous' | 'mixed'
        user_question:  Any natural language question about handling
    """
    scene  = make_integration_scene(scene_type)
    result = robot_system.run(scene, user_question=user_question)

    print(f"\n{'═'*62}")
    print(f"  Scene   : {scene_type.upper()}")
    print(f"  Question: {user_question}")
    print(f"{'═'*62}\n")

    for d in result['detections'][:3]:
        print(f"  Detected: {d['class']}  ({d['confidence']*100:.1f}%)")

    for qa in result['rag_answers']:
        print(f"\n  📋 Answer for {qa['class']}:\n")
        for line in qa['answer'].split('\n')[:10]:
            print(f"     {line}")
        print(f"\n  Sources: {' | '.join(qa['citations'])}")

    plot_integration_dashboard(scene_type, result, scene)


# ── Example custom queries ──
ask_about_scene('fragile',   "What grip force and speed settings should I use?")
ask_about_scene('hazardous', "What are the maximum travel speed limits for this item?")
ask_about_scene('mixed',     "What emergency procedures apply if E-STOP is triggered?")


# ─────────────────────────────────────────────────────────────
# CELL 10: Pipeline Architecture Diagram
# ─────────────────────────────────────────────────────────────

def draw_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_title("Integrated Warehouse Robot Intelligence — Pipeline Architecture",
                 fontsize=15, fontweight='bold', color='#00E5CC', pad=18)

    stages = [
        (1.5,  3.5, "📷 INPUT\nIMAGE",              '#1E3A5F', '#4FC3F7'),
        (5.0,  3.5, "🔍 STAGE 1\nOpenCV CV\n─────────\nEdge Detection\nContour Analysis\nBounding Boxes", '#1B4332','#52B788'),
        (9.0,  3.5, "🧠 STAGE 2\nML Classifier\n─────────\nMobileNetV2\nTransfer Learning\nFRAGILE/HEAVY/HAZ", '#4A1942','#DA77F2'),
        (13.0, 3.5, "📚 STAGE 3\nRAG System\n─────────\nEmbedding Search\nFAISS Index\nDoc Retrieval",   '#7B3F00','#FFA94D'),
        (17.5, 3.5, "✅ OUTPUT\nAnnotated\nImage +\nHandling\nGuidance",  '#1B3A4B','#00E5CC'),
    ]

    box_w, box_h = 2.8, 3.6
    for (cx, cy, label, bg, fc) in stages:
        fancy = FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h,
                               boxstyle="round,pad=0.15",
                               facecolor=bg, edgecolor=fc, linewidth=2)
        ax.add_patch(fancy)
        ax.text(cx, cy, label, ha='center', va='center',
                color=fc, fontsize=9, fontweight='bold',
                multialignment='center')

    # Arrows
    arrow_xs = [(1.5+box_w/2, 5.0-box_w/2),
                (5.0+box_w/2, 9.0-box_w/2),
                (9.0+box_w/2, 13.0-box_w/2),
                (13.0+box_w/2, 17.5-box_w/2)]
    arrow_labels = ["Raw\nPixels", "Detected\nROIs", "Class\n+ Conf", "Query\n+ Class"]
    for (x0,x1), lbl in zip(arrow_xs, arrow_labels):
        ax.annotate("", xy=(x1, 3.5), xytext=(x0, 3.5),
                    arrowprops=dict(arrowstyle="-|>", color='#555',
                                   lw=2.0, mutation_scale=20))
        ax.text((x0+x1)/2, 4.6, lbl, ha='center', va='center',
                color='#888', fontsize=8, style='italic')

    # Sub-labels
    sub = [("64×64 BGR", 1.5, 1.55),
           ("all-MiniLM\n384-dim", 9.0, 1.5),
           ("FAISS\nIndexFlatIP", 13.0, 1.5)]
    for (txt, cx, cy) in sub:
        ax.text(cx, cy, txt, ha='center', va='center',
                color='#666', fontsize=7.5)

    plt.tight_layout()
    plt.show()

draw_pipeline_diagram()
print("✅ Architecture diagram shown!")


# ─────────────────────────────────────────────────────────────
# CELL 11: Save All Integration Results
# ─────────────────────────────────────────────────────────────

os.makedirs('/content/results', exist_ok=True)

# Save annotated images
for name, result in results.items():
    cv2.imwrite(f'/content/results/integrated_{name}.png', result['annotated_img'])

# Save full integration report
report = {}
for name, result in results.items():
    report[name] = {
        'timing': result['timing'],
        'detections': [
            {'class': d['class'], 'confidence': round(d['confidence'],3),
             'bbox': list(d['bbox']), 'center': list(d['center'])}
            for d in result['detections']
        ],
        'rag_answers': [
            {'class': qa['class'], 'question': qa['question'],
             'citations': qa['citations']}
            for qa in result['rag_answers']
        ]
    }

with open('/content/results/integration_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("✅ Saved:")
for fname in sorted(os.listdir('/content/results')):
    size = os.path.getsize(f'/content/results/{fname}')
    print(f"   {fname}  ({size:,} bytes)")

from google.colab import files
import zipfile

with zipfile.ZipFile('/content/Part4_Integration_Results.zip', 'w') as zf:
    for fname in os.listdir('/content/results'):
        zf.write(f'/content/results/{fname}', fname)

files.download('/content/Part4_Integration_Results.zip')
print("\n📦 Part4_Integration_Results.zip downloaded!")

print("\n" + "╔" + "═"*55 + "╗")
print("║  ✅  ALL 4 PARTS COMPLETE!                          ║")
print("╠" + "═"*55 + "╣")
print("║  Part 1 → Computer Vision (OpenCV)                  ║")
print("║  Part 2 → ML Classifier  (MobileNetV2)              ║")
print("║  Part 3 → RAG System     (FAISS + SentenceTransf.)  ║")
print("║  Part 4 → Full Integration Pipeline                 ║")
print("╚" + "═"*55 + "╝")
