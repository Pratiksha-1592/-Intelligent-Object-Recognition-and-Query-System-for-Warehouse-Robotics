# ============================================================
# PART 3: RAG System - Robotics Documentation Retrieval
# AI Research Intern - Technical Assessment
# Compatible with: Google Colab (CPU)
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Install & Import Dependencies
# ─────────────────────────────────────────────────────────────

!pip install sentence-transformers faiss-cpu numpy scikit-learn matplotlib -q

import numpy as np
import json, os, re, time, textwrap
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

print("✅ All libraries imported!")


# ─────────────────────────────────────────────────────────────
# CELL 2: Knowledge Base — 15 Warehouse Robotics Documents
# Covers: handling instructions, safety protocols,
#         equipment specs, troubleshooting guides
# ─────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [

    # ── OBJECT HANDLING INSTRUCTIONS ────────────────────────
    {
        "id": "DOC-001",
        "title": "Fragile Item Handling Protocol",
        "category": "Handling Instructions",
        "content": """
Fragile Item Handling Protocol — Standard Operating Procedure

Fragile items include glassware, electronics, ceramics, precision instruments,
and any package labelled FRAGILE, HANDLE WITH CARE, or marked with a glass symbol.

Gripper Configuration:
- Use soft-padded foam gripper attachment (Model: GRP-SOFT-200)
- Set maximum grip force to 15 N — never exceed this for fragile items
- Enable tactile feedback mode: sensor threshold 0.2 N/mm²
- Approach speed: maximum 0.05 m/s during contact phase

Lifting Procedure:
1. Scan item dimensions before approach; confirm no protruding edges
2. Lower gripper vertically — avoid lateral scooping motions
3. Grip at the widest stable point, never at corners or edges
4. Lift at a maximum rate of 0.08 m/s vertically
5. Maintain item below 1.2 m height during transport
6. Decelerate to 0.03 m/s within 20 cm of destination

Placement:
- Set down at 0.03 m/s — never drop or release abruptly
- Confirm surface stability before releasing grip
- Log fragile handling event in warehouse management system (WMS)

Prohibited Actions:
- Do NOT stack fragile items unless explicitly rated for stacking
- Do NOT exceed 5 km/h robot travel speed while carrying fragile items
- Do NOT place near heat sources or vibrating machinery
        """.strip()
    },

    {
        "id": "DOC-002",
        "title": "Heavy Item Handling and Weight Limits",
        "category": "Handling Instructions",
        "content": """
Heavy Item Handling — Weight Classification and Robot Limits

Weight Classification:
- Class A (Light):    0–10 kg    → Standard gripper, any robot unit
- Class B (Medium):  10–30 kg   → Reinforced gripper, Units R2/R3/R5 only
- Class C (Heavy):   30–80 kg   → Heavy-duty fork attachment required
- Class D (Ultra):   80–150 kg  → Dual-robot lift mode only; supervisor approval needed
- Items over 150 kg: Manual forklift only — robot must NOT attempt lift

Gripper Arm Maximum Payload:
- Standard Gripper (GRP-STD-100):    25 kg maximum
- Reinforced Gripper (GRP-REF-300):  80 kg maximum
- Fork Attachment (FRK-HVY-500):    150 kg maximum

Heavy Item Protocol:
1. Weigh item via floor sensor before approach if label absent
2. Select correct attachment based on weight class
3. Lower centre of gravity: grip at or below item midpoint
4. Use two-point contact for items wider than 60 cm
5. Travel speed limited to 3 km/h for Class C and above
6. Avoid inclines greater than 5° when carrying Class C/D items
7. Never carry heavy items above conveyor height (1.0 m)

Stacking Rules:
- Heaviest items always on bottom of stack
- Maximum stack height: 2.5 m
- Interlock required for stacks above 1.8 m
        """.strip()
    },

    {
        "id": "DOC-003",
        "title": "Hazardous Material Handling — Robot Guidelines",
        "category": "Handling Instructions",
        "content": """
Hazardous Material Handling Protocol

Hazard Categories Handled by Robot:
- Class 1: Flammable liquids (marked with flame symbol)
- Class 2: Corrosive substances (diamond warning symbol)
- Class 3: Toxic materials (skull symbol)
- Class 4: Oxidising agents (circle with flame)
- Class 5: Compressed gas cylinders

Pre-Handling Checklist (MANDATORY):
1. Confirm hazmat clearance in WMS — robot MUST receive HAZMAT_CLEAR signal
2. Verify containment integrity: no visible leaks, dents, or label damage
3. Activate chemical-resistant gripper coating (spray nozzle A3)
4. Switch ventilation in handling zone to maximum (fan speed: HIGH)
5. Alert human supervisor via LED beacon (Red flashing) before starting
6. Ensure spill containment tray deployed beneath robot arm workspace

During Transport:
- Maximum speed: 2 km/h for Class 1 and 3; 1 km/h for Class 2
- Keep item level at all times — tilt sensors must stay below 5°
- No sharp turns; minimum turn radius 1.5 m
- Dedicated hazmat lane must be clear of other robots

Post-Handling:
- Decontaminate gripper with solvent spray (nozzle B1) for 30 seconds
- Log hazmat event with timestamp, material class, start/end location
- Alert supervisor when task complete; await human confirmation
- Run self-diagnostic on tilt and grip sensors after every hazmat task
        """.strip()
    },

    {
        "id": "DOC-004",
        "title": "Package Sorting and Conveyor Belt Operations",
        "category": "Handling Instructions",
        "content": """
Package Sorting Operations — Robot-Conveyor Integration

Conveyor Belt Specifications:
- Speed range: 0.1 – 1.5 m/s (set via WMS control panel)
- Maximum package weight: 50 kg per slot
- Package size limit: 80 cm × 60 cm × 60 cm
- Barcode scanner array at zones: Z1 (intake), Z3 (sort), Z7 (dispatch)

Robot-Conveyor Handoff Protocol:
1. Robot approaches conveyor at 90° angle — never parallel approach
2. Synchronise robot arm speed with belt speed before placement
3. Place item within ±2 cm of centre line — misalignment triggers re-sort
4. Release grip only after confirming item velocity matches belt (≥0.9× belt speed)
5. Withdraw arm within 0.5 seconds of release — collision zone active

Sorting Logic:
- Zone A (Red label):  Fragile → slow belt, padded exit chute
- Zone B (Blue label): Standard → normal operations
- Zone C (Yellow):     Hazardous → sealed lane, restricted access
- Zone D (Grey):       Returns → side spur to returns processing

Error Conditions:
- ITEM_MISALIGN: Re-grip and re-place; log event
- BELT_JAM: Stop immediately; alert maintenance; do NOT force item
- BARCODE_FAIL: Route to manual inspection station Z9
        """.strip()
    },

    # ── SAFETY PROTOCOLS ────────────────────────────────────
    {
        "id": "DOC-005",
        "title": "Pre-Operation Safety Checklist",
        "category": "Safety Protocols",
        "content": """
Pre-Operation Safety Checklist — Daily Startup

This checklist must be completed before any robot begins a shift.
Failure to complete all items results in an automatic lock-out.

Mechanical Checks:
□ Inspect all joint seals — no oil leaks or cracking
□ Check gripper jaw alignment: gap deviation < 0.5 mm
□ Verify all bolts at torque spec (torque log required weekly)
□ Test emergency stop (E-STOP) button — must halt arm within 0.3 s
□ Inspect cable conduits — no chafing, cuts, or exposed wires

Sensor Checks:
□ LiDAR self-test: range accuracy ±2 cm at 5 m
□ Camera calibration: run checkerboard test, verify <1 px reprojection error
□ Tactile sensors: press test confirms ±0.05 N accuracy
□ Tilt sensor: verify zero-point calibration on flat surface
□ Proximity sensors: confirm 10 cm detection threshold

Software Checks:
□ WMS connection: ping < 50 ms
□ Confirm latest firmware installed (check update log)
□ Run diagnostic routine DIAG-001: all green indicators required
□ Verify work zone boundary map loaded and matches physical barriers

Environment Checks:
□ Floor clear of obstacles and spills in robot travel zone
□ All personnel clear of Zone 1 (robot operating area)
□ Emergency lighting operational
□ Spill kits accessible at stations S1, S3, S7

Sign-Off: Supervisor must digitally confirm checklist before robot activation.
        """.strip()
    },

    {
        "id": "DOC-006",
        "title": "Emergency Stop and Incident Response Procedures",
        "category": "Safety Protocols",
        "content": """
Emergency Stop and Incident Response

E-STOP Activation Triggers (Automatic):
- Human detected within 50 cm of robot arm (LiDAR + camera fusion)
- Joint torque exceeds 110% rated maximum for > 0.1 s
- Communication loss with WMS for > 2 s
- Battery below 8% (emergency return to dock)
- Fire/smoke detector signal received from warehouse system
- Tilt > 15° on any axis

E-STOP Recovery Procedure:
1. Identify and resolve the triggering condition
2. Ensure all personnel are clear of robot zone
3. Press physical RESET button on robot base (not WMS only)
4. Run abbreviated diagnostic DIAG-002 (takes 45 s)
5. Log incident in WMS with: time, trigger type, resolution action
6. Supervisor must authorise restart if E-STOP triggered by human proximity

Collision Incident Response:
1. Immediately activate E-STOP manually
2. Assess for injury — call first aid if required (+999 internal)
3. Do NOT move robot until incident assessed by safety officer
4. Preserve sensor logs — do not clear WMS cache
5. Complete incident report IR-001 within 2 hours
6. Robot must pass full DIAG-001 before returning to service

Fire Response:
- Robot automatically moves to safe park position (SP-01) when fire alarm triggers
- Do NOT attempt to retrieve robot during active fire
- Robot enters hibernate mode; battery enters safe-discharge at 0.1 A
        """.strip()
    },

    {
        "id": "DOC-007",
        "title": "Human-Robot Collaboration Safety Zones",
        "category": "Safety Protocols",
        "content": """
Human-Robot Collaboration (HRC) Safety Zone Definitions

Zone 1 — Exclusion Zone (Red, 0–50 cm from robot):
- No human entry during robot operation
- Automatic E-STOP if zone breached
- Physical barrier (light curtain + floor marking required)

Zone 2 — Warning Zone (Amber, 50–150 cm from robot):
- Human entry triggers speed reduction to 0.5 m/s
- Audible warning: beep every 0.5 s
- Robot avoids extension towards human

Zone 3 — Collaboration Zone (Green, 150–300 cm from robot):
- Human-robot concurrent work permitted
- Robot speed limited to 1.5 m/s
- All sharp/protruding tools must be retracted

Zone 4 — Free Zone (> 300 cm):
- Normal robot speed and operation
- Standard LiDAR monitoring only

Shared Workspace Rules:
- Robot always yields to human path — reroutes within 1 s
- Eye-contact emulation via LED panel signals intent direction
- Voice alert plays 3 s before robot enters new aisle
- Collaborative tasks (hand-off mode) require human glove sensor pairing

Safety Officer Responsibilities:
- Review zone logs daily
- Update zone map within 4 hours of any warehouse layout change
- Conduct quarterly HRC drills
        """.strip()
    },

    {
        "id": "DOC-008",
        "title": "Battery Safety and Charging Protocols",
        "category": "Safety Protocols",
        "content": """
Battery Safety and Charging Protocols

Battery Specifications (Model: BAT-LFP-48V):
- Chemistry: Lithium Iron Phosphate (LFP) — safer than Li-ion
- Capacity: 48 V, 100 Ah (4.8 kWh)
- Operating temperature: 5°C – 45°C
- Storage temperature: -10°C – 35°C
- Cycle life: 3,000 cycles at 80% DoD (depth of discharge)

Charging Rules:
- Only use approved charging stations (CHG-STD or CHG-FAST)
- Fast charge (CHG-FAST): 0→80% in 1.5 h — use only when shift-critical
- Standard charge (CHG-STD): 0→100% in 4 h — preferred for daily cycle
- Never charge below -5°C ambient temperature
- Maintain 20%–80% state of charge for longest battery life
- Full charge (100%) only before extended operation shifts

Warning Indicators:
- BATT_LOW (15%): Return to dock within 10 min
- BATT_CRITICAL (8%): Immediate E-STOP; auto-dock initiated
- BATT_HOT (> 50°C): Halt charging; alert maintenance
- BATT_SWELL: Do NOT charge; isolate battery; call maintenance

Monthly Maintenance:
- Capacity test: full charge/discharge cycle, log capacity
- Terminal inspection: clean corrosion with dry brush
- BMS firmware update check
        """.strip()
    },

    # ── EQUIPMENT SPECIFICATIONS ─────────────────────────────
    {
        "id": "DOC-009",
        "title": "Gripper Arm Technical Specifications",
        "category": "Equipment Specifications",
        "content": """
Gripper Arm Technical Specifications — Model ARM-6DOF-PRO

Mechanical Specifications:
- Degrees of Freedom: 6 (3 positional + 3 rotational)
- Reach: 1,200 mm (fully extended)
- Minimum reach: 150 mm
- Payload capacity:
    Standard gripper: 25 kg
    Reinforced gripper: 80 kg
    Fork attachment: 150 kg
- Positioning accuracy: ±0.5 mm (repeatability ±0.2 mm)
- Maximum end-effector speed: 2.0 m/s (unrestricted zone)
- Maximum joint speed: 180°/s

Force and Torque:
- Maximum grip force: 200 N (standard); 500 N (reinforced)
- Torque sensing: all 6 joints, resolution 0.1 Nm
- Collision detection threshold: configurable 5–50 N

End-Effector Attachments (tool-change time: < 30 s):
- GRP-STD-100: Standard parallel jaw gripper
- GRP-SOFT-200: Foam-padded gripper for fragile items
- GRP-REF-300: Reinforced steel gripper for heavy items
- GRP-VAC-400: Vacuum suction (flat surfaces only, max 20 kg)
- FRK-HVY-500: Fork attachment for palletised loads
- CAM-VIS-600: Vision-only attachment (no gripping)

Electrical:
- Power consumption: 350 W average, 800 W peak
- Communication: EtherCAT (1 Gbps)
- Safety category: PLd (ISO 13849)

Maintenance Schedule:
- Daily: visual inspection, joint lubrication check
- Weekly: torque calibration, cable inspection
- Monthly: full joint bearing replacement check
        """.strip()
    },

    {
        "id": "DOC-010",
        "title": "Vision System and Camera Specifications",
        "category": "Equipment Specifications",
        "content": """
Vision System Specifications — WarehouseVision v3.2

Camera Hardware:
- Primary RGB Camera: 4K (3840×2160), 60 fps, FOV 90°
- Depth Camera (ToF): range 0.3–10 m, accuracy ±1 cm at 5 m
- Wide-angle fisheye: 180° FOV (navigation camera)
- Stereo pair: baseline 120 mm, for close-range depth < 0.5 m

Lighting:
- Built-in IR illuminator: effective range 8 m
- Operating: 50–100,000 lux (auto-exposure)
- Minimum for reliable detection: 50 lux (WMS alert below 30 lux)

Object Detection Performance:
- Barcode scan: up to 3 m distance, ±30° angle, 99.7% read rate
- Package size estimation: ±1 cm accuracy at < 2 m distance
- Object classification: 200 ms per frame (onboard GPU)
- Human detection: 99.9% recall at < 5 m (ISO 15066 compliant)

Supported Detection Models:
- Onboard: YOLOv8n (real-time), EfficientDet-D0
- Cloud offload: YOLOv8x (when latency > 500 ms acceptable)

Calibration:
- Automatic recalibration: triggered daily at shift start
- Manual calibration: use checkerboard CAL-CB-001 at 1.5 m
- Calibration validity period: 7 days (WMS locks robot if expired)

Data Output:
- 3D bounding boxes in robot base frame
- Confidence scores per detection
- Class labels from onboard model
        """.strip()
    },

    {
        "id": "DOC-011",
        "title": "Mobile Base and Navigation System Specs",
        "category": "Equipment Specifications",
        "content": """
Mobile Base Specifications — Model BASE-AMR-200

Physical Dimensions:
- Footprint: 800 mm × 600 mm
- Height: 350 mm (without arm: 1,750 mm with arm at rest)
- Weight: 120 kg (without payload)
- Maximum total weight (robot + payload): 270 kg

Drive System:
- Type: 4-wheel differential drive with omnidirectional capability
- Maximum speed: 6 km/h (unrestricted); 2 km/h (loaded Class C+)
- Maximum slope: 10° (8° when carrying payload > 50 kg)
- Turning radius: 0 mm (in-place rotation)
- Wheel diameter: 200 mm, polyurethane tyres

Navigation:
- SLAM algorithm: LiDAR-inertial odometry (LIO-SAM)
- LiDAR: 360° scan, 16-beam, 10 Hz refresh, 30 m range
- Map update frequency: real-time occupancy grid at 0.05 m resolution
- Path planning: A* with dynamic obstacle avoidance (DWA)
- Localisation accuracy: ±2 cm position, ±0.5° heading

Communication:
- WiFi 6 (802.11ax): primary WMS link
- 4G LTE: backup link (auto-failover < 1 s)
- Latency to WMS: < 20 ms (WiFi), < 80 ms (4G backup)

Floor Requirements:
- Maximum floor gap: 15 mm
- Minimum aisle width: 1,200 mm (robot: 800 mm + 200 mm clearance each side)
- Floor load rating: 500 kg/m² required under robot path
        """.strip()
    },

    {
        "id": "DOC-012",
        "title": "Warehouse Management System (WMS) Integration",
        "category": "Equipment Specifications",
        "content": """
WMS Integration — Robot API Reference v2.4

Connection:
- Protocol: REST API over HTTPS + WebSocket for real-time events
- Authentication: OAuth 2.0 token (refresh every 3 600 s)
- Endpoint base: https://wms.warehouse.local/api/v2/

Key Endpoints:
GET  /robots/{id}/status        — battery, position, task, alerts
POST /robots/{id}/task          — assign new pick/place task
POST /robots/{id}/estop         — software E-STOP (physical takes priority)
GET  /robots/{id}/logs          — last 1 000 event log entries
GET  /inventory/item/{barcode}  — weight, class, handling instructions
POST /incidents                 — submit incident report

Task Payload (JSON):
{
  "task_type":  "PICK_PLACE",
  "item_id":    "WH-12345",
  "from_loc":   "A3-S2",
  "to_loc":     "C7-S1",
  "priority":   2,
  "handling":   "FRAGILE",
  "weight_kg":  4.5
}

Event Codes (WebSocket stream):
- EVT_001: Task assigned
- EVT_010: E-STOP triggered (includes trigger_type field)
- EVT_020: Battery warning
- EVT_030: Sensor fault (includes sensor_id)
- EVT_050: Hazmat clearance granted/revoked
- EVT_099: Critical system fault — requires human intervention

Rate Limits: 100 API calls/minute per robot; burst 200/minute
        """.strip()
    },

    # ── TROUBLESHOOTING GUIDES ───────────────────────────────
    {
        "id": "DOC-013",
        "title": "Gripper Failure and Malfunction Troubleshooting",
        "category": "Troubleshooting Guides",
        "content": """
Gripper Failure Troubleshooting Guide

ISSUE: Gripper will not close / partial closure
Possible Causes:
1. Obstruction in jaw mechanism — inspect visually; clear debris
2. Actuator fault: run self-test CMD: GRIPPER_TEST_001
3. Grip force set to 0 — check WMS force parameter
4. Low battery affecting actuator torque — check battery level

Steps:
a) Issue CMD: GRIPPER_RESET; wait 10 s
b) If no response: power cycle gripper via breaker GB-4
c) Run GRIPPER_TEST_001 — reports jaw force and position
d) If test fails: replace actuator module GRP-ACT-01 (Level 2 tech)

ISSUE: Dropping items mid-transport
Possible Causes:
1. Grip force too low for item weight — recalibrate
2. Tactile sensor drift: recalibrate via CMD: CAL_TACTILE
3. Gripper jaw worn — measure jaw gap (should be < 0.5 mm deviation)
4. Vibration from floor surface exceeding grip margin

Steps:
a) Weigh item; verify grip force is ≥ 3× item weight in Newtons
b) Recalibrate tactile sensors: CMD: CAL_TACTILE (takes 2 min)
c) Check jaw wear — replace if gap deviation > 1 mm
d) Reduce travel speed by 30% and retest

ISSUE: False contact detection (gripper stops before touching item)
Cause: Tactile sensor oversensitivity — likely contamination
Fix: Clean sensors with IPA wipe; recalibrate CMD: CAL_TACTILE

ISSUE: Gripper overheating (temp > 70°C)
Action: Immediate stop; allow 20 min cool-down; inspect motor brushes
        """.strip()
    },

    {
        "id": "DOC-014",
        "title": "Navigation and Localisation Error Troubleshooting",
        "category": "Troubleshooting Guides",
        "content": """
Navigation and Localisation Troubleshooting

ISSUE: Robot loses position (ERR_LOCALISE_001)
Symptoms: Robot stops, requests re-localisation; WMS shows unknown position
Causes:
1. LiDAR blocked or dirty
2. Reflective surfaces causing LiDAR multipath errors
3. Map outdated — new obstacles not yet in map

Steps:
a) Check LiDAR lens — wipe with microfibre cloth if dusty
b) CMD: RELOCALISE — robot will spin 360° and rebuild position estimate
c) If persistent: CMD: MAP_UPDATE to refresh occupancy grid
d) Place robot at known fiducial marker (QR tile FM-01 or FM-02)
   and CMD: SET_POSE_FROM_FIDUCIAL
e) If still failing: return robot to dock; run DIAG-001 full

ISSUE: Robot taking inefficient/long paths
Cause: Stale obstacle map; ghost obstacles from previous runs
Fix: CMD: CLEAR_GHOST_OBSTACLES; then CMD: REROUTE

ISSUE: Robot colliding with dynamic obstacles
Check: Dynamic obstacle avoidance (DOA) module status
a) Verify camera feed is active: CMD: CAM_STATUS
b) Increase DOA lookahead: set param DOA_LOOKAHEAD_M = 2.5
c) Reduce max speed by 20%

ISSUE: WiFi disconnection causing robot to stop
a) Robot auto-reconnects within 10 s — wait before intervening
b) If not recovered: CMD: NETWORK_RESET (physical button on base)
c) 4G backup activates automatically if WiFi unavailable > 5 s
        """.strip()
    },

    {
        "id": "DOC-015",
        "title": "Sensor Fault Codes and Diagnostics",
        "category": "Troubleshooting Guides",
        "content": """
Sensor Fault Codes Reference — Diagnostic Guide

FAULT CODE: SEN-001 (LiDAR Scan Failure)
Severity: CRITICAL — robot halts
Cause: LiDAR hardware fault or spin motor failure
Action:
1. Check LiDAR LED: solid red = hardware fault; blinking = spin fault
2. CMD: LIDAR_RESTART (soft restart, 30 s)
3. If unresolved: replace LiDAR unit LD-VLP-16 (Level 3 tech only)
4. Do NOT operate robot without functioning LiDAR

FAULT CODE: SEN-002 (Camera Calibration Invalid)
Severity: WARNING — robot speed reduced to 1 m/s
Cause: Calibration expired or camera physically shifted
Action: CMD: CAL_CAMERA — takes 5 min; robot must be stationary at dock

FAULT CODE: SEN-003 (Tilt Sensor Fault)
Severity: HIGH — hazmat transport locked
Cause: Accelerometer drift or hardware fault
Action:
1. Place robot on confirmed flat surface
2. CMD: CAL_TILT — zeroes tilt reference
3. If fault persists after calibration: replace IMU module IMU-6DOF

FAULT CODE: SEN-004 (Tactile Sensor Array Fault)
Severity: MEDIUM — fragile handling locked; heavy/standard allowed
Cause: Contamination or sensor wire damage
Action:
1. Visual inspection of sensor array
2. Clean with IPA wipe; CMD: CAL_TACTILE
3. If > 3 sensors unresponsive: replace sensor mat GRP-TAC-001

FAULT CODE: SEN-010 (Battery Management System Fault)
Severity: CRITICAL — robot returns to dock immediately
Action: Do NOT override; allow full BMS diagnostic; call battery technician

Diagnostic Commands Summary:
- DIAG-001: Full system diagnostic (~5 min)
- DIAG-002: Abbreviated post-E-STOP diagnostic (~45 s)
- DIAG-003: Sensor-only diagnostic (~2 min)
        """.strip()
    }
]

print(f"✅ Knowledge base loaded: {len(KNOWLEDGE_BASE)} documents")
print(f"\nDocument Index:")
for doc in KNOWLEDGE_BASE:
    print(f"   {doc['id']}  [{doc['category']:28s}]  {doc['title']}")


# ─────────────────────────────────────────────────────────────
# CELL 3: Document Chunker
# Splits long docs into overlapping chunks for finer retrieval
# ─────────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Splits documents into overlapping text chunks.

    Strategy:
    - Split on paragraph/section boundaries (double newline)
    - Merge short paragraphs until chunk reaches target size
    - Overlap: carry last `overlap_sentences` into next chunk
    - Preserve document metadata in every chunk
    """

    def __init__(self, chunk_size=300, overlap_words=40, min_chunk_words=30):
        self.chunk_size    = chunk_size     # target words per chunk
        self.overlap_words = overlap_words  # words to carry between chunks
        self.min_chunk     = min_chunk_words

    def chunk_document(self, doc):
        """Chunk a single document dict → list of chunk dicts."""
        content    = doc['content']
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        chunks       = []
        current_buf  = []
        current_wds  = 0
        chunk_idx    = 0

        for para in paragraphs:
            words = para.split()
            current_buf.append(para)
            current_wds += len(words)

            if current_wds >= self.chunk_size:
                chunk_text = '\n\n'.join(current_buf)
                chunks.append(self._make_chunk(doc, chunk_text, chunk_idx))
                chunk_idx += 1

                # Overlap: keep last `overlap_words` worth
                overlap_text = chunk_text.split()[-self.overlap_words:]
                current_buf  = [' '.join(overlap_text)]
                current_wds  = len(overlap_text)

        # Flush remaining
        if current_wds >= self.min_chunk:
            chunk_text = '\n\n'.join(current_buf)
            chunks.append(self._make_chunk(doc, chunk_text, chunk_idx))

        return chunks

    def _make_chunk(self, doc, text, idx):
        return {
            'chunk_id':  f"{doc['id']}-C{idx:02d}",
            'doc_id':    doc['id'],
            'title':     doc['title'],
            'category':  doc['category'],
            'text':      text,
            'word_count': len(text.split()),
        }

    def chunk_all(self, documents):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


chunker    = DocumentChunker(chunk_size=250, overlap_words=40)
all_chunks = chunker.chunk_all(KNOWLEDGE_BASE)

print(f"✅ Chunking complete!")
print(f"   Documents : {len(KNOWLEDGE_BASE)}")
print(f"   Chunks    : {len(all_chunks)}")
print(f"   Avg words : {np.mean([c['word_count'] for c in all_chunks]):.0f}")

# Preview
print(f"\nFirst 5 chunks:")
for c in all_chunks[:5]:
    print(f"   {c['chunk_id']:15s}  words={c['word_count']:3d}  {c['title'][:45]}")


# ─────────────────────────────────────────────────────────────
# CELL 4: Embedding Engine + FAISS Vector Index
# ─────────────────────────────────────────────────────────────

import faiss

class EmbeddingEngine:
    """
    Embeds text chunks using sentence-transformers (all-MiniLM-L6-v2).
    Builds a FAISS flat L2 index for fast similarity search.

    Why all-MiniLM-L6-v2?
    - 384-dim embeddings, very fast on CPU
    - Strong semantic similarity for short-to-medium passages
    - 6× smaller than large models — suitable for Colab CPU
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"⬇️  Loading embedding model: {model_name}...")
        self.model   = SentenceTransformer(model_name)
        self.dim     = self.model.get_sentence_embedding_dimension()
        self.index   = None
        self.chunks  = []
        print(f"✅ Model loaded | Embedding dim: {self.dim}")

    def embed(self, texts, batch_size=64, show_progress=True):
        """Embed a list of strings → numpy array [N, dim]."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2 normalise → cosine via dot product
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def build_index(self, chunks):
        """Build FAISS index from chunk list."""
        self.chunks = chunks
        texts       = [c['text'] for c in chunks]

        print(f"📐 Embedding {len(texts)} chunks...")
        embeddings  = self.embed(texts)

        # FAISS IndexFlatIP = inner product (cosine when normalised)
        self.index  = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        print(f"✅ FAISS index built:")
        print(f"   Vectors indexed: {self.index.ntotal}")
        print(f"   Embedding dim  : {self.dim}")
        return embeddings

    def search(self, query, top_k=5):
        """
        Search the index for chunks most similar to query.
        Returns list of (chunk_dict, score) tuples.
        """
        q_emb    = self.embed([query], show_progress=False)   # [1, dim]
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results


engine     = EmbeddingEngine()
embeddings = engine.build_index(all_chunks)


# ─────────────────────────────────────────────────────────────
# CELL 5: Response Generator
# Combines retrieved chunks into a coherent answer
# ─────────────────────────────────────────────────────────────

class ResponseGenerator:
    """
    Generates answers by:
    1. Retrieving relevant chunks via semantic search
    2. Re-ranking by keyword overlap (hybrid scoring)
    3. Deduplicating chunks from same document
    4. Formatting a structured answer with citations
    """

    def __init__(self, engine, top_k=4, min_score=0.25):
        self.engine    = engine
        self.top_k     = top_k
        self.min_score = min_score

    def _keyword_boost(self, query, text):
        """Simple keyword overlap boost for hybrid retrieval."""
        q_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        t_words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        overlap = q_words & t_words
        return len(overlap) / max(len(q_words), 1) * 0.15  # max 0.15 boost

    def _deduplicate(self, results):
        """Keep at most 2 chunks per source document."""
        seen_docs = defaultdict(int)
        filtered  = []
        for chunk, score in results:
            doc_id = chunk['doc_id']
            if seen_docs[doc_id] < 2:
                filtered.append((chunk, score))
                seen_docs[doc_id] += 1
        return filtered

    def generate(self, query, verbose=True):
        """Full RAG pipeline: retrieve → rerank → format answer."""
        t0 = time.time()

        # ── Step 1: Semantic Retrieval ──
        raw_results = self.engine.search(query, top_k=self.top_k * 2)

        # ── Step 2: Hybrid Reranking (semantic + keyword) ──
        reranked = []
        for chunk, sem_score in raw_results:
            boost       = self._keyword_boost(query, chunk['text'])
            final_score = sem_score + boost
            reranked.append((chunk, final_score, sem_score))

        reranked.sort(key=lambda x: x[1], reverse=True)

        # ── Step 3: Filter by min score ──
        filtered = [(c, fs, ss) for c, fs, ss in reranked if fs >= self.min_score]

        # ── Step 4: Deduplicate ──
        deduped  = self._deduplicate([(c, fs) for c, fs, ss in filtered])

        # ── Step 5: Take top_k ──
        top_results = deduped[:self.top_k]

        elapsed = time.time() - t0

        if not top_results:
            return self._no_result_response(query)

        return self._format_response(query, top_results, elapsed, reranked[:self.top_k])

    def _format_response(self, query, results, elapsed, reranked):
        """Format retrieved context into a readable answer."""
        lines = []
        lines.append("=" * 65)
        lines.append(f"  QUERY: {query}")
        lines.append("=" * 65)

        lines.append(f"\n📚 RETRIEVED CONTEXT  ({len(results)} sources, {elapsed*1000:.0f} ms)\n")

        for rank, (chunk, score) in enumerate(results, 1):
            lines.append(f"  {'─'*60}")
            lines.append(f"  [{rank}] {chunk['title']}")
            lines.append(f"       Source: {chunk['doc_id']} | "
                         f"Category: {chunk['category']} | "
                         f"Score: {score:.3f}")
            lines.append(f"  {'─'*60}")
            # Wrap text nicely
            for para in chunk['text'].split('\n\n'):
                wrapped = textwrap.fill(para.strip(), width=60,
                                        initial_indent='  ',
                                        subsequent_indent='  ')
                lines.append(wrapped)
            lines.append("")

        # ── Synthesised Answer ──
        lines.append(f"{'─'*65}")
        lines.append(f"  💡 SYNTHESISED ANSWER")
        lines.append(f"{'─'*65}")

        answer = self._synthesise(query, results)
        for line in answer.split('\n'):
            lines.append(f"  {line}")

        lines.append(f"\n  📎 CITATIONS:")
        seen = set()
        for rank, (chunk, score) in enumerate(results, 1):
            if chunk['doc_id'] not in seen:
                lines.append(f"     [{rank}] {chunk['doc_id']}: {chunk['title']}")
                seen.add(chunk['doc_id'])

        lines.append("=" * 65)
        return '\n'.join(lines)

    def _synthesise(self, query, results):
        """
        Rule-based synthesiser: extracts key bullet points from
        retrieved chunks relevant to the query.
        (In production this would call an LLM like GPT or Claude.)
        """
        q_lower = query.lower()

        # Combine all retrieved text
        all_text = '\n'.join(chunk['text'] for chunk, _ in results)

        # Extract relevant lines (contain query keywords or action verbs)
        q_keywords = set(re.findall(r'\b\w{4,}\b', q_lower))
        action_words = {'set', 'use', 'check', 'ensure', 'verify', 'run',
                        'limit', 'maximum', 'minimum', 'activate', 'required',
                        'never', 'always', 'must', 'avoid', 'confirm'}

        relevant_lines = []
        for line in all_text.split('\n'):
            line = line.strip()
            if not line or len(line) < 20:
                continue
            l_lower = line.lower()
            l_words = set(re.findall(r'\b\w{4,}\b', l_lower))

            kw_match     = len(q_keywords & l_words) >= 1
            action_match = len(action_words & l_words) >= 1

            if kw_match and (action_match or len(line) > 60):
                relevant_lines.append(line)

        if not relevant_lines:
            relevant_lines = all_text.split('\n')[:8]

        # Deduplicate and take top lines
        seen_lines = set()
        unique     = []
        for line in relevant_lines:
            key = line[:60]
            if key not in seen_lines:
                seen_lines.add(key)
                unique.append(line)

        top_lines = unique[:8]

        answer_lines = [
            f"Based on the warehouse robotics documentation:\n"
        ]
        for line in top_lines:
            # Format as bullet if not already
            line = line.lstrip('•-□').strip()
            if line:
                answer_lines.append(f"• {line}")

        src_titles = list(dict.fromkeys(c['title'] for c, _ in results))
        answer_lines.append(f"\n↳ Source documents: {'; '.join(src_titles)}")

        return '\n'.join(answer_lines)

    def _no_result_response(self, query):
        return (f"No relevant documentation found for query:\n'{query}'\n"
                f"Please check the knowledge base or rephrase your query.")


generator_rag = ResponseGenerator(engine, top_k=3, min_score=0.20)
print("✅ RAG pipeline ready!")


# ─────────────────────────────────────────────────────────────
# CELL 6: Demo — 5 Query-Response Examples
# ─────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "How should the robot handle fragile items?",
    "What is the maximum weight capacity for the gripper arm?",
    "What safety checks are needed before moving hazardous materials?",
    "How do I fix a gripper that keeps dropping items during transport?",
    "What should the robot do when the battery is critically low?",
]

print("🤖 WAREHOUSE ROBOTICS RAG SYSTEM — LIVE DEMO")
print("=" * 65)

responses = []
for i, query in enumerate(DEMO_QUERIES, 1):
    print(f"\n{'🔍' * 3}  QUERY {i} of {len(DEMO_QUERIES)}  {'🔍' * 3}\n")
    response = generator_rag.generate(query)
    responses.append(response)
    print(response)
    print()


# ─────────────────────────────────────────────────────────────
# CELL 7: Interactive Query Interface
# ─────────────────────────────────────────────────────────────

def interactive_rag():
    """
    Simple interactive loop for querying the RAG system.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "="*65)
    print("  🤖 INTERACTIVE WAREHOUSE ROBOTICS RAG ASSISTANT")
    print("  Type your question, or 'quit' to exit")
    print("="*65 + "\n")

    while True:
        try:
            query = input("❓ Your question: ").strip()
        except EOFError:
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("👋 Exiting RAG assistant.")
            break

        response = generator_rag.generate(query)
        print(response)
        print()

# ── Uncomment to run interactively ──
# interactive_rag()
print("ℹ️  Uncomment 'interactive_rag()' above to ask your own questions!")


# ─────────────────────────────────────────────────────────────
# CELL 8: Retrieval Visualisation — Similarity Heatmap
# ─────────────────────────────────────────────────────────────

def plot_retrieval_analysis(queries, engine, all_chunks, top_k=5):
    """
    For each query, show a bar chart of top retrieved chunk scores.
    Also shows an overall query-document category heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Part 3: RAG Retrieval Analysis",
                 fontsize=14, fontweight='bold', color='navy')

    # ── Left: Score bars for each query ──
    ax = axes[0]
    colors_q = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6']
    y_offset  = 0
    ytick_pos = []
    ytick_lbl = []

    for qi, query in enumerate(queries[:5]):
        results = engine.search(query, top_k=top_k)
        scores  = [s for _, s in results]
        labels  = [f"{c['doc_id']}" for c, _ in results]

        positions = [y_offset + j * 1.1 for j in range(len(scores))]
        bars = ax.barh(positions, scores,
                       color=colors_q[qi], alpha=0.75,
                       edgecolor='white', height=0.9)

        for pos, bar, lbl in zip(positions, bars, labels):
            ax.text(bar.get_width() + 0.005, pos,
                    lbl, va='center', fontsize=7.5, color='#2C3E50')

        mid = np.mean(positions)
        ytick_pos.append(mid)
        ytick_lbl.append(f"Q{qi+1}: {query[:28]}…" if len(query) > 28 else f"Q{qi+1}: {query}")
        y_offset += top_k * 1.1 + 1.5

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lbl, fontsize=8)
    ax.set_xlabel('Cosine Similarity Score', fontweight='bold')
    ax.set_title('Top Retrieved Chunks per Query', fontweight='bold')
    ax.axvline(0.25, color='grey', linestyle='--', linewidth=1, label='Min score (0.25)')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # ── Right: Category hit heatmap ──
    ax2 = axes[1]
    categories  = list(dict.fromkeys(c['category'] for c in all_chunks))
    hit_matrix  = np.zeros((len(queries), len(categories)))

    for qi, query in enumerate(queries):
        results = engine.search(query, top_k=5)
        for chunk, score in results:
            cat_idx = categories.index(chunk['category'])
            hit_matrix[qi, cat_idx] = max(hit_matrix[qi, cat_idx], score)

    im = ax2.imshow(hit_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.8)
    plt.colorbar(im, ax=ax2, label='Max Score')

    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels([c.replace(' ', '\n') for c in categories],
                         fontsize=9, ha='center')
    ax2.set_yticks(range(len(queries)))
    ax2.set_yticklabels([f"Q{i+1}" for i in range(len(queries))], fontsize=9)
    ax2.set_title('Query → Document Category Heatmap\n(max retrieval score)', fontweight='bold')

    for qi in range(len(queries)):
        for ci in range(len(categories)):
            if hit_matrix[qi, ci] > 0.05:
                ax2.text(ci, qi, f"{hit_matrix[qi,ci]:.2f}",
                         ha='center', va='center', fontsize=7.5,
                         color='white' if hit_matrix[qi,ci] > 0.5 else 'black')

    plt.tight_layout()
    plt.show()


plot_retrieval_analysis(DEMO_QUERIES, engine, all_chunks, top_k=4)
print("✅ Retrieval analysis plotted!")


# ─────────────────────────────────────────────────────────────
# CELL 9: Knowledge Base Statistics Dashboard
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Part 3: Knowledge Base Statistics",
             fontsize=14, fontweight='bold', color='navy')

# ── Doc count per category ──
cat_counts = defaultdict(int)
for doc in KNOWLEDGE_BASE:
    cat_counts[doc['category']] += 1

axes[0].bar(range(len(cat_counts)),
            list(cat_counts.values()),
            color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60'],
            edgecolor='white', linewidth=1.2)
axes[0].set_xticks(range(len(cat_counts)))
axes[0].set_xticklabels([k.replace(' ', '\n') for k in cat_counts.keys()], fontsize=9)
axes[0].set_title('Documents per Category', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(cat_counts.values()):
    axes[0].text(i, v + 0.05, str(v), ha='center', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# ── Chunk word count distribution ──
word_counts = [c['word_count'] for c in all_chunks]
axes[1].hist(word_counts, bins=20, color='steelblue', edgecolor='white')
axes[1].axvline(np.mean(word_counts), color='red', linestyle='--',
                label=f'Mean: {np.mean(word_counts):.0f}')
axes[1].set_title('Chunk Word Count Distribution', fontweight='bold')
axes[1].set_xlabel('Words per Chunk')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(alpha=0.3)

# ── Embedding similarity matrix (subset) ──
subset_embs = embeddings[:20]
sim_matrix  = cosine_similarity(subset_embs)
im = axes[2].imshow(sim_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(im, ax=axes[2])
axes[2].set_title('Chunk Embedding Similarity\n(first 20 chunks)', fontweight='bold')
axes[2].set_xlabel('Chunk Index')
axes[2].set_ylabel('Chunk Index')

plt.tight_layout()
plt.show()
print("✅ Knowledge base statistics dashboard shown!")


# ─────────────────────────────────────────────────────────────
# CELL 10: Save Outputs
# ─────────────────────────────────────────────────────────────

os.makedirs('/content/results', exist_ok=True)

# Save knowledge base
with open('/content/results/knowledge_base.json', 'w') as f:
    json.dump(KNOWLEDGE_BASE, f, indent=2)

# Save chunks
with open('/content/results/chunks.json', 'w') as f:
    json.dump(all_chunks, f, indent=2)

# Save demo Q&A
qa_log = []
for query, response in zip(DEMO_QUERIES, responses):
    qa_log.append({'query': query, 'response': response})

with open('/content/results/demo_qa.json', 'w') as f:
    json.dump(qa_log, f, indent=2)

# Save embeddings
np.save('/content/results/embeddings.npy', embeddings)

# Save FAISS index
faiss.write_index(engine.index, '/content/results/faiss.index')

print("✅ Saved:")
for fname in sorted(os.listdir('/content/results')):
    size = os.path.getsize(f'/content/results/{fname}')
    print(f"   {fname}  ({size:,} bytes)")

# Download
from google.colab import files
import zipfile

with zipfile.ZipFile('/content/Part3_RAG_Results.zip', 'w') as zf:
    for fname in os.listdir('/content/results'):
        zf.write(f'/content/results/{fname}', fname)

files.download('/content/Part3_RAG_Results.zip')
print("\n📦 Part3_RAG_Results.zip downloaded!")


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE WRITE-UP (200-300 words)
# ─────────────────────────────────────────────────────────────
"""
PART 3 – RAG ARCHITECTURE CHOICES (200-300 words)
═══════════════════════════════════════════════════════════════

CHUNKING STRATEGY
Documents are split using a paragraph-aware chunker targeting
~250 words per chunk with a 40-word overlap between consecutive
chunks. Paragraph boundaries are respected to avoid splitting
mid-sentence. Overlap ensures that information spanning two
paragraphs is retrievable from both surrounding chunks, reducing
the risk of missing context at chunk edges.

EMBEDDING MODEL
all-MiniLM-L6-v2 (SentenceTransformers) was chosen for its
excellent trade-off between semantic quality and speed on CPU.
It produces 384-dimensional L2-normalised embeddings, enabling
cosine similarity via simple dot product. For production, a
larger model such as all-mpnet-base-v2 or a domain-adapted
technical embedding model would improve precision on specialised
robotics terminology.

VECTOR INDEX
FAISS IndexFlatIP provides exact inner-product search over
normalised vectors (equivalent to cosine similarity). Flat
index was chosen over IVF/HNSW because the corpus is small
(<100 chunks), where exact search is faster than approximate.
For corpora above 10,000 chunks, FAISS HNSW would be preferred.

HYBRID RETRIEVAL
A lightweight keyword-overlap boost (+0–0.15) is added to the
semantic score to handle cases where exact technical terms
(e.g., "GRP-SOFT-200", "DIAG-001") may not be captured by
embedding similarity alone. This hybrid approach improves recall
for specific part numbers and command codes.

RESPONSE SYNTHESIS
The current synthesiser extracts relevant bullet points
rule-based. In production, retrieved context would be passed
to an LLM (e.g., Claude or GPT-4) as a system prompt, enabling
coherent, conversational, and contextually-accurate responses.
═══════════════════════════════════════════════════════════════
"""

print("\n" + "="*55)
print("  ✅ PART 3: RAG SYSTEM COMPLETE!")
print("="*55)
print("\nNext step → Run Part 4: Integration (all 3 parts together!)")
