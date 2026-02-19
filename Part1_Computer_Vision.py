# ============================================================
# PART 1: Computer Vision Module - Warehouse Object Detection
# AI Research Intern - Technical Assessment
# Compatible with: Google Colab
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Install & Import Dependencies
# ─────────────────────────────────────────────────────────────
# Run this cell first in Google Colab

!pip install opencv-python-headless numpy matplotlib Pillow -q

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import urllib.request
import os
from google.colab.patches import cv2_imshow  # Colab-specific display

print("✅ All libraries imported successfully!")
print(f"OpenCV version: {cv2.__version__}")


# ─────────────────────────────────────────────────────────────
# CELL 2: Helper - Display Function for Colab
# ─────────────────────────────────────────────────────────────

def show_image(img, title="Image", figsize=(12, 8)):
    """
    Display an OpenCV image in Colab using matplotlib.
    Handles BGR to RGB conversion automatically.
    """
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_side_by_side(img1, img2, title1="Original", title2="Processed", figsize=(16, 8)):
    """Display two images side by side for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left image
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1, 
                   cmap='gray' if len(img1.shape) == 2 else None)
    axes[0].set_title(title1, fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Right image
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2, 
                   cmap='gray' if len(img2.shape) == 2 else None)
    axes[1].set_title(title2, fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

print("✅ Display helper functions ready!")


# ─────────────────────────────────────────────────────────────
# CELL 3: Generate Synthetic Warehouse Images
# (No external dataset needed - self-contained)
# ─────────────────────────────────────────────────────────────

def create_synthetic_warehouse_scene(scene_id=1):
    """
    Creates a realistic synthetic warehouse scene with:
    - Cardboard boxes (brown shades)
    - Packages (various sizes)
    - Floor/background context
    Returns a BGR image ready for OpenCV processing.
    """
    # Canvas: 640x480 warehouse scene
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # ── Background: warehouse floor (grey concrete) ──
    img[:] = (80, 75, 70)  # Dark grey background (BGR)
    
    # ── Floor gradient ──
    for i in range(480):
        alpha = i / 480
        img[i, :] = np.clip(np.array([80, 75, 70]) * (0.7 + 0.3 * alpha), 0, 255).astype(np.uint8)

    if scene_id == 1:
        # Scene 1: Three boxes of different sizes
        
        # Box 1: Large brown box (left)
        cv2.rectangle(img, (40, 150), (200, 380), (35, 80, 140), -1)      # Main face (BGR brown)
        cv2.rectangle(img, (200, 130), (230, 360), (25, 60, 110), -1)     # Right side (darker)
        cv2.rectangle(img, (40, 130), (230, 155), (45, 100, 170), -1)     # Top face (lighter)
        cv2.rectangle(img, (40, 150), (200, 380), (20, 50, 90), 2)        # Outline
        # Box tape
        cv2.line(img, (120, 150), (120, 380), (60, 130, 200), 3)
        cv2.line(img, (40, 265), (200, 265), (60, 130, 200), 3)
        # Label
        cv2.putText(img, "BOX-A", (70, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 230, 255), 2)
        cv2.putText(img, "HEAVY", (65, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 150, 220), 1)
        
        # Box 2: Medium package (center)
        cv2.rectangle(img, (260, 220), (410, 390), (30, 100, 165), -1)    # Main face
        cv2.rectangle(img, (410, 200), (435, 370), (20, 75, 130), -1)     # Right side
        cv2.rectangle(img, (260, 200), (435, 225), (40, 120, 185), -1)    # Top
        cv2.rectangle(img, (260, 220), (410, 390), (15, 45, 80), 2)       # Outline
        # FRAGILE sticker
        cv2.rectangle(img, (280, 270), (395, 330), (40, 40, 200), -1)     # Red sticker
        cv2.putText(img, "FRAGILE", (285, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "PKG-B", (300, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1)
        
        # Box 3: Small box (right)
        cv2.rectangle(img, (470, 290), (580, 400), (25, 90, 155), -1)     # Main face
        cv2.rectangle(img, (580, 275), (600, 385), (18, 65, 120), -1)     # Right side
        cv2.rectangle(img, (470, 275), (600, 295), (35, 110, 175), -1)    # Top
        cv2.rectangle(img, (470, 290), (580, 400), (15, 45, 80), 2)       # Outline
        cv2.putText(img, "PKG-C", (488, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1)
        
        # Pallet under boxes
        cv2.rectangle(img, (30, 380), (610, 420), (20, 50, 80), -1)
        for x in range(50, 600, 40):
            cv2.line(img, (x, 380), (x, 420), (30, 65, 100), 2)
            
    elif scene_id == 2:
        # Scene 2: Stacked packages + hazardous drum
        
        # Bottom large pallet
        cv2.rectangle(img, (20, 380), (620, 430), (15, 40, 70), -1)
        for x in range(40, 620, 45):
            cv2.line(img, (x, 380), (x, 430), (25, 55, 90), 2)
        
        # Stack of boxes (left pile)
        boxes_left = [(30, 230, 200, 385), (50, 130, 180, 235), (70, 80, 160, 135)]
        shades = [(30, 90, 150), (25, 75, 130), (35, 105, 170)]
        for i, (x1, y1, x2, y2) in enumerate(boxes_left):
            cv2.rectangle(img, (x1, y1), (x2, y2), shades[i], -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (10, 30, 60), 2)
            cv2.line(img, ((x1+x2)//2, y1), ((x1+x2)//2, y2), (45, 120, 190), 2)
        
        # Hazardous drum (yellow cylinder - center)
        cv2.ellipse(img, (350, 180), (55, 20), 0, 0, 360, (0, 180, 220), -1)   # Top ellipse
        cv2.rectangle(img, (295, 178), (405, 380), (0, 160, 200), -1)           # Body
        cv2.ellipse(img, (350, 380), (55, 20), 0, 0, 360, (0, 130, 170), -1)   # Bottom ellipse
        # Hazard symbol area
        cv2.rectangle(img, (305, 230), (395, 330), (0, 0, 0), -1)
        cv2.putText(img, "!", (330, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 200, 240), 3)
        cv2.putText(img, "HAZARD", (300, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Single box (right)
        cv2.rectangle(img, (450, 250), (600, 385), (28, 85, 145), -1)
        cv2.rectangle(img, (450, 250), (600, 385), (10, 30, 60), 2)
        cv2.putText(img, "EXPRESS", (455, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1)
    
    # ── Add noise for realism ──
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

# Generate and preview scenes
scene1 = create_synthetic_warehouse_scene(1)
scene2 = create_synthetic_warehouse_scene(2)

show_side_by_side(scene1, scene2, "Scene 1: Mixed Boxes", "Scene 2: Stacked + Hazardous")
print("✅ Synthetic warehouse scenes created!")


# ─────────────────────────────────────────────────────────────
# CELL 4: Core Detection Class
# ─────────────────────────────────────────────────────────────

class WarehouseObjectDetector:
    """
    Multi-method object detector for warehouse environments.
    
    Approach:
    - Uses contour analysis (edge detection) as primary method
    - Color-based segmentation as secondary classifier
    - Computes bounding boxes, dimensions, center coordinates
    - Assigns object categories: BOX, PACKAGE, DRUM/HAZARDOUS
    """
    
    def __init__(self):
        # ── Color ranges in HSV for object type classification ──
        # Brown/Cardboard boxes
        self.color_ranges = {
            'BOX_CARDBOARD': {
                'lower': np.array([8, 40, 40]),
                'upper': np.array([25, 255, 200]),
                'display_color': (0, 165, 255),   # Orange in BGR
                'label': 'Cardboard Box'
            },
            'PACKAGE_YELLOW': {
                'lower': np.array([20, 100, 100]),
                'upper': np.array([35, 255, 255]),
                'display_color': (0, 230, 230),    # Yellow in BGR
                'label': 'Hazardous Drum'
            },
            'PACKAGE_FRAGILE': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255]),
                'display_color': (50, 50, 220),    # Red in BGR
                'label': 'Fragile Package'
            }
        }
        
        self.min_contour_area = 1500    # Filter out noise
        self.detection_results = []      # Store results for reporting
    
    def preprocess(self, img):
        """
        Preprocessing pipeline:
        1. Gaussian blur to reduce noise
        2. Convert to grayscale
        3. CLAHE for contrast enhancement
        """
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # CLAHE: Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return blurred, gray, enhanced
    
    def detect_edges_and_contours(self, img):
        """
        METHOD 1: Edge Detection + Contour Analysis
        - Canny edge detection to find object boundaries
        - Morphological operations to close gaps in edges
        - Contour extraction and filtering
        """
        _, gray, enhanced = self.preprocess(img)
        
        # Canny edge detection (auto-threshold using Otsu's method)
        otsu_thresh, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(enhanced, otsu_thresh * 0.5, otsu_thresh)
        
        # Morphological closing: fills gaps between edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        # Find contours (RETR_EXTERNAL: outer contours only)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return edges, dilated, contours
    
    def classify_by_color(self, img, roi_mask=None):
        """
        METHOD 2: Color-Based Segmentation
        - Convert to HSV colorspace for robust color detection
        - Apply color range masks for each object type
        - Returns dominant classification for the region
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        color_votes = {}
        for class_name, props in self.color_ranges.items():
            mask = cv2.inRange(hsv, props['lower'], props['upper'])
            if roi_mask is not None:
                mask = cv2.bitwise_and(mask, roi_mask)
            pixel_count = cv2.countNonZero(mask)
            color_votes[class_name] = pixel_count
        
        # Return class with most pixels (if significant)
        best_class = max(color_votes, key=color_votes.get)
        if color_votes[best_class] > 200:  # Minimum pixel threshold
            return best_class, color_votes
        return 'PACKAGE', color_votes  # Default fallback
    
    def heuristic_classify(self, w, h, aspect_ratio, area):
        """
        METHOD 3: Geometry-Based Heuristic Classification
        Uses bounding box shape properties as fallback classifier.
        """
        if aspect_ratio > 1.8:
            return 'PALLET', (100, 200, 100)    # Wide = pallet
        elif area > 18000:
            return 'BOX_LARGE', (0, 165, 255)   # Large area = big box
        elif area > 7000:
            return 'BOX_MEDIUM', (0, 120, 200)  # Medium area = package
        elif aspect_ratio < 0.6:
            return 'BOX_TALL', (200, 100, 0)    # Tall = upright box
        else:
            return 'BOX_SMALL', (150, 75, 0)    # Default = small box
    
    def detect(self, img):
        """
        Main detection pipeline.
        Returns: annotated image + list of detection dictionaries.
        """
        self.detection_results = []
        output = img.copy()
        
        # Step 1: Get edges and contours
        edges, processed, contours = self.detect_edges_and_contours(img)
        
        # Step 2: Filter and analyze valid contours
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        
        # Sort by area (largest first)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:8]
        
        for i, contour in enumerate(valid_contours):
            area = cv2.contourArea(contour)
            
            # ── Bounding Box ──
            x, y, w, h = cv2.boundingRect(contour)
            
            # ── Center Coordinates ──
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # ── Aspect Ratio ──
            aspect_ratio = w / h if h > 0 else 1.0
            
            # ── Classification ──
            # Create ROI mask for color classification
            roi_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(roi_mask, [contour], -1, 255, -1)
            
            color_class, color_votes = self.classify_by_color(img, roi_mask)
            geo_class, geo_color    = self.heuristic_classify(w, h, aspect_ratio, area)
            
            # Final label logic
            label_map = {
                'PACKAGE_YELLOW':  ('🔴 HAZARDOUS', (0, 0, 220),   'HIGH'),
                'PACKAGE_FRAGILE': ('⚠️  FRAGILE',   (50, 50, 220), 'MEDIUM'),
                'BOX_CARDBOARD':   ('📦 BOX',        (0, 165, 255), 'LOW'),
            }
            
            if color_class in label_map:
                obj_label, box_color, risk = label_map[color_class]
            else:
                obj_label = f"📦 {geo_class.replace('_', ' ')}"
                box_color = geo_color
                risk = 'LOW'
            
            # ── Store Result ──
            detection = {
                'id': i + 1,
                'label': obj_label,
                'bounding_box': (x, y, w, h),
                'center': (cx, cy),
                'area_px': int(area),
                'width_px': w,
                'height_px': h,
                'aspect_ratio': round(aspect_ratio, 2),
                'risk_level': risk,
                'color': box_color
            }
            self.detection_results.append(detection)
            
            # ── Draw Bounding Box ──
            cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
            
            # ── Draw Center Crosshair ──
            cv2.drawMarker(output, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
            
            # ── Draw Label Background ──
            label_text = f"#{i+1} {obj_label.split()[-1]}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(output, (x, y - th - 10), (x + tw + 6, y), box_color, -1)
            cv2.putText(output, label_text, (x + 3, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            # ── Draw Dimensions ──
            dim_text = f"{w}x{h}px | c:({cx},{cy})"
            cv2.putText(output, dim_text, (x, y + h + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        
        # ── Draw Info Panel ──
        panel_h = 40
        cv2.rectangle(output, (0, 0), (output.shape[1], panel_h), (30, 30, 30), -1)
        cv2.putText(output, f"Warehouse Object Detector  |  Objects Found: {len(valid_contours)}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 180), 2)
        
        return output, edges, processed
    
    def print_report(self):
        """Print a structured detection report."""
        print("\n" + "="*60)
        print("  WAREHOUSE OBJECT DETECTION REPORT")
        print("="*60)
        print(f"  Total objects detected: {len(self.detection_results)}\n")
        
        for d in self.detection_results:
            x, y, w, h = d['bounding_box']
            print(f"  Object #{d['id']}: {d['label']}")
            print(f"    Bounding Box : x={x}, y={y}, w={w}px, h={h}px")
            print(f"    Center       : ({d['center'][0]}, {d['center'][1]})")
            print(f"    Area         : {d['area_px']:,} px²")
            print(f"    Aspect Ratio : {d['aspect_ratio']}")
            print(f"    Risk Level   : {d['risk_level']}")
            print()
        print("="*60)

print("✅ WarehouseObjectDetector class defined!")


# ─────────────────────────────────────────────────────────────
# CELL 5: Run Detection on Scene 1
# ─────────────────────────────────────────────────────────────

detector = WarehouseObjectDetector()

print("🔍 Running detection on Scene 1 (Mixed Boxes)...")
result1, edges1, processed1 = detector.detect(scene1)

# Show full pipeline visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Part 1: Warehouse Object Detection Pipeline - Scene 1", 
             fontsize=16, fontweight='bold', color='navy')

axes[0, 0].imshow(cv2.cvtColor(scene1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("① Original Input Image", fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(edges1, cmap='gray')
axes[0, 1].set_title("② Canny Edge Detection", fontweight='bold')
axes[0, 1].axis('off')

axes[1, 0].imshow(processed1, cmap='gray')
axes[1, 0].set_title("③ Morphological Processing (Closed + Dilated)", fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("④ Final Detection Output", fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

detector.print_report()


# ─────────────────────────────────────────────────────────────
# CELL 6: Run Detection on Scene 2
# ─────────────────────────────────────────────────────────────

print("🔍 Running detection on Scene 2 (Stacked + Hazardous)...")
result2, edges2, processed2 = detector.detect(scene2)

show_side_by_side(scene2, result2, "Scene 2: Input", "Scene 2: Detections")
detector.print_report()


# ─────────────────────────────────────────────────────────────
# CELL 7: Color Segmentation Visualization
# ─────────────────────────────────────────────────────────────

def visualize_color_segmentation(img, title="Color Segmentation"):
    """
    Visualize the HSV color-based segmentation masks.
    Shows which pixels belong to each object category.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    masks = {}
    color_ranges = {
        'Brown/Cardboard': (np.array([8, 40, 40]),   np.array([25, 255, 200]), (0, 165, 255)),
        'Yellow/Hazardous': (np.array([20, 100, 100]), np.array([35, 255, 255]), (0, 230, 230)),
        'Red/Fragile': (np.array([0, 100, 100]),   np.array([10, 255, 255]), (50, 50, 220)),
    }
    
    fig, axes = plt.subplots(1, len(color_ranges) + 1, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontweight='bold')
    axes[0].axis('off')
    
    # Each mask
    for i, (name, (lower, upper, color)) in enumerate(color_ranges.items()):
        mask = cv2.inRange(hsv, lower, upper)
        axes[i + 1].imshow(mask, cmap='gray')
        axes[i + 1].set_title(f"{name}\n(pixels: {cv2.countNonZero(mask):,})", fontweight='bold')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_color_segmentation(scene1, "Color Segmentation Masks - Scene 1")
visualize_color_segmentation(scene2, "Color Segmentation Masks - Scene 2")
print("✅ Color segmentation visualized!")


# ─────────────────────────────────────────────────────────────
# CELL 8: BONUS - Multi-Frame Object Tracking
# ─────────────────────────────────────────────────────────────

class SimpleObjectTracker:
    """
    BONUS: Frame-to-frame object tracker using centroid matching.
    
    Algorithm:
    - Detect objects in each frame using WarehouseObjectDetector
    - Match detected centroids to existing tracked objects (nearest neighbor)
    - Assign persistent IDs and draw motion trails
    - Mark objects as 'lost' if not seen for N consecutive frames
    """
    
    def __init__(self, max_distance=80, max_lost_frames=5):
        self.detector   = WarehouseObjectDetector()
        self.tracked    = {}          # {id: {'center': (x,y), 'lost': 0, 'trail': []}}
        self.next_id    = 1
        self.max_dist   = max_distance
        self.max_lost   = max_lost_frames
        self.trail_len  = 20
        
        # Color palette for track IDs
        self.colors = [
            (0, 200, 255), (0, 255, 100), (255, 100, 0),
            (200, 0, 255), (255, 200, 0), (0, 100, 255)
        ]
    
    def update(self, frame):
        """Process one frame and update all tracks."""
        output, _, _ = self.detector.detect(frame)
        detections   = self.detector.detection_results
        
        # Current frame centroids
        current_centers = [d['center'] for d in detections]
        
        # ── Match detections to existing tracks ──
        matched_ids = set()
        for obj_id, track in self.tracked.items():
            if not current_centers:
                break
            prev_center = track['center']
            
            # Find nearest detection
            dists = [np.hypot(cx - prev_center[0], cy - prev_center[1])
                     for cx, cy in current_centers]
            min_idx = int(np.argmin(dists))
            
            if dists[min_idx] < self.max_dist:
                # Match found - update track
                new_center = current_centers.pop(min_idx)
                track['center'] = new_center
                track['lost']   = 0
                track['trail'].append(new_center)
                track['trail']  = track['trail'][-self.trail_len:]  # Keep last N
                matched_ids.add(obj_id)
            else:
                track['lost'] += 1
        
        # ── Register new tracks for unmatched detections ──
        for center in current_centers:
            self.tracked[self.next_id] = {
                'center': center,
                'lost':   0,
                'trail':  [center]
            }
            self.next_id += 1
        
        # ── Remove lost tracks ──
        self.tracked = {k: v for k, v in self.tracked.items()
                        if v['lost'] <= self.max_lost}
        
        # ── Draw tracks and trails on output ──
        for obj_id, track in self.tracked.items():
            color = self.colors[(obj_id - 1) % len(self.colors)]
            cx, cy = track['center']
            
            # Draw motion trail
            trail = track['trail']
            for j in range(1, len(trail)):
                alpha = j / len(trail)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(output, trail[j-1], trail[j], faded, 2)
            
            # Draw track ID badge
            cv2.circle(output, (cx, cy), 14, color, -1)
            cv2.putText(output, str(obj_id), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            
            status = "" if track['lost'] == 0 else f"[LOST:{track['lost']}]"
            cv2.putText(output, f"TRK-{obj_id} {status}",
                        (cx + 16, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)
        
        return output
    
    def simulate_video(self, base_scene, n_frames=8):
        """
        Simulate a video sequence by applying random small translations
        to the base scene across multiple frames.
        """
        print(f"\n🎬 Simulating {n_frames}-frame tracking sequence...")
        
        frames = []
        dx, dy = 0, 0
        
        for f in range(n_frames):
            # Simulate object motion (small random shifts)
            dx = np.clip(dx + np.random.randint(-4, 5), -30, 30)
            dy = np.clip(dy + np.random.randint(-2, 3), -15, 15)
            
            # Translate scene
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(base_scene, M, (base_scene.shape[1], base_scene.shape[0]))
            
            tracked_frame = self.update(shifted)
            
            # Add frame counter
            cv2.putText(tracked_frame, f"Frame {f+1}/{n_frames}",
                        (tracked_frame.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            frames.append(tracked_frame)
        
        return frames

# ── Run the tracker ──
tracker = SimpleObjectTracker(max_distance=90)
tracked_frames = tracker.simulate_video(scene1, n_frames=6)

# Display tracking results in a grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("BONUS: Multi-Frame Object Tracking with Motion Trails",
             fontsize=14, fontweight='bold', color='navy')

for i, frame in enumerate(tracked_frames):
    r, c = i // 3, i % 3
    axes[r, c].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[r, c].set_title(f"Frame {i+1}", fontweight='bold')
    axes[r, c].axis('off')

plt.tight_layout()
plt.show()
print("✅ Multi-frame tracking complete!")


# ─────────────────────────────────────────────────────────────
# CELL 9: Upload Your Own Image (Optional)
# ─────────────────────────────────────────────────────────────

from google.colab import files

def run_on_uploaded_image():
    """
    Upload any warehouse image and run detection on it.
    Supports: JPG, PNG, JPEG
    """
    print("📁 Please upload an image file (JPG/PNG)...")
    uploaded = files.upload()
    
    for filename, data in uploaded.items():
        # Decode uploaded bytes
        nparr = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"❌ Could not decode '{filename}'. Ensure it's a valid image.")
            continue
        
        print(f"✅ Loaded '{filename}': {img.shape[1]}x{img.shape[0]} pixels")
        
        # Resize if very large
        max_dim = 800
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            print(f"   Resized to: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Run detection
        d = WarehouseObjectDetector()
        result, edges, processed = d.detect(img)
        
        # Display
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Detection on: {filename}", fontsize=14, fontweight='bold')
        
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input")
        axes[0].axis('off')
        
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title("Edges")
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Detections")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        d.print_report()

# ── Uncomment the line below to upload your own image ──
# run_on_uploaded_image()
print("ℹ️  To test your own image: uncomment the last line above and re-run this cell.")


# ─────────────────────────────────────────────────────────────
# CELL 10: Save All Output Images
# ─────────────────────────────────────────────────────────────

import os

os.makedirs('/content/results', exist_ok=True)

cv2.imwrite('/content/results/scene1_original.png', scene1)
cv2.imwrite('/content/results/scene1_detected.png', result1)
cv2.imwrite('/content/results/scene1_edges.png', edges1)
cv2.imwrite('/content/results/scene2_original.png', scene2)
cv2.imwrite('/content/results/scene2_detected.png', result2)

for i, frame in enumerate(tracked_frames):
    cv2.imwrite(f'/content/results/tracking_frame_{i+1}.png', frame)

print("✅ All result images saved to /content/results/")
print("\n📁 Files saved:")
for f in sorted(os.listdir('/content/results')):
    size = os.path.getsize(f'/content/results/{f}')
    print(f"   {f}  ({size:,} bytes)")

# Optional: Download all results as ZIP
import zipfile

with zipfile.ZipFile('/content/Part1_CV_Results.zip', 'w') as zf:
    for f in os.listdir('/content/results'):
        zf.write(f'/content/results/{f}', f)

print("\n📦 Results packaged: /content/Part1_CV_Results.zip")
files.download('/content/Part1_CV_Results.zip')
print("✅ Download triggered!")


# ─────────────────────────────────────────────────────────────
# APPROACH EXPLANATION (200-300 words)
# ─────────────────────────────────────────────────────────────
"""
PART 1 – APPROACH EXPLANATION
═══════════════════════════════════════════════════════════════

The warehouse object detection system combines three complementary
methods to robustly identify boxes, packages, pallets, and hazardous
drums under variable lighting conditions.

PRIMARY: Edge Detection + Contour Analysis
The pipeline first preprocesses each frame with Gaussian blur (noise
reduction) and CLAHE (contrast enhancement). Canny edge detection
then extracts object boundaries, using Otsu's method to auto-select
optimal thresholds — removing the need for manual tuning. Morphological
closing and dilation operations bridge gaps in partial edges (e.g.,
where shadows fall on box surfaces), producing clean closed regions.
Contours are extracted, filtered by minimum area (eliminating spurious
detections), and sorted by size.

SECONDARY: Color-Based Segmentation
For each detected contour, an HSV color mask is applied within the
bounding region. HSV was chosen over RGB because it separates
chromaticity (Hue) from illumination (Value), making colour ranges
consistent across different lighting. Cardboard brown, hazardous-drum
yellow, and fragile-sticker red are mapped to classification labels.

TERTIARY: Geometric Heuristics
When colour is ambiguous (neutral packaging), aspect ratio and area
thresholds provide a fallback: wide shapes → pallets, tall narrow
shapes → upright boxes, large areas → heavy items.

BONUS – Tracking: Centroid-based tracking matches detections across
frames using nearest-neighbour distance, maintaining persistent IDs
and drawing motion trails to visualise object movement over time.

Limitations: The colour segmentation is sensitive to non-standard
package colours and would benefit from a learned colour model in
production. Deep learning (e.g., YOLOv8) would significantly improve
detection accuracy on real warehouse footage.
═══════════════════════════════════════════════════════════════
"""
print("\n📝 Approach explanation printed above (in source code comments).")
print("\n" + "="*55)
print("  ✅ PART 1: COMPUTER VISION MODULE COMPLETE!")
print("="*55)
print("\nNext step → Run Part 2: Machine Learning Classifier")
