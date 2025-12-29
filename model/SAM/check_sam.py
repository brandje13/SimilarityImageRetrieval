import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor='none', lw=2))


# 1. Setup
CHECKPOINT_PATH = "./weights/sam_vit_h_4b8939.pth"
DEVICE = "cuda"
MODEL_TYPE = "vit_h"

print(f"Loading {MODEL_TYPE} model to {DEVICE}...")

# 2. Load Model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

print("Model loaded! VRAM usage check recommended (should be ~6GB).")

# 3. Create Dummy Image (or load your own)
# Simulating a 1024x1024 image
image_bgr = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
# Draw a white rectangle in the middle to simulate an object
cv2.rectangle(image_bgr, (300, 300), (700, 700), (255, 255, 255), -1)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 4. Set Image Embedding
print("Generating image embedding (this is the heavy part)...")
predictor.set_image(image_rgb)

# 5. Simulate a Query Box (e.g., from ILIAS ground truth)
# Box format: [x_min, y_min, x_max, y_max]
input_box = np.array([250, 250, 750, 750])

# 6. Predict Mask
print("Predicting mask...")
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

print(f"Success! Mask shape: {masks.shape}, Score: {scores[0]:.3f}")

# Optional: Visualize if you are in a notebook
# plt.figure(figsize=(10, 10))
# plt.imshow(image_rgb)
# show_mask(masks[0], plt.gca())
# show_box(input_box, plt.gca())
# plt.axis('off')
# plt.show()