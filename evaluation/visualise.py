import numpy as np
import cv2
from PIL import Image
from typing import Optional
import matplotlib.pyplot as plt


def visualise_predictions(
    image_path: str,
    predictions: list, # List of [x1, y1, x2, y2, conf, cls]
    class_names: list,
    save_path: Optional[str] = None
) -> np.ndarray:
    """Overlays predictions on image with confidence scores"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for pred in predictions:
        x1, y1, x2, y2, conf, cls_id = pred
        label = f"{class_names[int(cls_id)]: {conf:.2f}}"

        cv2.rectangle(img, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), 2)
        cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + len(label) * 10, int(y1)), (0, 255, 0), -1)
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img


def create_comparison_grid(
    image_path: str,
    predictions_a: list,
    predictions_b: list,
    class_names: list,
    title_a: str = "Baseline",
    title_b: str = "CoLoRA"
) -> None:
    """Creates side-by-side comparison of two models on the same image"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    
    img_a = visualise_predictions(image_path, predictions_a, class_names)
    img_b = visualise_predictions(image_path, predictions_b, class_names)

    ax1.imshow(img_a)
    ax1.set_title(f"{title_a}")
