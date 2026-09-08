"""The 49 colour, texture and edge measurements from the original notebook."""

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage


def extract_simple_features(image_path):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((192, 192), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    red_excess = red - 0.5 * (green + blue)

    grad_x = ndimage.sobel(gray, axis=1, mode="reflect") / 8.0
    grad_y = ndimage.sobel(gray, axis=0, mode="reflect") / 8.0
    edge = np.hypot(grad_x, grad_y)
    local_mean = ndimage.uniform_filter(gray, size=9, mode="reflect")
    local_difference = gray - local_mean
    dark_line = np.maximum(local_mean - gray, 0.0)
    laplacian = ndimage.laplace(gray, mode="reflect")

    values = {}
    for name, channel in [("red", red), ("green", green), ("blue", blue)]:
        values[f"{name}_mean"] = channel.mean()
        values[f"{name}_std"] = channel.std()
        values[f"{name}_q10"] = np.quantile(channel, 0.10)
        values[f"{name}_q90"] = np.quantile(channel, 0.90)

    values.update({
        "gray_mean": gray.mean(),
        "gray_std": gray.std(),
        "gray_q10": np.quantile(gray, 0.10),
        "gray_q90": np.quantile(gray, 0.90),
        "dark_pixel_fraction": (gray < 0.25).mean(),
        "saturation_mean": saturation.mean(),
        "saturation_std": saturation.std(),
        "saturation_q90": np.quantile(saturation, 0.90),
        "high_saturation_fraction": (saturation > 0.35).mean(),
        "red_excess_mean": red_excess.mean(),
        "red_excess_std": red_excess.std(),
        "red_excess_q90": np.quantile(red_excess, 0.90),
        "rust_colour_fraction": ((red_excess > 0.08) & (red > 0.25)).mean(),
        "edge_mean": edge.mean(),
        "edge_std": edge.std(),
        "edge_q90": np.quantile(edge, 0.90),
        "edge_fraction": (edge > 0.12).mean(),
        "horizontal_gradient_mean": np.abs(grad_y).mean(),
        "vertical_gradient_mean": np.abs(grad_x).mean(),
        "gradient_axis_ratio": (np.abs(grad_x).mean() + 1e-6) / (np.abs(grad_y).mean() + 1e-6),
        "dark_line_mean": dark_line.mean(),
        "dark_line_q90": np.quantile(dark_line, 0.90),
        "dark_line_fraction": (dark_line > 0.08).mean(),
        "local_contrast_std": local_difference.std(),
        "laplacian_variance": laplacian.var(),
    })

    height, width = gray.shape
    row_edges = [0, max(1, height // 2), height]
    col_edges = [0, max(1, width // 2), width]
    for row in range(2):
        for col in range(2):
            region = np.s_[min(row_edges[row], height - 1):row_edges[row + 1],
                           min(col_edges[col], width - 1):col_edges[col + 1]]
            suffix = f"r{row + 1}c{col + 1}"
            values[f"red_excess_{suffix}"] = red_excess[region].mean()
            values[f"saturation_{suffix}"] = saturation[region].mean()
            values[f"edge_{suffix}"] = edge[region].mean()
    return values


def source_class_weights(table):
    """Give each source equal initial mass, then balance the two classes."""
    labels = table["label"].to_numpy()
    if set(labels) != {0, 1}:
        raise ValueError("Training data must contain both classes.")
    weights = 1.0 / table.groupby("source_id")["source_id"].transform("size").to_numpy()
    for label in (0, 1):
        selected = labels == label
        weights[selected] /= weights[selected].sum()
    return weights / weights.mean()


def predict_lgbm(model, image_path):
    """Predict one raw image using a saved LightGBM Booster."""
    features = pd.DataFrame([extract_simple_features(image_path)])
    probability = float(model.predict(features[model.feature_name()])[0])
    label = int(probability >= 0.5)
    return {"p_crack": probability, "label": label,
            "class_name": "crack" if label else "corrosion", "threshold": 0.5}



def fit_feature_threshold(values, labels):
    """Fit the slide's one-feature rule using training values and labels only."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(values)
    thresholds = np.unique(np.round(np.r_[unique[0] - 0.0001,
                                          (unique[:-1] + unique[1:]) / 2,
                                          unique[-1] + 0.0001], 4))
    means = [values[labels == label].mean() for label in (0, 1)]
    direction = '<=' if means[1] < means[0] else '>='
    predictions = (values[None, :] <= thresholds[:, None] if direction == '<='
                   else values[None, :] >= thresholds[:, None])
    tp = predictions[:, labels == 1].sum(axis=1)
    fp = predictions[:, labels == 0].sum(axis=1)
    fn = (labels == 1).sum() - tp
    tn = (labels == 0).sum() - fp
    macro_f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1)
                + 2 * tn / np.maximum(2 * tn + fp + fn, 1)) / 2
    accuracy = (tp + tn) / len(labels)
    order = np.lexsort((thresholds, abs(thresholds - np.mean(means)), -accuracy, -macro_f1))
    best = order[0]
    return {'direction': direction, 'threshold': float(thresholds[best]),
            'train_macro_f1': float(macro_f1[best])}
