"""Inference-safe lower-limb features for neuropathic gait screening."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def _ratio(a, b):
    return np.abs(a) / (np.abs(b) + 1e-6)


def _magnitude(df: pd.DataFrame, stat: str, landmark: int) -> pd.Series:
    x = _col(df, f"{stat}_lm{landmark}_x")
    y = _col(df, f"{stat}_lm{landmark}_y")
    return np.sqrt(x * x + y * y)


def _angle(ax, ay, bx, by, cx, cy):
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    cosine = (v1x * v2x + v1y * v2y) / (
        np.sqrt(v1x * v1x + v1y * v1y)
        * np.sqrt(v2x * v2x + v2y * v2y)
        + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def add_neuropathic_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add steppage, distal-motion and left/right-asymmetry proxies."""
    out = df.copy()
    features: dict[str, pd.Series] = {}
    pairs = {
        "hip": (23, 24),
        "knee": (25, 26),
        "ankle": (27, 28),
        "heel": (29, 30),
        "foot": (31, 32),
    }

    magnitudes: dict[tuple[str, str, str], pd.Series] = {}
    for stat in ["range", "std", "mean_abs_vel"]:
        for part, (left_id, right_id) in pairs.items():
            left = _magnitude(out, stat, left_id)
            right = _magnitude(out, stat, right_id)
            magnitudes[(stat, part, "left")] = left
            magnitudes[(stat, part, "right")] = right
            features[f"neuro_left_{part}_{stat}_xy"] = left
            features[f"neuro_right_{part}_{stat}_xy"] = right
            features[f"neuro_{part}_{stat}_asymmetry"] = np.abs(left - right)
            features[f"neuro_{part}_{stat}_ratio"] = _ratio(left, right)

    for stat in ["range", "mean_abs_vel"]:
        for side in ["left", "right"]:
            hip = magnitudes[(stat, "hip", side)]
            knee = magnitudes[(stat, "knee", side)]
            ankle = magnitudes[(stat, "ankle", side)]
            foot = magnitudes[(stat, "foot", side)]
            features[f"neuro_{side}_ankle_to_hip_{stat}_ratio"] = _ratio(ankle, hip)
            features[f"neuro_{side}_foot_to_hip_{stat}_ratio"] = _ratio(foot, hip)
            features[f"neuro_{side}_foot_to_knee_{stat}_ratio"] = _ratio(foot, knee)

    left_foot_y = _col(out, "mean_lm31_y")
    right_foot_y = _col(out, "mean_lm32_y")
    left_ankle_y = _col(out, "mean_lm27_y")
    right_ankle_y = _col(out, "mean_lm28_y")
    features["neuro_left_foot_ankle_y_offset"] = left_foot_y - left_ankle_y
    features["neuro_right_foot_ankle_y_offset"] = right_foot_y - right_ankle_y
    features["neuro_foot_ankle_y_offset_asymmetry"] = np.abs(
        features["neuro_left_foot_ankle_y_offset"]
        - features["neuro_right_foot_ankle_y_offset"]
    )

    left_angle = _angle(
        _col(out, "mean_lm23_x"), _col(out, "mean_lm23_y"),
        _col(out, "mean_lm25_x"), _col(out, "mean_lm25_y"),
        _col(out, "mean_lm27_x"), _col(out, "mean_lm27_y"),
    )
    right_angle = _angle(
        _col(out, "mean_lm24_x"), _col(out, "mean_lm24_y"),
        _col(out, "mean_lm26_x"), _col(out, "mean_lm26_y"),
        _col(out, "mean_lm28_x"), _col(out, "mean_lm28_y"),
    )
    features["neuro_left_knee_angle_proxy"] = left_angle
    features["neuro_right_knee_angle_proxy"] = right_angle
    features["neuro_knee_angle_asymmetry"] = np.abs(left_angle - right_angle)

    feature_df = pd.DataFrame(features, index=out.index)
    out = pd.concat([out, feature_df], axis=1)
    return out, list(feature_df.columns)
