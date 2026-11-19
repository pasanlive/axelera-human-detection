r"""
Sri Lankan DMT License Plate Pre-processing, Post-processing, and Correction.

Implements:
1. Perspective deskew via homography when corner quad points are available.
2. 2-line square plate detection (aspect ratio < 2.0) with horizontal splitting and side-by-side stitching.
3. DMT character validation & positional correction:
   - Optional 2-letter Sri Lankan province prefix: WP, CP, SP, NW, NC, UP, SG, EP, NP.
   - Positional character correction:
     * Last 4 characters must be digits (O/Q -> 0, I/L -> 1, Z -> 2, S -> 5, B -> 8, G -> 6).
     * Preceding 2-3 characters must be letters (0 -> O, 1 -> I, 2 -> Z, 5 -> S, 8 -> B, 6 -> G).
   - Regex validation: ^(?:(WP|CP|SP|NW|NC|UP|SG|EP|NP)\s)?[A-Z]{2,3}-\d{4}$
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

# Official 9 Sri Lankan Provinces
SRI_LANKA_PROVINCES = frozenset({"WP", "CP", "SP", "NW", "NC", "UP", "SG", "EP", "NP"})

# DMT Plate Validation Pattern
DMT_PLATE_REGEX = re.compile(
    r"^(?:(WP|CP|SP|NW|NC|UP|SG|EP|NP)\s)?[A-Z]{2,3}-\d{4}$"
)

# Character confusion mappings
LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
    "A": "4",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
    "4": "A",
}


def deskew_plate_crop(
    crop: np.ndarray,
    corners: Optional[Union[np.ndarray, List[List[float]]]] = None,
) -> np.ndarray:
    """
    Applies perspective deskew / homography on plate crop if 4 corner points are provided.
    Corners should be ordered: [top-left, top-right, bottom-right, bottom-left].
    If no valid corners are provided, returns the crop as-is.
    """
    if crop is None or crop.size == 0:
        return crop

    if corners is None:
        return crop

    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        return crop

    # Order points: [tl, tr, br, bl]
    # Sum of coordinates: tl has smallest, br has largest
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # Difference of coordinates: tr has smallest (x - y is largest), bl has largest (y - x is largest)
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_w = max(int(max(width_top, width_bottom)), 16)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_h = max(int(max(height_left, height_right)), 16)

    dst_pts = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )

    try:
        M = cv2.getPerspectiveTransform(ordered, dst_pts)
        deskewed = cv2.warpPerspective(crop, M, (max_w, max_h), flags=cv2.INTER_LINEAR)
        return deskewed
    except Exception:
        return crop


def split_and_stitch_two_line_plate(
    crop: np.ndarray,
    two_line_aspect_thresh: float = 2.0,
    target_height: int = 48,
) -> np.ndarray:
    """
    Detects 2-line square plates (e.g. motorcycles, tuk-tuks, SUVs) where width / height < 2.0.
    Splits the crop horizontally into top (province/letters) and bottom (digits) strips,
    resizes both to target_height, and concatenates them side-by-side into a single-line image
    for OCR.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop

    aspect = float(w) / float(h)
    if aspect >= two_line_aspect_thresh:
        # Already single-line rectangular plate
        return crop

    # 2-line square plate: horizontal split at midpoint (50%)
    split_y = h // 2
    top_strip = crop[:split_y, :]
    bottom_strip = crop[split_y:, :]

    if top_strip.size == 0 or bottom_strip.size == 0:
        return crop

    # Resize strips to uniform height while preserving strip aspect ratios
    th_top, tw_top = top_strip.shape[:2]
    th_bot, tw_bot = bottom_strip.shape[:2]

    new_w_top = max(16, int(tw_top * (target_height / max(1, th_top))))
    new_w_bot = max(16, int(tw_bot * (target_height / max(1, th_bot))))

    top_resized = cv2.resize(top_strip, (new_w_top, target_height), interpolation=cv2.INTER_LINEAR)
    bottom_resized = cv2.resize(bottom_strip, (new_w_bot, target_height), interpolation=cv2.INTER_LINEAR)

    # Concatenate side-by-side: top strip on left, bottom strip on right
    stitched = np.hstack([top_resized, bottom_resized])
    return stitched


def preprocess_plate_crop(
    crop: np.ndarray,
    corners: Optional[Union[np.ndarray, List[List[float]]]] = None,
    two_line_aspect_thresh: float = 2.0,
    target_height: int = 48,
    tight_padding_pct: float = 0.04,
) -> np.ndarray:
    """
    Combined plate pre-processing:
    1. Perspective deskew via homography if corners are provided.
    2. Tight safety padding without clipping characters.
    3. 2-line square plate splitting and side-by-side horizontal stitching.
    """
    if crop is None or crop.size == 0:
        return crop

    # Step 1: Deskew if corners available
    if corners is not None:
        crop = deskew_plate_crop(crop, corners)
    elif tight_padding_pct > 0.0:
        # Ensure tight padding without clipping
        h, w = crop.shape[:2]
        pad_y = max(1, int(h * tight_padding_pct))
        pad_x = max(1, int(w * tight_padding_pct))
        crop = cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)

    # Step 2: 2-line plate split & stitch
    crop = split_and_stitch_two_line_plate(
        crop,
        two_line_aspect_thresh=two_line_aspect_thresh,
        target_height=target_height,
    )
    return crop


def clean_and_correct_dmt_plate(raw_text: str) -> Optional[Tuple[str, float]]:
    r"""
    Post-processing for Sri Lankan DMT number plates:
    1. Strips spaces, hyphens, and non-alphanumeric noise.
    2. Identifies optional 2-letter province prefix (WP, CP, SP, NW, NC, UP, SG, EP, NP).
    3. Enforces positional character correction:
       - Last 4 characters must be digits (e.g. O/Q -> 0, I/L -> 1, Z -> 2, S -> 5, B -> 8, G -> 6).
       - Preceding 2-3 characters must be letters (e.g. 0 -> O, 1 -> I, 2 -> Z, 5 -> S, 8 -> B, 6 -> G).
    4. Validates against pattern ^(?:(WP|CP|SP|NW|NC|UP|SG|EP|NP)\s)?[A-Z]{2,3}-\d{4}$
    
    Returns:
        (formatted_plate_str, confidence_factor) or None if invalid.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip().upper()
    if not cleaned:
        return None

    # Check for explicit token separation (e.g. "WP CAA-1234", "WP-CAA-1234", "WP CAA 1234")
    tokens = [t for t in re.split(r"[\s\-_]+", cleaned) if t]
    province: Optional[str] = None
    remaining_tokens: List[str] = []

    if tokens and tokens[0] in SRI_LANKA_PROVINCES:
        province = tokens[0]
        remaining_tokens = tokens[1:]
    else:
        remaining_tokens = tokens

    # Recombine remaining alphanumeric characters
    body_raw = re.sub(r"[^A-Z0-9]", "", "".join(remaining_tokens))

    # If province was not found in tokens, check if raw string starts with a province code
    # Valid body lengths are 6 (2 letters + 4 digits) or 7 (3 letters + 4 digits).
    # If total length is 8 or 9, first 2 characters could be province.
    if province is None:
        full_alnum = re.sub(r"[^A-Z0-9]", "", cleaned)
        if len(full_alnum) in (8, 9) and full_alnum[:2] in SRI_LANKA_PROVINCES:
            province = full_alnum[:2]
            body_raw = full_alnum[2:]
        else:
            body_raw = full_alnum

    # Body must have 6 or 7 characters (2 or 3 letters + 4 digits)
    if len(body_raw) not in (6, 7):
        return None

    num_letters = len(body_raw) - 4
    raw_letters = body_raw[:num_letters]
    raw_digits = body_raw[num_letters:]

    corrections_applied = 0

    # 1. Preceding 2-3 characters must be letters
    corrected_letters_list = []
    for ch in raw_letters:
        if ch.isdigit():
            rep = DIGIT_TO_LETTER.get(ch, ch)
            if rep != ch:
                corrections_applied += 1
            corrected_letters_list.append(rep)
        else:
            corrected_letters_list.append(ch)
    corrected_letters = "".join(corrected_letters_list)

    # 2. Last 4 characters must be digits
    corrected_digits_list = []
    for ch in raw_digits:
        if ch.isalpha():
            rep = LETTER_TO_DIGIT.get(ch, ch)
            if rep != ch:
                corrections_applied += 1
            corrected_digits_list.append(rep)
        else:
            corrected_digits_list.append(ch)
    corrected_digits = "".join(corrected_digits_list)

    # Construct final candidate
    if province:
        candidate = f"{province} {corrected_letters}-{corrected_digits}"
    else:
        candidate = f"{corrected_letters}-{corrected_digits}"

    # Validate against regex
    if not DMT_PLATE_REGEX.match(candidate):
        return None

    # Base confidence weight: slight penalty if many character substitutions were required
    conf_factor = max(0.6, 1.0 - (0.08 * corrections_applied))
    return candidate, conf_factor
