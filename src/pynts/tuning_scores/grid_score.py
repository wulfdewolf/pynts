import warnings
from typing import Optional

import cv2
import numpy as np
import pynapple as nap
from numba import njit
from numpy.typing import ArrayLike
from scipy.ndimage import rotate
from scipy.signal import correlate
from scipy.stats import circmean
from skimage.feature.peak import peak_local_max

from pynts.smoothing import apply_smoothing


def classify_grid_score(grid_info, null_distribution, alpha=0.05):
    return {
        "sig": grid_info["grid_score"]
        > np.nanpercentile(null_distribution["grid_score"], 100 * (1 - alpha)),
        "pval": (
            np.nansum(null_distribution["grid_score"] >= grid_info["grid_score"]) + 1
        )
        / (len(null_distribution["grid_score"]) + 1),
    }


def compute_grid_score(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    num_bins: Optional[int] = None,
    bin_size: float = 2.5,
    range: Optional[ArrayLike] = None,
    smooth_sigma: float | ArrayLike = 2,
    epoch: Optional[nap.IntervalSet] = None,
    is_shuffle: bool = False,
    ellipse_transform: bool = False,
):
    """
    Computes the grid score for a given cluster.
    Based on the description in:
    https://www.biorxiv.org/content/10.1101/230250v1.full.pdf
    """
    if epoch is None:
        epoch = cluster.time_support

    range = (
        [
            (np.nanmin(session["P_x"]), np.nanmax(session["P_x"])),
            (np.nanmin(session["P_y"]), np.nanmax(session["P_y"])),
        ]
        if range is None
        else range
    )
    P = np.stack([session["P_x"], session["P_y"]], axis=1)
    if num_bins is None:
        bins = [int((dim_range[1] - dim_range[0]) // bin_size) for dim_range in range]
    else:
        bins = num_bins
    max_bins = np.max(np.array(bins))

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster,
            P,
            bins=bins,
            range=range,
            epochs=epochs.intersect(session["moving"]),
        )[0]

    tc, smooth_sigma = apply_smoothing(
        compute_tuning_curve,
        epoch=epoch,
        dim=2,
        smooth_sigma=smooth_sigma,
        sigma_range=np.linspace(1, 4, 20),
        mode="fill",
        keep=True,
    )
    center = tc.shape
    autocorr = autocorr2d(tc.values)
    peaks = peak_local_max(
        np.nan_to_num(autocorr),
        min_distance=2,
        exclude_border=True,
    )
    if len(peaks) < 7:
        return {
            "grid_score": np.nan,
            "field_size": np.nan,
            "_smooth_sigma": smooth_sigma,
            "_ellipse_transform": ellipse_transform,
        }

    peaks_xy = peaks[:, [1, 0]].astype(np.float32)
    center_xy = np.array([center[1], center[0]], dtype=np.float32)

    distances = np.linalg.norm(peaks_xy - center_xy, axis=1)
    sorted_idx = np.argsort(distances)[1:7]
    peaks_xy = peaks_xy[sorted_idx]
    distances = distances[sorted_idx]

    if ellipse_transform and not is_shuffle:
        autocorr, peaks_xy = ellipse_to_circle_transform(
            np.nan_to_num(autocorr, 0.0), peaks_xy, center_xy
        )
        distances = np.array([np.linalg.norm(center - peak) for peak in peaks_xy])

    # Define the ring size
    mean_distance = np.mean(distances)
    inner_radius = mean_distance * 0.5
    outer_radius = mean_distance * 1.25

    # Extract a ring around the center
    y, x = np.ogrid[: autocorr.shape[0], : autocorr.shape[1]]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 >= inner_radius**2
    mask &= (x - center[1]) ** 2 + (y - center[0]) ** 2 <= outer_radius**2
    ring = np.where(mask, autocorr, np.nan)

    data = ring.copy()
    valid = ~np.isnan(data)

    data0 = np.nan_to_num(data, nan=0.0)
    valid0 = valid.astype(float)

    angle_scores = {}
    angles = [30, 60, 90, 120, 150]
    for angle in angles:
        # rotate signal
        rot_data = rotate(
            data0,
            angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )

        # rotate validity mask separately
        rot_valid = (
            rotate(
                valid0,
                angle,
                reshape=False,
                order=0,
                mode="constant",
                cval=0.0,
            )
            > 0.5
        )

        # intersection of valid pixels in both frames
        combined_mask = valid & rot_valid

        if np.sum(combined_mask) < 10:
            angle_scores[angle] = np.nan
            continue

        angle_scores[angle] = np.corrcoef(data[combined_mask], rot_data[combined_mask])[
            0, 1
        ]

    # Compute the grid score as the difference between the minimum correlation
    # coefficient for rotations of 60 and 120 degrees and the maximum correlation
    # coefficient for rotations of 30, 90, and 150 degrees
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="All-Nan axis encountered"
        )
        scale = (range[0][1] - range[0][0]) / max_bins
        return {
            "grid_score": np.nanmin([angle_scores[60], angle_scores[120]])
            - np.nanmax([angle_scores[30], angle_scores[90], angle_scores[150]]),
            "field_size": (outer_radius - inner_radius) * scale,
            "field_spacing": mean_distance * scale,
            "orientation": circmean(
                np.mod(
                    np.arctan2(peaks[:, 0] - center[0], peaks[:, 1] - center[1]),
                    np.pi / 3,
                ),
                high=np.pi / 3,
                low=0,
            ),
            "_smooth_sigma": smooth_sigma,
            "_ellipse_transform": ellipse_transform,
        }


@njit
def autocorr2d(lambda_matrix, min_n=20):
    rows, cols = lambda_matrix.shape  # row-major: rows (height), cols (width)
    max_tau_x = 2 * (cols - 1)
    max_tau_y = 2 * (rows - 1)

    # Use shape (max_tau_y+1, max_tau_x+1) so first index is tau_y (rows), second is tau_x (cols)
    autocorr_map = np.full((max_tau_y + 1, max_tau_x + 1), np.nan)

    for tau_x in range(-cols + 1, cols):
        for tau_y in range(-rows + 1, rows):
            sum_lambda = 0.0
            sum_lambda_tau = 0.0
            sum_lambda_product = 0.0
            sum_lambda_sq = 0.0
            sum_lambda_tau_sq = 0.0
            n = 0

            for row in range(rows):
                for col in range(cols):
                    r2 = row + tau_y
                    c2 = col + tau_x
                    if 0 <= c2 < cols and 0 <= r2 < rows:
                        val = lambda_matrix[row, col]
                        val_tau = lambda_matrix[r2, c2]
                        if not np.isnan(val) and not np.isnan(val_tau):
                            sum_lambda += val
                            sum_lambda_tau += val_tau
                            sum_lambda_product += val * val_tau
                            sum_lambda_sq += val * val
                            sum_lambda_tau_sq += val_tau * val_tau
                            n += 1

            if n < min_n:
                continue

            num = n * sum_lambda_product - sum_lambda * sum_lambda_tau
            den = (n * sum_lambda_sq - sum_lambda * sum_lambda) * (
                n * sum_lambda_tau_sq - sum_lambda_tau * sum_lambda_tau
            )
            if den <= 0.0:
                autocorr = np.nan
            else:
                autocorr = num / np.sqrt(den)

            # store with tau_y as row index and tau_x as col index
            autocorr_map[tau_y + rows - 1, tau_x + cols - 1] = autocorr

    return autocorr_map


def ellipse_to_circle_transform(autocorr, peaks_xy, center_xy):
    """
    Transform elliptical grid pattern to circular.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        try:
            ellipse_params = cv2.fitEllipse(peaks_xy)
        except Exception as _:
            return autocorr, peaks_xy

    ellipse_center, axes, angle = ellipse_params

    # Determine major/minor axes
    axis1, axis2 = axes
    if axis1 > axis2:
        major_axis = axis1
        minor_axis = axis2
        major_angle = angle
    else:
        major_axis = axis2
        minor_axis = axis1
        major_angle = angle + 90

    scale_ratio = minor_axis / major_axis

    # print(f"Ellipse fit: major={major_axis:.1f}, minor={minor_axis:.1f}, ratio={scale_ratio:.2f}, angle={major_angle:.1f}°")

    # Safety checks
    if scale_ratio > 0.85:
        return autocorr, peaks_xy

    if scale_ratio < 0.3:
        # print("Warning: Too elliptical - likely bad fit")
        return autocorr, peaks_xy

    # Build transformation matrix around image center
    angle_rad = np.deg2rad(major_angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Use the center of the autocorr as transformation origin
    cx, cy = center_xy

    # Build transformation matrix
    M = np.array(
        [
            [
                cos_a**2 * scale_ratio + sin_a**2,
                cos_a * sin_a * (scale_ratio - 1),
                cx
                - cx * (cos_a**2 * scale_ratio + sin_a**2)
                - cy * cos_a * sin_a * (scale_ratio - 1),
            ],
            [
                cos_a * sin_a * (scale_ratio - 1),
                sin_a**2 * scale_ratio + cos_a**2,
                cy
                - cx * cos_a * sin_a * (scale_ratio - 1)
                - cy * (sin_a**2 * scale_ratio + cos_a**2),
            ],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # Transform peaks
    peaks_homogeneous = np.hstack([peaks_xy, np.ones((peaks_xy.shape[0], 1))])
    peaks_transformed_xy = (peaks_homogeneous @ M.T)[:, :2]

    # Keep the same output size as input
    # Use BORDER_REPLICATE to avoid white borders
    autocorr_transformed = cv2.warpAffine(
        autocorr.astype(np.float64),
        M[:2, :],
        (autocorr.shape[1], autocorr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # Replicate edge values instead of filling with 0
    )

    return autocorr_transformed, peaks_transformed_xy
