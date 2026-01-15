import warnings

import cv2
import numpy as np
import pynapple as nap
from numba import njit
from scipy.ndimage import rotate
from scipy.stats import circmean
from skimage.feature.peak import peak_local_max

from pynts.util import gaussian_filter_nan
from pynts.wrappers import find_optimal_smoothing


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
    session,
    session_type,
    cluster,
    num_bins,
    bounds=None,
    do_ellipse_transform=False,
    smooth_sigma=True,
    epoch=None,
):
    """
    Computes the grid score for a given cluster.
    Based on the description in:
        https://www.biorxiv.org/content/10.1101/230250v1.full.pdf
    """
    if epoch is None:
        epoch = cluster.time_support

    P = np.stack([session["P_x"], session["P_y"]], axis=1)
    if bounds == None:
        bounds = [(np.nanmin(P.values), np.nanmax(P.values))] * 2

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster,
            P,
            bins=num_bins,
            range=bounds,
            epochs=epochs.intersect(session["moving"]),
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        if isinstance(smooth_sigma, bool) and smooth_sigma:
            smooth_sigma = [0] + [
                find_optimal_smoothing(
                    compute_tuning_curve,
                    epoch,
                    np.arange(
                        int(num_bins // 8),
                    ),
                    mode="reflect",
                )
            ] * 2
        tc = compute_tuning_curve(epoch)
        if smooth_sigma:
            tc = gaussian_filter_nan(
                tc,
                smooth_sigma,
                mode="reflect",
            )
    tc = tc[0]
    center = tc.shape
    autocorr = autocorr2d(tc.values)
    peaks = peak_local_max(
        np.nan_to_num(autocorr),
        min_distance=4,
        exclude_border=True,
    )
    if len(peaks) < 7:
        return {"grid_score": np.nan, "field_size": np.nan}
    distances = np.array([np.linalg.norm(center - peak) for peak in peaks])
    sorted = np.argsort(distances)[1:7]
    peaks = peaks[sorted]
    distances = distances[sorted]
    if do_ellipse_transform:
        autocorr, peaks = ellipse_to_circle_transform(np.nan_to_num(autocorr), peaks)
        distances = np.array([np.linalg.norm(center - peak) for peak in peaks])

    # Define the ring size
    mean_distance = np.mean(distances)
    inner_radius = mean_distance * 0.5
    outer_radius = mean_distance * 1.25

    # Extract a ring around the center
    y, x = np.ogrid[: autocorr.shape[0], : autocorr.shape[1]]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 >= inner_radius**2
    mask &= (x - center[1]) ** 2 + (y - center[0]) ** 2 <= outer_radius**2
    ring = np.where(mask, autocorr, np.nan)

    # Compute the rotational symmetry of the autocorrelation map
    angles = [30, 60, 90, 120, 150]
    angle_scores = {}

    valid_mask = ~np.isnan(ring)
    ring_filled = np.nan_to_num(ring, nan=0.0)

    for angle in angles:
        rotated_ring = rotate(
            ring_filled, angle, reshape=False, mode="constant", cval=0.0
        )
        rotated_mask = (
            rotate(
                valid_mask.astype(float),
                angle,
                reshape=False,
                mode="constant",
                cval=0.0,
            )
            >= 0.5
        )

        # Re-apply the ring mask
        combined_mask = mask & rotated_mask & valid_mask
        if np.sum(combined_mask) < 10:
            angle_scores[angle] = np.nan
            continue

        angle_scores[angle] = np.corrcoef(
            ring[combined_mask], rotated_ring[combined_mask]
        )[0, 1]

    # Compute the grid score as the difference between the minimum correlation
    # coefficient for rotations of 60 and 120 degrees and the maximum correlation
    # coefficient for rotations of 30, 90, and 150 degrees
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="All-Nan axis encountered"
        )
        scale = (bounds[0][1] - bounds[0][0]) / num_bins
        return {
            "grid_score": np.nanmin([angle_scores[60], angle_scores[120]])
            - np.nanmax([angle_scores[30], angle_scores[90], angle_scores[150]]),
            "field_size": (outer_radius - inner_radius) * scale,
            "field_spacing": mean_distance * scale,
            "orientation": circmean(
                np.arctan2(peaks[:, 0] - center[0], peaks[:, 1] - center[1]),
                high=np.pi,
                low=-np.pi,
            ),
            "_smooth_sigma": smooth_sigma,
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


def ellipse_to_circle_transform(autocorr, peaks):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        center, (major_axis, minor_axis), angle = cv2.fitEllipse(peaks)
    angle_rad = np.deg2rad(angle)

    # Get the scaling factors
    if major_axis == 0 or minor_axis == 0:
        return autocorr, peaks
    scale_x = (
        minor_axis / major_axis
    )  # Scale the x-axis to match the y-axis (minor axis)
    scale_y = 1.0

    # Translation to origin
    T1 = np.array(
        [[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]],
        dtype=np.float32,
    )

    # Rotation
    R1 = np.array(
        [
            [np.cos(angle_rad), np.sin(angle_rad), 0],
            [-np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Scaling
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)

    # Inverse rotation
    R2 = np.array(
        [
            [np.cos(-angle_rad), np.sin(-angle_rad), 0],
            [-np.sin(-angle_rad), np.cos(-angle_rad), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Translation back
    T2 = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]], dtype=np.float32)

    # Combine all transformations: T2 * R2 * S * R1 * T1
    M = T2 @ R2 @ S @ R1 @ T1

    # Transform
    peaks_transformed = (np.hstack([peaks, np.ones((peaks.shape[0], 1))]) @ M.T)[:, :2]

    autocorr_transformed = cv2.warpAffine(
        autocorr, M[:2, :], (autocorr.shape[1], autocorr.shape[0])
    )
    return autocorr_transformed, peaks_transformed
