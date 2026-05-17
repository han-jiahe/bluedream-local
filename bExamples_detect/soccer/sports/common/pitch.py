# pitch.py - Robust Pitch Registration
from collections import deque
import cv2
import numpy as np

try:
    from sports.configs.soccer import SoccerPitchConfiguration
except ImportError:
    class SoccerPitchConfiguration:
        width = 6800
        length = 10500
        @property
        def vertices(self):
            return [
                (0, 0), (0, 1384), (0, 2484), (0, 4316), (0, 5416), (0, 6800),
                (550, 2484), (550, 4316), (1100, 3400),
                (1650, 1384), (1650, 2484), (1650, 4316), (1650, 5416),
                (5250, 0), (5250, 2485), (5250, 4315), (5250, 6800),
                (8850, 1384), (8850, 2484), (8850, 4316), (8850, 5416),
                (9400, 3400), (9950, 2484), (9950, 4316),
                (10500, 0), (10500, 1384), (10500, 2484), (10500, 4316), (10500, 5416), (10500, 6800),
                (4335, 3400), (6165, 3400),
            ]


class PitchRegistrator:
    """Robust pitch-to-image registration with multi-source fusion."""

    def __init__(self, config=None, smoothing_alpha=0.65, min_keypoints=5,
                 ransac_threshold=3.0, temporal_buffer_size=5):
        self.config = config or SoccerPitchConfiguration()
        self.smoothing_alpha = smoothing_alpha
        self.min_keypoints = min_keypoints
        self.ransac_threshold = ransac_threshold
        self._H = None
        self._H_buffer = deque(maxlen=temporal_buffer_size)
        self._last_valid_H = None
        self._last_source = "none"  # "keypoints", "lines", "partial", "fallback"
        self._target_pts = np.array(self.config.vertices, dtype=np.float32)

    def register(self, frame, keypoints_xy=None, keypoints_conf=None, conf_threshold=0.5):
        if keypoints_xy is not None and len(keypoints_xy) > 0:
            H = self._register_from_keypoints(keypoints_xy, keypoints_conf, conf_threshold)
            if H is not None:
                return self._commit(H, source="keypoints")
        H = self._register_from_lines(frame)
        if H is not None:
            return self._commit(H, source="lines")
        if keypoints_xy is not None and len(keypoints_xy) >= 2:
            H = self._register_from_partial(keypoints_xy, keypoints_conf, conf_threshold)
            if H is not None:
                return self._commit(H, source="partial")
        if self._last_valid_H is not None:
            self._H = self._last_valid_H
            return self._last_valid_H
        return None

    def _commit(self, H, source="unknown"):
        self._last_source = source
        H = self._apply_temporal_smoothing(H)
        self._H = H
        self._last_valid_H = H
        self._H_buffer.append(H)
        return H

    def transform(self, points):
        if self._H is None:
            raise ValueError("No homography. Call register() first.")
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self._H)
        return transformed.reshape(-1, 2).astype(np.float32)

    @property
    def H(self):
        return self._H

    def reset(self):
        self._H = None
        self._H_buffer.clear()
        self._last_valid_H = None

    def _register_from_keypoints(self, kpts_xy, kpts_conf, conf_threshold):
        n_kpts = kpts_xy.shape[0]
        if n_kpts < self.min_keypoints:
            return None
        if kpts_conf is not None and len(kpts_conf) == n_kpts:
            valid = kpts_conf > conf_threshold
            src = kpts_xy[valid]
            dst = self._target_pts[valid]
        else:
            valid = (kpts_xy[:, 0] > 1) & (kpts_xy[:, 1] > 1)
            src = kpts_xy[valid]
            dst = self._target_pts[valid]
        if len(src) < 4:
            return None
        H, mask = cv2.findHomography(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=2000, confidence=0.995)
        if H is None:
            return None
        if mask is not None and np.sum(mask) / len(mask) < 0.4:
            return None
        return H

    def _register_from_lines(self, frame):
        h, w = frame.shape[:2]
        green_mask = self._green_mask(frame)
        green_ratio = np.sum(green_mask) / (h * w)
        if green_ratio < 0.05:
            return None
        lines = self._detect_pitch_lines(frame, green_mask)
        if lines is None or len(lines) < 4:
            return None
        h_lines, v_lines = self._classify_lines(lines)
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None
        corners = self._find_corner_intersections(h_lines, v_lines)
        if corners is None or len(corners) < 4:
            return None
        corners = self._order_corners(corners)
        if corners is None:
            return None
        pitch_corners = np.array(
            [[0, 0], [self.config.length, 0],
             [self.config.length, self.config.width],
             [0, self.config.width]], dtype=np.float32)
        H, _ = cv2.findHomography(
            corners, pitch_corners, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold)
        return H

    def _green_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([35, 40, 40], dtype=np.uint8)
        upper = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask > 0

    def _detect_pitch_lines(self, frame, green_mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, white_mask = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), dtype=np.uint8)
        gd = cv2.dilate(green_mask.astype(np.uint8) * 255, kernel, iterations=1)
        white_on_green = cv2.bitwise_and(white_mask, gd)
        edges = cv2.Canny(white_on_green, 50, 150)
        for thresh in (60, 40, 25):
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=thresh,
                minLineLength=min(frame.shape[0] // 6, 50),
                maxLineGap=max(frame.shape[0] // 20, 10))
            if lines is not None and len(lines) >= 4:
                return lines
        return lines

    @staticmethod
    def _classify_lines(lines):
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 20 or angle > 160:
                h_lines.append((y1 + y2) / 2)
            elif 70 < angle < 110:
                v_lines.append((x1 + x2) / 2)
        return h_lines, v_lines

    def _find_corner_intersections(self, h_ys, v_xs):
        if len(h_ys) < 2 or len(v_xs) < 2:
            return None
        h_arr = np.array(h_ys, dtype=np.float64)
        if len(h_arr) >= 4:
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                KMeans = None
            if KMeans is not None:
                km = KMeans(n_clusters=2, n_init=10, random_state=0)
                labels = km.fit_predict(h_arr.reshape(-1, 1))
                g0 = h_arr[labels == 0]; g1 = h_arr[labels == 1]
            else:
                # Simple percentile-based split
                median = float(np.median(h_arr))
                g0 = h_arr[h_arr <= median]
                g1 = h_arr[h_arr > median]
            if np.median(g0) < np.median(g1):
                top_y, bottom_y = float(np.median(g0)), float(np.median(g1))
            else:
                top_y, bottom_y = float(np.median(g1)), float(np.median(g0))
        else:
            top_y = float(np.min(h_arr))
            bottom_y = float(np.max(h_arr))
        v_arr = np.array(v_xs, dtype=np.float64)
        if len(v_arr) >= 4:
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                KMeans = None
            if KMeans is not None:
                km = KMeans(n_clusters=2, n_init=10, random_state=0)
                labels = km.fit_predict(v_arr.reshape(-1, 1))
                g0 = v_arr[labels == 0]; g1 = v_arr[labels == 1]
            else:
                median = float(np.median(v_arr))
                g0 = v_arr[v_arr <= median]
                g1 = v_arr[v_arr > median]
            if np.median(g0) < np.median(g1):
                left_x, right_x = float(np.median(g0)), float(np.median(g1))
            else:
                left_x, right_x = float(np.median(g1)), float(np.median(g0))
        else:
            left_x = float(np.min(v_arr))
            right_x = float(np.max(v_arr))
        return np.array([[left_x, top_y], [right_x, top_y],
                         [right_x, bottom_y], [left_x, bottom_y]], dtype=np.float32)

    @staticmethod
    def _order_corners(pts):
        if len(pts) != 4:
            return None
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _register_from_partial(self, kpts_xy, kpts_conf, conf_threshold):
        if kpts_conf is not None and len(kpts_conf) == len(kpts_xy):
            valid = kpts_conf > conf_threshold
            src = kpts_xy[valid]
            dst = self._target_pts[valid]
        else:
            valid = (kpts_xy[:, 0] > 1) & (kpts_xy[:, 1] > 1)
            src = kpts_xy[valid]
            dst = self._target_pts[valid]
        n = len(src)
        if n < 2:
            return None
        if n >= 3:
            H, _ = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold)
            if H is not None:
                H_full = np.eye(3, dtype=np.float32)
                H_full[:2] = H
                return H_full
        if n == 2 and self._last_valid_H is not None:
            H = self._estimate_similarity(src, dst, self._last_valid_H)
            if H is not None:
                return H
        return None

    @staticmethod
    def _estimate_similarity(src, dst, prior_H):
        if len(src) != 2:
            return None
        src_center = np.mean(src, axis=0)
        dst_center = np.mean(dst, axis=0)
        src_vec = src[1] - src[0]
        dst_vec = dst[1] - dst[0]
        src_angle = np.arctan2(src_vec[1], src_vec[0])
        dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
        rotation = dst_angle - src_angle
        src_len = np.linalg.norm(src_vec)
        dst_len = np.linalg.norm(dst_vec)
        if src_len < 1e-6:
            return None
        scale = dst_len / src_len
        if prior_H is not None:
            prior_scale = np.linalg.norm(prior_H[:2, 0])
            if prior_scale > 0:
                scale = np.clip(scale, prior_scale * 0.5, prior_scale * 2.0)
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        H = np.eye(3, dtype=np.float32)
        H[:2, :2] = R * scale
        H[:2, 2] = dst_center - (scale * R @ src_center)
        return H

    def _apply_temporal_smoothing(self, H_new):
        if self._last_valid_H is None:
            return H_new
        delta = np.linalg.norm(H_new - self._last_valid_H)
        alpha = self.smoothing_alpha
        if delta > 0.3:
            alpha = 0.3
        return alpha * self._last_valid_H + (1 - alpha) * H_new

    def reprojection_error(self, src, dst):
        if self._H is None or len(src) == 0:
            return float("inf")
        projected = self.transform(src)
        errors = np.linalg.norm(projected - dst, axis=1)
        return float(np.mean(errors))
