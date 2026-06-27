"""UI 模板匹配器 — 轻量级纯模板匹配方案"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TemplateMatcher:
    """
    纯模板匹配方案

    适用于:
    - 安卓车机 Dark/Light 模式切换
    - 特定图标/按钮出现/消失检测
    - 区域状态比对
    """

    MATCH_METHODS = {
        "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
        "ccorr_normed": cv2.TM_CCORR_NORMED,
        "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
    }

    def __init__(self, templates_dir: Optional[str] = None):
        self.templates: dict[str, np.ndarray] = {}
        if templates_dir:
            self.load_templates(templates_dir)

    def load_templates(self, directory: str):
        """从目录加载所有模板图像"""
        path = Path(directory)
        if not path.exists():
            logger.warning("模板目录不存在: %s", directory)
            return

        for p in path.glob("*.png"):
            img = cv2.imread(str(p))
            if img is not None:
                name = p.stem
                self.templates[name] = img
                logger.info("加载模板: %s (%dx%d)", name, img.shape[1], img.shape[0])

    def match(self, screenshot: np.ndarray, template_name: str,
              method: str = "ccoeff_normed",
              threshold: float = 0.5) -> list[dict]:
        """
        在截图中查找模板匹配

        Returns: [{"bbox": (x1,y1,x2,y2), "confidence": float, "center": (x,y)}, ...]
        """
        template = self.templates.get(template_name)
        if template is None:
            logger.warning("模板不存在: %s", template_name)
            return []

        match_method = self.MATCH_METHODS.get(method, cv2.TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(screenshot, template, match_method)

        # 找到所有超过阈值的匹配
        locations = np.where(result >= threshold)
        h, w = template.shape[:2]
        matches = []

        for pt in zip(*locations[::-1]):
            x1, y1 = pt
            x2, y2 = x1 + w, y1 + h
            conf = float(result[y1, x1])

            # NMS: 过滤重叠框
            if not self._is_overlapping(matches, x1, y1, x2, y2):
                matches.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": round(conf, 4),
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "template": template_name,
                    "method": method,
                })

        return matches

    def multi_scale_match(self, screenshot: np.ndarray, template_name: str,
                          scales: list[float] = None,
                          threshold: float = 0.5) -> list[dict]:
        """多尺度模板匹配（处理不同分辨率）"""
        if scales is None:
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]

        template = self.templates.get(template_name)
        if template is None:
            return []

        all_matches = []
        h, w = template.shape[:2]

        for scale in scales:
            scaled = cv2.resize(template, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_LINEAR)
            if scaled.shape[0] > screenshot.shape[0] or scaled.shape[1] > screenshot.shape[1]:
                continue

            result = cv2.matchTemplate(screenshot, scaled, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                conf = float(result[pt[1], pt[0]])
                all_matches.append({
                    "bbox": (int(pt[0]), int(pt[1]),
                             int(pt[0] + scaled.shape[1]), int(pt[1] + scaled.shape[0])),
                    "confidence": round(conf, 4),
                    "center": (int(pt[0] + scaled.shape[1] / 2),
                               int(pt[1] + scaled.shape[0] / 2)),
                    "template": template_name,
                    "scale": scale,
                    "method": "multi_scale",
                })

        # NMS
        return self._nms(all_matches, iou_threshold=0.5)

    def detect_region_change(self, before: np.ndarray, after: np.ndarray,
                             region: tuple[int, int, int, int]) -> dict:
        """检测特定区域的变化程度"""
        x1, y1, x2, y2 = region
        roi_before = before[y1:y2, x1:x2]
        roi_after = after[y1:y2, x1:x2]

        if roi_before.shape != roi_after.shape:
            return {"changed": True, "change_score": 1.0, "error": "尺寸不匹配"}

        # 计算 MSE
        diff = cv2.absdiff(roi_before, roi_after)
        mse = float(np.mean(diff ** 2))
        normalized = min(1.0, mse / (255 ** 2))

        # 计算结构相似性 (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            gray_before = cv2.cvtColor(roi_before, cv2.COLOR_BGR2GRAY)
            gray_after = cv2.cvtColor(roi_after, cv2.COLOR_BGR2GRAY)
            ssim_score = ssim(gray_before, gray_after)
        except ImportError:
            ssim_score = 1.0 - normalized

        return {
            "changed": normalized > 0.05,
            "change_score": round(normalized, 4),
            "ssim": round(float(ssim_score), 4),
            "mse": round(mse, 2),
        }

    def _is_overlapping(self, matches: list, x1: int, y1: int,
                        x2: int, y2: int, iou_thresh: float = 0.5) -> bool:
        """检查是否与已有匹配重叠"""
        for m in matches:
            mx1, my1, mx2, my2 = m["bbox"]
            ix1 = max(x1, mx1)
            iy1 = max(y1, my1)
            ix2 = min(x2, mx2)
            iy2 = min(y2, my2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (x2 - x1) * (y2 - y1) + (mx2 - mx1) * (my2 - my1) - inter
                if inter / max(union, 1) > iou_thresh:
                    return True
        return False

    def _nms(self, matches: list, iou_threshold: float = 0.5) -> list:
        """非极大值抑制"""
        if not matches:
            return []
        sorted_matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)
        keep = []
        for m in sorted_matches:
            if not self._is_overlapping(keep, *m["bbox"], iou_threshold):
                keep.append(m)
        return keep
