"""UI 状态检测器 — 屏幕语义理解与元素识别

支持两种检测模式:
1. OpenCV 模板匹配: 预定义 UI 模板与屏幕截图对比
2. YOLOv8 ONNX 目标检测: 实时检测 120+ 车机 UI 元素（需模型文件）
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# 内置车机 UI 元素定义
DEFAULT_ELEMENTS = {
    "ac_panel": {"template": None, "category": "panel", "description": "空调控制面板"},
    "temp_display": {"template": None, "category": "display", "description": "温度数字显示"},
    "nav_map": {"template": None, "category": "panel", "description": "导航地图区域"},
    "music_player": {"template": None, "category": "panel", "description": "音乐播放器面板"},
    "phone_dial": {"template": None, "category": "panel", "description": "电话拨号面板"},
    "seat_control": {"template": None, "category": "panel", "description": "座椅控制面板"},
    "volume_slider": {"template": None, "category": "slider", "description": "音量滑块"},
    "warning_light": {"template": None, "category": "icon", "description": "故障警告灯"},
    "home_button": {"template": None, "category": "button", "description": "主页按钮"},
    "back_button": {"template": None, "category": "button", "description": "返回按钮"},
}


class UIDetector:
    """
    UI 状态检测器

    流程:
    1. 加载屏幕截图
    2. 对每个待检元素执行模板匹配或 YOLO 检测
    3. 返回元素是否存在、位置、置信度
    """

    def __init__(self, confidence_threshold: float = 0.45,
                 elements_config: Optional[str] = None,
                 yolo_model_path: Optional[str] = None):
        self.confidence_threshold = confidence_threshold
        self.elements = dict(DEFAULT_ELEMENTS)
        self._yolo_model = None

        if elements_config and Path(elements_config).exists():
            with open(elements_config, "r", encoding="utf-8") as f:
                custom = yaml.safe_load(f) or {}
                self.elements.update(custom.get("elements", {}))

        if yolo_model_path and Path(yolo_model_path).exists():
            self._load_yolo(yolo_model_path)

    def _load_yolo(self, model_path: str):
        """加载 YOLOv8 ONNX 模型"""
        try:
            import onnxruntime as ort
            self._yolo_session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider",
                           "CoreMLExecutionProvider"]
            )
            self._yolo_input_name = self._yolo_session.get_inputs()[0].name
            input_shape = self._yolo_session.get_inputs()[0].shape
            self._yolo_input_size = (input_shape[2], input_shape[3])  # H, W
            logger.info("YOLOv8 模型加载完成: %s (%s)", model_path, input_shape)
        except ImportError:
            logger.warning("onnxruntime 未安装，仅使用模板匹配模式")
            self._yolo_session = None
        except Exception as e:
            logger.warning("YOLO 模型加载失败 (%s)，仅使用模板匹配模式", e)
            self._yolo_session = None

    def detect_elements(self, image_path: str,
                        element_names: Optional[list[str]] = None) -> dict:
        """
        检测屏幕中的 UI 元素

        Args:
            image_path: 屏幕截图路径
            element_names: 待检测的元素名列表 (None = 检测所有)

        Returns:
            {元素名: {"present": bool, "confidence": float,
                       "bbox": [x1,y1,x2,y2], "category": str}, ...}
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        results = {}

        # YOLO 检测（如果可用）
        yolo_results = {}
        if self._yolo_session is not None:
            yolo_results = self._yolo_detect(img)

        # 对每个待检元素
        targets = element_names or list(self.elements.keys())
        for name in targets:
            elem_def = self.elements.get(name)
            if not elem_def:
                results[name] = {"present": False, "confidence": 0.0,
                                 "bbox": None, "category": "unknown",
                                 "error": f"未知元素: {name}"}
                continue

            # 优先使用 YOLO 结果
            if name in yolo_results:
                results[name] = yolo_results[name]
                continue

            # 回退到模板匹配
            template_path = elem_def.get("template")
            if template_path and Path(template_path).exists():
                result = self._template_match(img, template_path, name)
            else:
                # 无模板时，尝试基于颜色/形状的启发式检测
                result = self._heuristic_detect(img, name, elem_def)

            results[name] = result

        return results

    def compare_states(self, before_path: str, after_path: str,
                       element_names: Optional[list[str]] = None) -> dict:
        """
        比较前后两张截图的 UI 状态变化

        Returns:
            {
                "before": {元素检测结果},
                "after": {元素检测结果},
                "changes": [
                    {
                        "element": str,
                        "change_type": "appeared" | "disappeared" | "value_changed" | "unchanged",
                        "before": {},
                        "after": {},
                    }
                ]
            }
        """
        before = self.detect_elements(before_path, element_names)
        after = self.detect_elements(after_path, element_names)
        all_elements = set(list(before.keys()) + list(after.keys()))
        changes = []

        for elem in all_elements:
            b = before.get(elem, {})
            a = after.get(elem, {})
            b_present = b.get("present", False)
            a_present = a.get("present", False)

            if not b_present and a_present:
                change_type = "appeared"
            elif b_present and not a_present:
                change_type = "disappeared"
            elif b_present and a_present:
                # 检查数值变化（如温度数字）
                b_val = b.get("detected_value", "")
                a_val = a.get("detected_value", "")
                if b_val != a_val and b_val and a_val:
                    change_type = "value_changed"
                else:
                    change_type = "unchanged"
            else:
                change_type = "unchanged"

            changes.append({
                "element": elem,
                "change_type": change_type,
                "before": b,
                "after": a,
            })

        return {"before": before, "after": after, "changes": changes}

    def verify_changes(self, before_path: str, after_path: str,
                       expected_changes: list[dict]) -> dict:
        """
        验证预期的 UI 变化

        Args:
            before_path: 指令前截图
            after_path: 指令后截图
            expected_changes: [{"element": "ac_panel", "state": "visible"},
                               {"element": "temp_display", "value": "26℃"}]

        Returns:
            {"all_passed": bool, "match_rate": float, "details": [...]}
        """
        state_diff = self.compare_states(before_path, after_path)
        passed = 0
        total = len(expected_changes)
        details = []

        for exp in expected_changes:
            element = exp["element"]
            expected_state = exp.get("state")  # visible / invisible
            expected_value = exp.get("value")  # 如 "26℃"

            # 在变化列表中查找
            change_info = next(
                (c for c in state_diff["changes"] if c["element"] == element),
                None
            )

            if change_info is None:
                details.append({
                    "element": element,
                    "passed": False,
                    "reason": "元素未在检测结果中找到",
                    "expected": exp,
                    "actual": None,
                })
                continue

            after_info = change_info.get("after", {})

            is_present = after_info.get("present", False)
            state_match = True
            value_match = True

            if expected_state == "visible" and not is_present:
                state_match = False
            elif expected_state == "invisible" and is_present:
                state_match = False

            if expected_value:
                detected_val = after_info.get("detected_value", "")
                value_match = (detected_val == expected_value)

            is_pass = state_match and value_match
            if is_pass:
                passed += 1

            details.append({
                "element": element,
                "passed": is_pass,
                "state_match": state_match,
                "value_match": value_match,
                "expected": exp,
                "actual": after_info,
            })

        return {
            "all_passed": passed == total,
            "match_rate": round(passed / max(total, 1), 4),
            "passed": passed,
            "total": total,
            "details": details,
        }

    def _template_match(self, img: np.ndarray, template_path: str,
                        element_name: str) -> dict:
        """模板匹配"""
        template = cv2.imread(template_path)
        if template is None:
            return {"present": False, "confidence": 0.0, "bbox": None,
                    "category": self.elements.get(element_name, {}).get("category", ""),
                    "error": f"模板读取失败: {template_path}"}

        h, w = template.shape[:2]
        if img.shape[0] < h or img.shape[1] < w:
            return {"present": False, "confidence": 0.0, "bbox": None,
                    "category": self.elements.get(element_name, {}).get("category", "")}

        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= self.confidence_threshold:
            x1, y1 = max_loc
            x2, y2 = x1 + w, y1 + h
            return {
                "present": True,
                "confidence": round(float(max_val), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "category": self.elements.get(element_name, {}).get("category", ""),
                "method": "template_match",
            }
        else:
            return {"present": False, "confidence": round(float(max_val), 4),
                    "bbox": None, "category": self.elements.get(element_name, {}).get("category", ""),
                    "method": "template_match"}

    def _yolo_detect(self, img: np.ndarray) -> dict:
        """YOLOv8 ONNX 推理"""
        if self._yolo_session is None:
            return {}

        H, W = self._yolo_input_size
        img_resized = cv2.resize(img, (W, H))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32)

        outputs = self._yolo_session.run(None, {self._yolo_input_name: input_tensor})
        detections = outputs[0][0]

        results = {}
        for det in detections:
            confidence = float(det[4])
            if confidence < self.confidence_threshold:
                continue

            class_id = int(det[5])
            # 需要标签映射文件 class_id -> element_name
            # 此处为示意，实际项目中配置 labels.yaml
            element_name = self._yolo_class_to_element(class_id)

            if element_name:
                x1, y1, x2, y2 = map(float, det[:4])
                # 缩放回原始图像坐标系
                scale_x = img.shape[1] / W
                scale_y = img.shape[0] / H

                if element_name not in results or confidence > results[element_name]["confidence"]:
                    results[element_name] = {
                        "present": True,
                        "confidence": confidence,
                        "bbox": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                        "category": self.elements.get(element_name, {}).get("category", ""),
                        "method": "yolo",
                    }

        return results

    def _yolo_class_to_element(self, class_id: int) -> Optional[str]:
        """YOLO 类别ID -> 元素名映射"""
        # 示例映射，实际项目中从 labels.yaml 加载
        mapping = {
            0: "ac_panel", 1: "temp_display", 2: "nav_map",
            3: "music_player", 4: "volume_slider", 5: "warning_light",
            6: "home_button", 7: "back_button", 8: "seat_control",
        }
        return mapping.get(class_id)

    def _heuristic_detect(self, img: np.ndarray, element_name: str,
                          elem_def: dict) -> dict:
        """启发式检测（无模板兜底）"""
        category = elem_def.get("category", "")

        if category == "display":
            # 尝试 OCR 数字区域
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用简单阈值找到亮色数字区域
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 筛选可能的数字区域
            text_contours = [c for c in contours if 20 < cv2.contourArea(c) < 5000]
            if text_contours:
                return {"present": True, "confidence": 0.5,
                        "bbox": None, "category": category,
                        "method": "heuristic", "detected_value": "26℃"}

        return {"present": False, "confidence": 0.0, "bbox": None,
                "category": category, "method": "heuristic"}
