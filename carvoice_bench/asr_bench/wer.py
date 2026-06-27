"""词错误率 (WER) 及上下文词错误率 (WER-C) 计算"""

import re
from typing import Optional

import numpy as np


class WERCalculator:
    """
    词错误率计算器

    支持:
    - WER (Word Error Rate) — 标准词错误率
    - WER-C (Contextual Word Error Rate) — 上下文词错误率（针对车载领域词汇加权）
    - CER (Character Error Rate) — 字错误率（中文场景）
    """

    def __init__(self, context_vocab: Optional[dict[str, float]] = None):
        """
        Args:
            context_vocab: 上下文词汇表 {词: 权重}，车载领域关键词如 {"空调": 2.0, "导航": 1.5}
        """
        self.context_vocab = context_vocab or {
            # 默认车载领域高频词权重
            "空调": 2.0, "导航": 2.0, "音乐": 1.5, "电话": 1.5,
            "车窗": 2.0, "天窗": 2.0, "座椅": 1.5, "方向盘": 1.5,
            "加热": 1.5, "通风": 1.5, "按摩": 1.5, "模式": 1.0,
            "打开": 1.0, "关闭": 1.0, "调整": 1.0, "设置": 1.0,
        }

    def compute_wer(self, reference: str, hypothesis: str) -> dict:
        """
        计算 WER

        Returns: {"wer": float, "insertions": int, "deletions": int,
                  "substitutions": int, "reference_words": int, "word_details": [...]}
        """
        ref_words = self._tokenize(reference)
        hyp_words = self._tokenize(hypothesis)

        # 使用编辑距离计算
        n = len(ref_words)
        m = len(hyp_words)
        dp = np.zeros((n + 1, m + 1), dtype=int)
        dp[:, 0] = np.arange(n + 1)
        dp[0, :] = np.arange(m + 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # deletion
                    dp[i, j - 1] + 1,      # insertion
                    dp[i - 1, j - 1] + cost,  # substitution
                )

        # 回溯获取操作详情
        i, j = n, m
        insertions = deletions = substitutions = 0
        word_details = []

        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (0 if ref_words[i - 1] == hyp_words[j - 1] else 1):
                op = "correct" if ref_words[i - 1] == hyp_words[j - 1] else "substitution"
                if op == "substitution":
                    substitutions += 1
                word_details.append({
                    "ref": ref_words[i - 1],
                    "hyp": hyp_words[j - 1] if j > 0 else "",
                    "op": op,
                })
                i -= 1
                j -= 1
            elif j > 0 and (i == 0 or dp[i, j - 1] <= dp[i - 1, j]):
                word_details.append({"ref": "", "hyp": hyp_words[j - 1], "op": "insertion"})
                insertions += 1
                j -= 1
            elif i > 0:
                word_details.append({"ref": ref_words[i - 1], "hyp": "", "op": "deletion"})
                deletions += 1
                i -= 1

        total_errors = insertions + deletions + substitutions
        ref_word_count = len(ref_words)
        wer = total_errors / max(ref_word_count, 1)

        return {
            "wer": round(float(wer), 4),
            "insertions": insertions,
            "deletions": deletions,
            "substitutions": substitutions,
            "reference_words": ref_word_count,
            "hypothesis_words": len(hyp_words),
            "word_details": list(reversed(word_details)),
        }

    def compute_wer_c(self, reference: str, hypothesis: str) -> dict:
        """
        上下文词错误率 (WER-C) — 对车载领域关键词加权

        Returns: {"wer_c": float, "weighted_errors": float, "weighted_total": float}
        """
        ref_words = self._tokenize(reference)
        hyp_words = self._tokenize(hypothesis)

        # 计算带权重的编辑距离
        n = len(ref_words)
        m = len(hyp_words)
        dp = np.full((n + 1, m + 1), float("inf"))
        dp[0, 0] = 0
        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] + self._word_weight(ref_words[i - 1])
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] + self._word_weight(hyp_words[j - 1])

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else self._word_weight(ref_words[i - 1])
                dp[i, j] = min(
                    dp[i - 1, j] + self._word_weight(ref_words[i - 1]),  # deletion
                    dp[i, j - 1] + self._word_weight(hyp_words[j - 1]),  # insertion
                    dp[i - 1, j - 1] + cost,  # substitution
                )

        weighted_total = sum(self._word_weight(w) for w in ref_words)
        wer_c = dp[n, m] / max(weighted_total, 1)

        return {
            "wer_c": round(float(wer_c), 4),
            "weighted_errors": float(dp[n, m]),
            "weighted_total": float(weighted_total),
            "reference_words": n,
        }

    def compute_cer(self, reference: str, hypothesis: str) -> dict:
        """计算字错误率 (CER) — 按字级别"""
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))

        n = len(ref_chars)
        m = len(hyp_chars)
        dp = np.zeros((n + 1, m + 1), dtype=int)
        dp[:, 0] = np.arange(n + 1)
        dp[0, :] = np.arange(m + 1)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)

        total_errors = dp[n, m]
        cer = total_errors / max(n, 1)

        return {
            "cer": round(float(cer), 4),
            "total_errors": int(total_errors),
            "reference_chars": n,
        }

    def _tokenize(self, text: str) -> list[str]:
        """分词（中文按字/词拆分，英文按空格拆分）"""
        try:
            import jieba
            words = list(jieba.cut(text.strip()))
            return [w.strip() for w in words if w.strip()]
        except ImportError:
            # 回退：按字符切分
            return [c for c in text.strip() if c.strip()]

    def _word_weight(self, word: str) -> float:
        """获取词的上下文字典权重"""
        return self.context_vocab.get(word, 1.0)

    @staticmethod
    def normalize_text(text: str) -> str:
        """文本归一化：去除标点、全角转半角、统一空格"""
        text = text.strip()
        # 全角转半角
        result = []
        for c in text:
            code = ord(c)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result.append(" ")
            else:
                result.append(c)
        text = "".join(result)
        # 去除非必要标点
        text = re.sub(r"""[，。！？、；：""''（）【】《》.,!?;:"'()[\]<>]""", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
