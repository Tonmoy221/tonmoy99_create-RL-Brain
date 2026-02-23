import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open_clip
import requests
import torch
from PIL import Image

from config import (
    FRAMES_TO_SAMPLE_FOR_REWARD,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT_SECONDS,
    OPENAI_API_KEY_ENV,
    OPENAI_TIMEOUT_SECONDS,
    REWARD_CHARACTER_CONSISTENCY,
    REWARD_LOCATION_CONSISTENCY,
    REWARD_NARRATIVE_COHERENCE,
    REWARD_VISUAL_QUALITY,
)


class RewardModel:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.latest_breakdown: Dict[str, float] = {
            "visual": 0.0,
            "character": 0.0,
            "location": 0.0,
            "narrative": 0.0,
            "final": 0.0,
        }

    @staticmethod
    def _normalize_clip_score(value: float) -> float:
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _normalize_image_reward(value: float) -> float:
        return float(np.clip((value + 2.0) / 4.0, 0.0, 1.0))

    @staticmethod
    def _extract_float(text: str, default: float = 0.5) -> float:
        match = re.search(r"([01](?:\.\d+)?)", text)
        if match:
            return float(np.clip(float(match.group(1)), 0.0, 1.0))
        return default

    def _sample_frames(
        self, clip_path: str, n_frames: int = FRAMES_TO_SAMPLE_FOR_REWARD
    ) -> List[Image.Image]:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open clip: {clip_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Clip has no frames: {clip_path}")

        indices = np.linspace(0, frame_count - 1, n_frames, dtype=int)
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()

        if not frames:
            raise RuntimeError(f"Failed to sample any frame from {clip_path}")
        return frames

    def _visual_quality_score(self, clip_path: str, prompt: str) -> float:
        try:
            import ImageReward as RM

            frames = self._sample_frames(clip_path)
            model = RM.load("ImageReward-v1.0")
            values: List[float] = []
            for frame in frames:
                score = model.score(prompt, frame)
                values.append(float(score))
            visual = self._normalize_image_reward(float(np.mean(values)))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return visual
        except Exception:
            return 0.5

    def _compute_clip_similarity(
        self, frames: List[Image.Image], refs: List[str]
    ) -> float:
        if not refs:
            return 0.5

        valid_refs: List[Image.Image] = []
        for path in refs:
            if os.path.exists(path):
                try:
                    valid_refs.append(Image.open(path).convert("RGB"))
                except Exception:
                    continue

        if not valid_refs:
            return 0.5

        model = None
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model = model.to(self.device)
            model.eval()

            frame_tensors = torch.stack([preprocess(image) for image in frames]).to(
                self.device
            )
            ref_tensors = torch.stack([preprocess(image) for image in valid_refs]).to(
                self.device
            )

            with torch.no_grad():
                frame_emb = model.encode_image(frame_tensors)
                ref_emb = model.encode_image(ref_tensors)
                frame_emb = frame_emb / frame_emb.norm(dim=-1, keepdim=True)
                ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
                similarity = frame_emb @ ref_emb.T
            return self._normalize_clip_score(float(similarity.mean().item()))
        except Exception:
            return 0.5
        finally:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _character_consistency_score(
        self, clip_path: str, character_refs: List[str]
    ) -> float:
        try:
            frames = self._sample_frames(clip_path)
            return self._compute_clip_similarity(frames, character_refs)
        except Exception:
            return 0.5

    def _location_consistency_score(
        self, clip_path: str, location_refs: List[str]
    ) -> float:
        try:
            frames = self._sample_frames(clip_path)
            return self._compute_clip_similarity(frames, location_refs)
        except Exception:
            return 0.5

    def _openai_score(self, continuity_log: List[str], prompt: str) -> float:
        try:
            from openai import OpenAI

            api_key = os.getenv(OPENAI_API_KEY_ENV)
            if not api_key:
                return 0.5
            client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)
            user_text = (
                'Return only a JSON object like {"score": 0.0-1.0, "reason": "..."}. '
                "Score narrative coherence for this scene against continuity log.\n\n"
                f"continuity_log={json.dumps(continuity_log, ensure_ascii=False)}\n"
                f"scene_prompt={prompt}\n"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict narrative continuity evaluator.",
                    },
                    {"role": "user", "content": user_text},
                ],
            )
            content = response.choices[0].message.content or ""
            try:
                parsed = json.loads(content)
                return float(np.clip(float(parsed.get("score", 0.5)), 0.0, 1.0))
            except Exception:
                return self._extract_float(content, default=0.5)
        except Exception:
            return 0.5

    def _ollama_score(self, continuity_log: List[str], prompt: str) -> float:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict narrative continuity evaluator.",
                },
                {
                    "role": "user",
                    "content": (
                        'Return only JSON: {"score": 0.0-1.0, "reason": "..."}. '
                        f"continuity_log={json.dumps(continuity_log, ensure_ascii=False)}\n"
                        f"scene_prompt={prompt}"
                    ),
                },
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            try:
                parsed = json.loads(content)
                return float(np.clip(float(parsed.get("score", 0.5)), 0.0, 1.0))
            except Exception:
                return self._extract_float(content, default=0.5)
        except Exception:
            return 0.5

    def _narrative_coherence_score(
        self, continuity_log: List[str], prompt: str
    ) -> float:
        if LLM_PROVIDER.lower() == "openai":
            return self._openai_score(continuity_log, prompt)
        if LLM_PROVIDER.lower() == "ollama":
            return self._ollama_score(continuity_log, prompt)
        return 0.5

    def compute(
        self,
        scene_id: str,
        clip_path: str,
        prompt: str,
        character_refs: List[str],
        location_refs: List[str],
        continuity_log: List[str],
    ) -> float:
        visual = self._visual_quality_score(clip_path, prompt)
        character = self._character_consistency_score(clip_path, character_refs)
        location = self._location_consistency_score(clip_path, location_refs)
        narrative = self._narrative_coherence_score(continuity_log, prompt)

        final_reward = (
            REWARD_VISUAL_QUALITY * visual
            + REWARD_CHARACTER_CONSISTENCY * character
            + REWARD_LOCATION_CONSISTENCY * location
            + REWARD_NARRATIVE_COHERENCE * narrative
        )

        self.latest_breakdown = {
            "scene_id": scene_id,
            "visual": float(visual),
            "character": float(character),
            "location": float(location),
            "narrative": float(narrative),
            "final": float(final_reward),
        }
        return float(final_reward)
