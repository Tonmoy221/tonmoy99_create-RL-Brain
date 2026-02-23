import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import MEMORY_DIR, MEMORY_STATE_FILE


@dataclass
class ConsistencyMemory:
    output_root: str = "."
    character_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    location_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    continuity_log: List[str] = field(default_factory=list)
    last_frame_path: Optional[str] = None
    rl_episode_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.memory_dir = os.path.join(self.output_root, MEMORY_DIR)
        self.state_path = os.path.join(self.memory_dir, MEMORY_STATE_FILE)
        self.embeddings_dir = os.path.join(self.memory_dir, "embeddings")
        self.frames_dir = os.path.join(self.memory_dir, "frames")
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

    def _embedding_path(self, registry_type: str, key: str) -> str:
        safe_key = str(key).replace("/", "_")
        return os.path.join(self.embeddings_dir, f"{registry_type}_{safe_key}.npy")

    def _serialize_registry(
        self, registry: Dict[str, Dict[str, Any]], registry_type: str
    ) -> Dict[str, Dict[str, Any]]:
        serialized: Dict[str, Dict[str, Any]] = {}
        for key, payload in registry.items():
            payload_copy = dict(payload)
            embedding = payload_copy.get("clip_embedding")
            if embedding is not None:
                path = self._embedding_path(registry_type, key)
                np.save(path, np.asarray(embedding, dtype=np.float32))
                payload_copy["clip_embedding"] = path
            serialized[key] = payload_copy
        return serialized

    def _deserialize_registry(
        self, payload: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        restored: Dict[str, Dict[str, Any]] = {}
        for key, item in payload.items():
            data = dict(item)
            emb = data.get("clip_embedding")
            if isinstance(emb, str) and os.path.exists(emb):
                try:
                    data["clip_embedding"] = np.load(emb)
                except Exception:
                    data["clip_embedding"] = None
            restored[key] = data
        return restored

    def save(self) -> None:
        try:
            payload = {
                "character_registry": self._serialize_registry(
                    self.character_registry, "character"
                ),
                "location_registry": self._serialize_registry(
                    self.location_registry, "location"
                ),
                "continuity_log": self.continuity_log,
                "last_frame_path": self.last_frame_path,
                "rl_episode_log": self.rl_episode_log,
            }
            with open(self.state_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to save memory state: {exc}") from exc

    def load(self) -> bool:
        if not os.path.exists(self.state_path):
            return False
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.character_registry = self._deserialize_registry(
                payload.get("character_registry", {})
            )
            self.location_registry = self._deserialize_registry(
                payload.get("location_registry", {})
            )
            self.continuity_log = payload.get("continuity_log", [])
            self.last_frame_path = payload.get("last_frame_path")
            self.rl_episode_log = payload.get("rl_episode_log", [])
            return True
        except Exception as exc:
            raise RuntimeError(f"Failed to load memory state: {exc}") from exc

    def _extract_last_frame(self, clip_path: str, scene_id: str) -> Optional[str]:
        if not os.path.exists(clip_path):
            return None
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            return None
        out_path = os.path.join(self.frames_dir, f"last_frame_{scene_id}.png")
        ok_write = cv2.imwrite(out_path, frame_bgr)
        return out_path if ok_write else None

    def update_after_scene(
        self, scene_id: str, clip_path: str, prompt: str, reward: float
    ) -> None:
        try:
            extracted_last = self._extract_last_frame(clip_path, scene_id)
            if extracted_last is not None:
                self.last_frame_path = extracted_last

            continuity_entry = (
                f"Scene {scene_id} completed. Prompt summary: {prompt[:220]}. "
                f"Reward={float(reward):.4f}. Clip={clip_path}."
            )
            self.continuity_log.append(continuity_entry)

            retries = 0
            for event in reversed(self.rl_episode_log):
                if event.get("scene_id") == scene_id:
                    retries += 1
                else:
                    break

            self.rl_episode_log.append(
                {
                    "scene_id": scene_id,
                    "prompt_used": prompt,
                    "reward_received": float(reward),
                    "retry_count": int(retries),
                }
            )
            self.save()
        except Exception as exc:
            raise RuntimeError(
                f"Failed update_after_scene for scene {scene_id}: {exc}"
            ) from exc

    def get_state_for_agent(self) -> str:
        character_summary = []
        for key, value in self.character_registry.items():
            character_summary.append(
                f"{key}: refs={len(value.get('reference_image_paths', []))}, desc={value.get('text_description', '')[:120]}"
            )

        location_summary = []
        for key, value in self.location_registry.items():
            location_summary.append(
                f"{key}: refs={len(value.get('reference_image_paths', []))}, desc={value.get('text_description', '')[:120]}"
            )

        continuity_tail = self.continuity_log[-5:]
        reward_tail = [
            entry.get("reward_received", 0.0) for entry in self.rl_episode_log[-10:]
        ]

        state_text = (
            "ConsistencyMemory State\n"
            f"Characters: {character_summary if character_summary else 'None'}\n"
            f"Locations: {location_summary if location_summary else 'None'}\n"
            f"Last frame path: {self.last_frame_path}\n"
            f"Recent continuity: {continuity_tail if continuity_tail else 'None'}\n"
            f"Recent rewards: {reward_tail if reward_tail else 'None'}"
        )
        return state_text
