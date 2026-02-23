import json
import os
import re
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from config import (
    CLIP_CONSISTENCY_THRESHOLD,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_SCENES,
    MIN_SCENES,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT_SECONDS,
    OPENAI_API_KEY_ENV,
    OPENAI_TIMEOUT_SECONDS,
    PPO_BATCH_SIZE,
    PPO_INIT_KL_COEF,
    PPO_LEARNING_RATE,
    PPO_MINI_BATCH_SIZE,
    PPO_POLICY_MODEL_NAME,
    RL_EPISODES_PER_SCENE,
)


@dataclass
class CritiqueResult:
    consistency_score: float
    critique: str
    retry: bool
    suggested_prompt: str


class DirectorAgent:
    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        model_name: str = LLM_MODEL,
        device: Optional[str] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(PPO_POLICY_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            PPO_POLICY_MODEL_NAME
        )
        self.policy_model = self.policy_model.to(self.device)

        ppo_config = PPOConfig(
            learning_rate=PPO_LEARNING_RATE,
            batch_size=PPO_BATCH_SIZE,
            mini_batch_size=PPO_MINI_BATCH_SIZE,
            init_kl_coef=PPO_INIT_KL_COEF,
        )
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            tokenizer=self.tokenizer,
        )

        self.reward_curve: List[float] = []

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI

            api_key = os.getenv(OPENAI_API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"Missing {OPENAI_API_KEY_ENV} environment variable")
            client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0.4,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"OpenAI call failed: {exc}") from exc

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.4},
        }
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider.lower() == "openai":
            return self._call_openai(system_prompt, user_prompt)
        if self.provider.lower() == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, Any]:
        text = text.strip()
        code_block = re.search(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if code_block:
            return json.loads(code_block.group(1))

        raw_json = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if raw_json:
            return json.loads(raw_json.group(1))

        return json.loads(text)

    @staticmethod
    def _validate_creative_document(doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required_top = [
            "characters",
            "locations",
            "scenes",
            "cinematic_style",
            "narrative_arc",
            "color_grading",
        ]
        errors: List[str] = []
        for key in required_top:
            if key not in doc:
                errors.append(f"Missing top-level key: {key}")

        if "characters" in doc and isinstance(doc["characters"], list):
            for index, ch in enumerate(doc["characters"]):
                for key in [
                    "id",
                    "name",
                    "age",
                    "gender",
                    "wardrobe",
                    "visual_description",
                ]:
                    if key not in ch:
                        errors.append(f"Character {index} missing {key}")
        else:
            errors.append("characters must be a list")

        if "locations" in doc and isinstance(doc["locations"], list):
            for index, loc in enumerate(doc["locations"]):
                for key in [
                    "id",
                    "name",
                    "scenery_type",
                    "material_palette",
                    "visual_description",
                ]:
                    if key not in loc:
                        errors.append(f"Location {index} missing {key}")
        else:
            errors.append("locations must be a list")

        if "scenes" in doc and isinstance(doc["scenes"], list):
            for index, scene in enumerate(doc["scenes"]):
                for key in [
                    "scene_id",
                    "narrative_description",
                    "character_ids",
                    "location_id",
                    "camera_style",
                    "emotional_tone",
                    "time_of_day",
                    "weather",
                    "scene_prompt",
                    "continuity_constraints",
                ]:
                    if key not in scene:
                        errors.append(f"Scene {index} missing {key}")
        else:
            errors.append("scenes must be a list")

        if len(doc.get("scenes", [])) < MIN_SCENES:
            errors.append(
                f"Scene count too low: expected at least {MIN_SCENES}, got {len(doc.get('scenes', []))}"
            )
        if len(doc.get("scenes", [])) > MAX_SCENES:
            errors.append(
                f"Scene count too high: max {MAX_SCENES}, got {len(doc.get('scenes', []))}"
            )

        character_ids = {
            c.get("id") for c in doc.get("characters", []) if isinstance(c, dict)
        }
        location_ids = {
            l.get("id") for l in doc.get("locations", []) if isinstance(l, dict)
        }

        for index, scene in enumerate(doc.get("scenes", [])):
            for cid in scene.get("character_ids", []):
                if cid not in character_ids:
                    errors.append(
                        f"Scene {index} references unknown character_id: {cid}"
                    )
            if scene.get("location_id") not in location_ids:
                errors.append(
                    f"Scene {index} references unknown location_id: {scene.get('location_id')}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def _normalize_creative_document(doc: Dict[str, Any]) -> Dict[str, Any]:
        characters: List[Dict[str, Any]] = []
        for index, item in enumerate(doc.get("characters", [])):
            cid = item.get("id", f"char_{index + 1}")
            characters.append(
                {
                    "id": str(cid),
                    "name": item.get("name", f"Character {index + 1}"),
                    "age": item.get("age", "adult"),
                    "gender": item.get("gender", "unspecified"),
                    "wardrobe": item.get("wardrobe", "consistent outfit"),
                    "visual_description": item.get("visual_description", ""),
                    "reference_image_paths": item.get("reference_image_paths", []),
                }
            )

        locations: List[Dict[str, Any]] = []
        for index, item in enumerate(doc.get("locations", [])):
            lid = item.get("id", f"loc_{index + 1}")
            locations.append(
                {
                    "id": str(lid),
                    "name": item.get("name", f"Location {index + 1}"),
                    "scenery_type": item.get("scenery_type", "environment"),
                    "material_palette": item.get("material_palette", "neutral tones"),
                    "visual_description": item.get("visual_description", ""),
                    "reference_image_paths": item.get("reference_image_paths", []),
                }
            )

        scenes: List[Dict[str, Any]] = []
        for index, item in enumerate(doc.get("scenes", [])):
            sid = item.get("scene_id", f"scene_{index + 1}")
            scenes.append(
                {
                    "scene_id": str(sid),
                    "narrative_description": item.get(
                        "narrative_description", "Cinematic progression scene"
                    ),
                    "scene_prompt": item.get(
                        "scene_prompt", item.get("narrative_description", "")
                    ),
                    "character_ids": [
                        str(cid) for cid in item.get("character_ids", [])
                    ],
                    "location_id": str(item.get("location_id", "")),
                    "camera_style": item.get("camera_style", "cinematic tracking shot"),
                    "emotional_tone": item.get("emotional_tone", "neutral"),
                    "time_of_day": item.get("time_of_day", "day"),
                    "weather": item.get("weather", "clear"),
                    "continuity_constraints": item.get(
                        "continuity_constraints",
                        "Keep character identity, outfit, and scenery palette unchanged.",
                    ),
                }
            )

        return {
            "characters": characters,
            "locations": locations,
            "scenes": scenes,
            "cinematic_style": doc.get("cinematic_style", "cinematic realism"),
            "narrative_arc": doc.get(
                "narrative_arc", "setup -> conflict -> resolution"
            ),
            "color_grading": doc.get("color_grading", "filmic teal-orange"),
        }

    def _self_critique_document(self, doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
        ok, local_errors = self._validate_creative_document(doc)
        if not ok:
            return False, local_errors

        system = "You are a strict screenplay continuity and visual consistency critic."
        user = (
            "Analyze this Creative Document JSON for missing visual details, contradictions, weak scene descriptions, "
            "and narrative inconsistencies. Return strictly JSON: "
            '{"valid": bool, "issues": [str]}\n\n'
            f"{json.dumps(doc, ensure_ascii=False)}"
        )
        try:
            response = self._call_llm(system, user)
            critique = self._extract_json_block(response)
            valid = bool(critique.get("valid", False))
            issues = critique.get("issues", [])
            if not isinstance(issues, list):
                issues = ["Critic returned non-list issues field"]
            return valid, [str(item) for item in issues]
        except Exception:
            return False, ["LLM critique failed", traceback.format_exc()]

    def generate_creative_document(
        self, seed_prompt: str, memory_state: str
    ) -> Dict[str, Any]:
        system = (
            "You are an expert cinematic director AI that must output strict JSON only. "
            "No markdown, no explanations."
        )
        user_template = (
            "Create a high-detail Creative Document JSON for autonomous cinematic video generation on Wan2.2.\n"
            "Break story into multiple contiguous scenes so each scene is roughly 10-12 seconds.\n"
            "Extract and lock characters and sceneries for whole film to keep consistency.\n"
            "Use this schema exactly:\n"
            "{\n"
            '  "characters": [{"id":"...","name":"...","age":"...","gender":"...","wardrobe":"...","visual_description":"..."}],\n'
            '  "locations": [{"id":"...","name":"...","scenery_type":"...","material_palette":"...","visual_description":"..."}],\n'
            '  "scenes": [{"scene_id":"...","narrative_description":"...","scene_prompt":"...","character_ids":["..."],"location_id":"...","camera_style":"...","emotional_tone":"...","time_of_day":"...","weather":"...","continuity_constraints":"..."}],\n'
            '  "cinematic_style": "...",\n'
            '  "narrative_arc": "setup/conflict/resolution summary",\n'
            '  "color_grading": "..."\n'
            "}\n"
            "Constraints: richly detailed character age, gender, and wardrobe; rich scenery details for roads/rivers/buildings/schools where relevant; coherent arc; no contradictions.\n"
            "Seed prompt:\n"
            f"{seed_prompt}\n\n"
            "Current continuity state:\n"
            f"{memory_state}\n"
        )

        issues: List[str] = []
        candidate_doc: Dict[str, Any] = {}
        for attempt in range(max(2, RL_EPISODES_PER_SCENE)):
            augmentation = ""
            if issues:
                augmentation = (
                    "Fix these issues from previous attempt:\n- " + "\n- ".join(issues)
                )
            response = self._call_llm(system, user_template + "\n" + augmentation)
            candidate_doc = self._normalize_creative_document(
                self._extract_json_block(response)
            )
            valid, issues = self._self_critique_document(candidate_doc)
            if valid:
                return candidate_doc

        raise RuntimeError(
            f"Failed to generate valid Creative Document after retries. Last issues: {issues}"
        )

    def update_policy(
        self, scene_state_text: str, action_prompt_text: str, reward: float
    ) -> Dict[str, Any]:
        self.reward_curve.append(float(reward))
        query_tensor = self.tokenizer.encode(scene_state_text, return_tensors="pt").to(
            self.device
        )[0]
        response_tensor = self.tokenizer.encode(
            action_prompt_text, return_tensors="pt"
        ).to(self.device)[0]
        reward_tensor = torch.tensor(float(reward), dtype=torch.float32).to(self.device)

        try:
            stats = self.ppo_trainer.step(
                [query_tensor], [response_tensor], [reward_tensor]
            )
            return {
                "status": "updated",
                "stats": {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in stats.items()
                },
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}


class PromptEnricher:
    def __init__(self, director: Optional[DirectorAgent] = None):
        self.director = director

    def enrich(
        self,
        scene: Dict[str, Any],
        creative_document: Dict[str, Any],
        memory_state: str,
    ) -> str:
        characters = {
            item["id"]: item
            for item in creative_document.get("characters", [])
            if "id" in item
        }
        locations = {
            item["id"]: item
            for item in creative_document.get("locations", [])
            if "id" in item
        }

        char_blocks = []
        for cid in scene.get("character_ids", []):
            char = characters.get(cid)
            if char:
                char_blocks.append(
                    f"{char.get('name', cid)} | age={char.get('age', 'adult')} | gender={char.get('gender', 'unspecified')} | wardrobe={char.get('wardrobe', 'consistent outfit')} | visual={char.get('visual_description', '')}"
                )

        loc = locations.get(scene.get("location_id"), {})
        location_text = loc.get("visual_description", "")

        base_prompt = (
            f"Scene ID: {scene.get('scene_id')}\n"
            f"Narrative: {scene.get('narrative_description')}\n"
            f"Scene prompt: {scene.get('scene_prompt', scene.get('narrative_description', ''))}\n"
            f"Continuity constraints: {scene.get('continuity_constraints', '')}\n"
            f"Camera style: {scene.get('camera_style')}\n"
            f"Emotional tone: {scene.get('emotional_tone')}\n"
            f"Time of day: {scene.get('time_of_day')}\n"
            f"Weather: {scene.get('weather')}\n"
            f"Characters: {' | '.join(char_blocks)}\n"
            f"Location: {loc.get('name', scene.get('location_id', 'unknown'))} | {location_text}\n"
            f"Cinematic style: {creative_document.get('cinematic_style', '')}\n"
            f"Color grading: {creative_document.get('color_grading', '')}\n"
            f"Continuity memory: {memory_state}\n"
            "Optimize for Wan2.2 T2V. Emphasize coherent composition, consistent character age/gender/wardrobe, stable scenery, and cinematic motion."
        )

        if self.director is None:
            return base_prompt

        system = "You are a prompt engineer for Wan2.2 text-to-video. Return plain prompt text only."
        user = (
            "Refine this scene spec into a high-fidelity generation prompt with concise, dense details:\n\n"
            + base_prompt
        )
        try:
            refined = self.director._call_llm(system, user).strip()
            return refined if refined else base_prompt
        except Exception:
            return base_prompt


class SelfCritic:
    def __init__(
        self,
        threshold: float = CLIP_CONSISTENCY_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _sample_frames(self, clip_path: str, count: int = 5) -> List[Image.Image]:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video clip: {clip_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"No frames in clip: {clip_path}")

        indices = np.linspace(0, total - 1, count, dtype=int)
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()

        if not frames:
            raise RuntimeError(f"Failed to sample frames from {clip_path}")
        return frames

    def _embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(image) for image in images]).to(
            self.device
        )
        with torch.no_grad():
            emb = self.model.encode_image(tensors)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def evaluate(
        self,
        clip_path: str,
        character_reference_images: List[str],
        location_reference_images: List[str],
        current_prompt: str,
    ) -> CritiqueResult:
        try:
            frames = self._sample_frames(clip_path, count=5)
            frame_emb = self._embed_images(frames)

            ref_images: List[Image.Image] = []
            for path in character_reference_images + location_reference_images:
                if os.path.exists(path):
                    ref_images.append(Image.open(path).convert("RGB"))

            if not ref_images:
                return CritiqueResult(
                    consistency_score=0.5,
                    critique="No reference images available; skipped strict CLIP consistency check.",
                    retry=False,
                    suggested_prompt=current_prompt,
                )

            ref_emb = self._embed_images(ref_images)
            similarity = frame_emb @ ref_emb.T
            score = float(similarity.mean().item())

            retry = score < self.threshold
            critique = f"Consistency score={score:.4f}. " + (
                "Below threshold; reinforce character facial traits, wardrobe, and location materials/lighting."
                if retry
                else "Consistency acceptable."
            )
            suggested = current_prompt
            if retry:
                suggested = (
                    current_prompt
                    + " Maintain strict identity lock: same facial structure, skin tone, eye color, hairstyle, clothing textures, and location palette as references."
                )
            return CritiqueResult(
                consistency_score=score,
                critique=critique,
                retry=retry,
                suggested_prompt=suggested,
            )
        except Exception as exc:
            return CritiqueResult(
                consistency_score=0.0,
                critique=f"SelfCritic failed: {exc}",
                retry=False,
                suggested_prompt=current_prompt,
            )

    def close(self) -> None:
        try:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
