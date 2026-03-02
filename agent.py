# RL AGENT BRAIN — Fully Local (RunPod GPU, no external API calls)
# All LLM inference runs on a local HuggingFace model via transformers pipeline.
# PPO policy update runs on the same GPU via TRL.
# CLIP-based SelfCritic runs on the same GPU via open_clip.

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import AutoTokenizer, pipeline as hf_pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from config import (
    CLIP_CONSISTENCY_THRESHOLD,
    MAX_SCENES,
    MIN_SCENES,
    OUTPUT_DIR,
    PPO_BATCH_SIZE,
    PPO_INIT_KL_COEF,
    PPO_LEARNING_RATE,
    PPO_MINI_BATCH_SIZE,
    PPO_POLICY_MODEL_NAME,
    RL_EPISODES_PER_SCENE,
    STORY_BIBLE_DIR,
)


# ──────────────────────────────────────────────────────────────────────────────
# Memory — mirrors LLMRunMemory but extended with RL reward tracking
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RLRunMemory:
    """
    Persistent continuity store for the RL pipeline.
    Mirrors LLMRunMemory exactly, adding reward_curve and clip_score for RL tracking.
    """
    state_path: str
    character_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    location_registry:  Dict[str, Dict[str, Any]] = field(default_factory=dict)
    continuity_log:     List[str]                  = field(default_factory=list)
    episode_log:        List[Dict[str, Any]]        = field(default_factory=list)
    reward_curve:       List[float]                 = field(default_factory=list)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        payload = {
            "character_registry": self.character_registry,
            "location_registry":  self.location_registry,
            "continuity_log":     self.continuity_log,
            "episode_log":        self.episode_log,
            "reward_curve":       self.reward_curve,
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.character_registry = payload.get("character_registry", {})
        self.location_registry  = payload.get("location_registry",  {})
        self.continuity_log     = payload.get("continuity_log",     [])
        self.episode_log        = payload.get("episode_log",        [])
        self.reward_curve       = payload.get("reward_curve",       [])
        return True

    def get_state_for_agent(self) -> str:
        """Build a formatted continuity string injected into every local LLM call."""
        character_summary = []
        for key, value in self.character_registry.items():
            appearance = (
                f"skin={value.get('skin_tone', '')} | "
                f"body={value.get('body_type', '')} | "
                f"height={value.get('height', '')} | "
                f"hair={value.get('hair_color', '')} {value.get('hair_style', '')} | "
                f"eyes={value.get('eye_color', '')} | "
                f"expression={value.get('facial_expression', '')} | "
                f"wardrobe={value.get('wardrobe', '')} | "
                f"features={value.get('distinguishing_features', '')}"
            )
            character_summary.append(f"{key}: {value.get('name', key)} | {appearance}")
        location_summary = [
            f"{key}: {value.get('name', key)} | palette={value.get('material_palette', '')}"
            for key, value in self.location_registry.items()
        ]
        return (
            "RLRunMemory State\n"
            f"Characters: {character_summary if character_summary else 'None'}\n"
            f"Locations: {location_summary if location_summary else 'None'}\n"
            f"Recent continuity: {self.continuity_log[-5:] if self.continuity_log else 'None'}\n"
            f"Reward curve (last 5): {self.reward_curve[-5:] if self.reward_curve else 'None'}"
        )

    def update_after_scene(
        self,
        scene_id:        str,
        prompt:          str,
        critique_score:  float,
        critique_issues: List[str],
        clip_score:      float,
        ppo_reward:      float,
    ) -> None:
        self.continuity_log.append(
            f"Scene {scene_id} finalized. "
            f"llm_score={critique_score:.4f} clip_score={clip_score:.4f} "
            f"ppo_reward={ppo_reward:.4f}. prompt={prompt[:220]}"
        )
        self.episode_log.append({
            "scene_id":        scene_id,
            "prompt_used":     prompt,
            "critique_score":  float(critique_score),
            "clip_score":      float(clip_score),
            "ppo_reward":      float(ppo_reward),
            "critique_issues": [str(i) for i in critique_issues],
            "timestamp":       int(time.time()),
        })
        self.reward_curve.append(float(ppo_reward))
        self.save()


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass returned by SelfCritic (CLIP-based)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CritiqueResult:
    consistency_score: float
    critique:          str
    retry:             bool
    suggested_prompt:  str


# ──────────────────────────────────────────────────────────────────────────────
# Local LLM inference helper
# ──────────────────────────────────────────────────────────────────────────────

class LocalLLM:
    """
    Wraps a HuggingFace text-generation pipeline loaded once on GPU.
    Replaces all OpenAI / Ollama calls. No network traffic.

    Supports any causal-LM on HuggingFace Hub or a local path:
      - meta-llama/Meta-Llama-3-8B-Instruct
      - mistralai/Mistral-7B-Instruct-v0.3
      - Qwen/Qwen2.5-7B-Instruct
      - /workspace/models/my-finetuned-model
    """

    def __init__(
        self,
        model_name_or_path: str = PPO_POLICY_MODEL_NAME,
        device: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.4,
        do_sample: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens     = max_new_tokens
        self.temperature        = temperature
        self.do_sample          = do_sample
        self.device             = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[LocalLLM] Loading '{model_name_or_path}' on {self.device} …")
        self._pipe = hf_pipeline(
            "text-generation",
            model=model_name_or_path,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        print("[LocalLLM] Model loaded.")

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        """
        Formats a system + user turn using the tokenizer's chat template if available,
        otherwise falls back to a plain concatenated prompt.
        Returns the model's reply as a plain string.
        """
        tokenizer = self._pipe.tokenizer

        # ── Try chat-template format (Llama-3, Mistral, Qwen, etc.) ──────────
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_prompt},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: some templates don't accept system role
                messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            # ── Plain prompt for models without a chat template ───────────────
            prompt = (
                f"### System\n{system_prompt}\n\n"
                f"### User\n{user_prompt}\n\n"
                f"### Assistant\n"
            )

        outputs = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,   # return only the generated part
        )
        return outputs[0]["generated_text"].strip()


# ──────────────────────────────────────────────────────────────────────────────
# Director Agent — fully local, PPO-enabled
# ──────────────────────────────────────────────────────────────────────────────

class DirectorAgent:
    """
    Cinematic director brain.

    LLM backbone  : local HuggingFace model (no API calls)
    RL backbone   : PPO via TRL (policy = same model weight class)
    CLIP critique : SelfCritic (separate class)

    Capabilities (matches LLM pipeline feature-for-feature):
      - generate_creative_document  (schema-driven, iterative self-critique)
      - refine_scene_prompt         (numbered aliases + appearance locks)
      - critique_scene_prompt       (LLM JSON critic)
      - update_policy               (PPO step with reward signal)
    """

    def __init__(
        self,
        local_llm:  LocalLLM,
        device:     Optional[str] = None,
    ):
        self.llm    = local_llm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── PPO policy model (separate from the generation LLM) ───────────────
        print(f"[DirectorAgent] Loading PPO policy '{PPO_POLICY_MODEL_NAME}' …")
        self.tokenizer = AutoTokenizer.from_pretrained(PPO_POLICY_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            PPO_POLICY_MODEL_NAME
        ).to(self.device)

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
        print("[DirectorAgent] PPO policy loaded.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal LLM call — routes to local model only
    # ─────────────────────────────────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Single entry-point for all LLM calls. 100% local, no network."""
        return self.llm(system_prompt, user_prompt)

    # ─────────────────────────────────────────────────────────────────────────
    # JSON helpers
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_creative_document(doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required_top = [
            "characters", "locations", "scenes",
            "cinematic_style", "narrative_arc", "color_grading",
        ]
        errors: List[str] = []
        for key in required_top:
            if key not in doc:
                errors.append(f"Missing top-level key: {key}")

        # ── Character field validation ────────────────────────────────────────
        if isinstance(doc.get("characters"), list):
            for index, ch in enumerate(doc["characters"]):
                if not isinstance(ch, dict):
                    errors.append(f"Character {index} must be an object")
                    continue
                for key in [
                    "id", "name", "age", "gender",
                    "skin_tone", "body_type", "height",
                    "hair_color", "hair_style", "eye_color",
                    "facial_expression", "distinguishing_features",
                    "wardrobe", "visual_description",
                ]:
                    if key not in ch:
                        errors.append(f"Character {index} missing '{key}'")
        else:
            errors.append("characters must be a list")

        # ── Location field validation ─────────────────────────────────────────
        if isinstance(doc.get("locations"), list):
            for index, loc in enumerate(doc["locations"]):
                if not isinstance(loc, dict):
                    errors.append(f"Location {index} must be an object")
                    continue
                for key in ["id", "name", "scenery_type", "material_palette", "visual_description"]:
                    if key not in loc:
                        errors.append(f"Location {index} missing '{key}'")
        else:
            errors.append("locations must be a list")

        # ── Scene validation ──────────────────────────────────────────────────
        if not isinstance(doc.get("scenes"), list):
            errors.append("scenes must be a list")

        scenes = doc.get("scenes", [])
        if len(scenes) < MIN_SCENES:
            errors.append(f"Scene count too low: expected at least {MIN_SCENES}, got {len(scenes)}")
        if len(scenes) > MAX_SCENES:
            errors.append(f"Scene count too high: max {MAX_SCENES}, got {len(scenes)}")

        character_ids = {c.get("id") for c in doc.get("characters", []) if isinstance(c, dict)}
        location_ids  = {l.get("id") for l in doc.get("locations",  []) if isinstance(l, dict)}

        for index, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                errors.append(f"Scene {index} must be an object")
                continue
            for key in [
                "scene_id", "narrative_description", "scene_prompt",
                "character_ids", "location_id", "camera_style",
                "emotional_tone", "time_of_day", "weather", "continuity_constraints",
            ]:
                if key not in scene:
                    errors.append(f"Scene {index} missing '{key}'")
            for cid in scene.get("character_ids", []):
                if cid not in character_ids:
                    errors.append(f"Scene {index} references unknown character_id: {cid}")
            if scene.get("location_id") not in location_ids:
                errors.append(f"Scene {index} references unknown location_id: {scene.get('location_id')}")

        return len(errors) == 0, errors

    # ─────────────────────────────────────────────────────────────────────────
    # Normalization
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_creative_document(doc: Dict[str, Any]) -> Dict[str, Any]:
        characters: List[Dict[str, Any]] = []
        for index, item in enumerate(doc.get("characters", [])):
            cid = item.get("id", f"char_{index + 1}")
            characters.append({
                "id":     str(cid),
                "name":   item.get("name",   f"Character {index + 1}"),
                "age":    item.get("age",    "adult"),
                "gender": item.get("gender", "unspecified"),
                # ── Detailed physical appearance ──────────────────────────────
                "skin_tone":  item.get("skin_tone",  "medium"),
                "body_type":  item.get("body_type",  "average"),        # slim/athletic/average/heavyset/muscular
                "height":     item.get("height",     "average height"), # e.g. "5ft 10in, tall"
                "hair_color": item.get("hair_color", "brown"),
                "hair_style": item.get("hair_style", "medium length"),
                "eye_color":  item.get("eye_color",  "brown"),
                "facial_expression":      item.get("facial_expression",     "neutral"),  # happy/sad/angry/determined/scared
                "distinguishing_features": item.get("distinguishing_features", ""),      # scars, tattoos, beard, glasses
                # ── Wardrobe & full visual summary ────────────────────────────
                "wardrobe":           item.get("wardrobe",           "consistent outfit"),
                "visual_description": item.get("visual_description", ""),
                "reference_image_paths": item.get("reference_image_paths", []),
            })

        locations: List[Dict[str, Any]] = []
        for index, item in enumerate(doc.get("locations", [])):
            lid = item.get("id", f"loc_{index + 1}")
            locations.append({
                "id":               str(lid),
                "name":             item.get("name",             f"Location {index + 1}"),
                "scenery_type":     item.get("scenery_type",     "environment"),
                "material_palette": item.get("material_palette", "neutral tones"),
                "visual_description": item.get("visual_description", ""),
                "reference_image_paths": item.get("reference_image_paths", []),
            })

        scenes: List[Dict[str, Any]] = []
        for index, item in enumerate(doc.get("scenes", [])):
            sid = item.get("scene_id", f"scene_{index + 1}")
            scenes.append({
                "scene_id":              str(sid),
                "narrative_description": item.get("narrative_description", "Cinematic progression scene"),
                "scene_prompt":          item.get("scene_prompt", item.get("narrative_description", "")),
                "character_ids":         [str(c) for c in item.get("character_ids", [])],
                "location_id":           str(item.get("location_id", "")),
                "camera_style":          item.get("camera_style",   "cinematic tracking shot"),
                "emotional_tone":        item.get("emotional_tone", "neutral"),
                "time_of_day":           item.get("time_of_day",    "day"),
                "weather":               item.get("weather",        "clear"),
                "continuity_constraints": item.get(
                    "continuity_constraints",
                    "Keep character identity, outfit, and scenery palette unchanged.",
                ),
            })

        return {
            "characters":      characters,
            "locations":       locations,
            "scenes":          scenes,
            "cinematic_style": doc.get("cinematic_style", "cinematic realism"),
            "narrative_arc":   doc.get("narrative_arc",   "setup -> conflict -> resolution"),
            "color_grading":   doc.get("color_grading",   "filmic teal-orange"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Character alias helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_scene_character_assignments(
        scene:      Dict[str, Any],
        characters: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Assign a stable per-scene number to each character that appears in a scene.
        Returns e.g. {"char_1": "Character_1", "char_2": "Character_2"}.
        Prevents multi-character prompt clashes across scenes.
        """
        assignments: Dict[str, str] = {}
        for slot_idx, cid in enumerate(scene.get("character_ids", []), start=1):
            if cid in characters:
                assignments[cid] = f"Character_{slot_idx}"
        return assignments

    @staticmethod
    def _format_character_block(char: Dict[str, Any], label: str) -> str:
        """
        Verbose locked character description block for prompt injection.
        `label` is the scene-local numbered alias e.g. 'Character_1'.
        """
        lines = [
            f"[{label} | ID={char.get('id')} | Name={char.get('name', label)}]",
            f"  Gender      : {char.get('gender',              'unspecified')}",
            f"  Age         : {char.get('age',                 'adult')}",
            f"  Skin tone   : {char.get('skin_tone',           'medium')}",
            f"  Body type   : {char.get('body_type',           'average')}",
            f"  Height      : {char.get('height',              'average height')}",
            f"  Hair        : {char.get('hair_color', 'brown')} | {char.get('hair_style', 'medium length')}",
            f"  Eye color   : {char.get('eye_color',           'brown')}",
            f"  Expression  : {char.get('facial_expression',   'neutral')}",
            f"  Features    : {char.get('distinguishing_features', 'none')}",
            f"  Wardrobe    : {char.get('wardrobe',            'consistent outfit')}",
            f"  Full desc.  : {char.get('visual_description',  '')}",
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Self-critique (structural validation + local LLM semantic check)
    # ─────────────────────────────────────────────────────────────────────────

    def _self_critique_document(self, doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Two-stage critique:
        1. Structural validation via _validate_creative_document.
        2. Local LLM semantic critique for missing details / contradictions.
        """
        ok, local_errors = self._validate_creative_document(doc)
        if not ok:
            return False, local_errors

        system = "You are a strict screenplay continuity and visual consistency critic."
        user = (
            "Analyze this Creative Document JSON for missing visual details, contradictions, "
            "weak scene descriptions, and narrative inconsistencies. Return strictly JSON: "
            '{"valid": bool, "issues": [str]}\n\n'
            f"{json.dumps(doc, ensure_ascii=False)}"
        )
        try:
            response = self._call_llm(system, user)
            critique = self._extract_json_block(response)
            valid    = bool(critique.get("valid", False))
            issues   = critique.get("issues", [])
            if not isinstance(issues, list):
                issues = ["Critic returned non-list issues field"]
            return valid, [str(i) for i in issues]
        except Exception:
            return False, ["Local LLM critique failed", traceback.format_exc()]

    # ─────────────────────────────────────────────────────────────────────────
    # Generate Creative Document
    # ─────────────────────────────────────────────────────────────────────────

    def generate_creative_document(
        self, seed_prompt: str, memory_state: str
    ) -> Dict[str, Any]:
        system = (
            "You are an expert cinematic director AI that outputs strict JSON only. "
            "No markdown and no explanations."
        )
        user_template = (
            "Create a high-detail Creative Document JSON for cinematic planning.\n"
            "Use this schema exactly:\n"
            "{\n"
            '  "characters": [\n'
            "    {\n"
            '      "id": "char_1",\n'
            '      "name": "Full character name",\n'
            '      "age": "e.g. 28",\n'
            '      "gender": "male | female | non-binary",\n'
            '      "skin_tone": "e.g. fair, medium olive, dark brown, pale",\n'
            '      "body_type": "slim | athletic | average | heavyset | muscular",\n'
            '      "height": "e.g. 5ft 8in, tall, short",\n'
            '      "hair_color": "e.g. jet black, sandy blonde, dark red",\n'
            '      "hair_style": "e.g. short cropped, long wavy, curly shoulder-length",\n'
            '      "eye_color": "e.g. hazel, deep brown, pale blue",\n'
            '      "facial_expression": "e.g. determined, anxious, joyful, cold",\n'
            '      "distinguishing_features": "e.g. scar on left cheek, thick beard, round glasses",\n'
            '      "wardrobe": "detailed clothing description locked across all scenes",\n'
            '      "visual_description": "one complete sentence summarising full physical appearance"\n'
            "    }\n"
            "  ],\n"
            '  "locations": [{"id":"...","name":"...","scenery_type":"...","material_palette":"...","visual_description":"..."}],\n'
            '  "scenes": [\n'
            "    {\n"
            '      "scene_id": "scene_1",\n'
            '      "narrative_description": "...",\n'
            '      "scene_prompt": "...",\n'
            '      "character_ids": ["char_1"],\n'
            '      "location_id": "loc_1",\n'
            '      "camera_style": "...",\n'
            '      "emotional_tone": "...",\n'
            '      "time_of_day": "...",\n'
            '      "weather": "...",\n'
            '      "continuity_constraints": "..."\n'
            "    }\n"
            "  ],\n"
            '  "cinematic_style": "...",\n'
            '  "narrative_arc": "setup/conflict/resolution summary",\n'
            '  "color_grading": "..."\n'
            "}\n\n"
            "IMPORTANT RULES:\n"
            "- Every character MUST have a complete, specific physical description.\n"
            "- Do NOT use vague terms like 'attractive' or 'ordinary'; be concrete and visual.\n"
            "- Keep character appearance IDENTICAL across every scene they appear in.\n"
            f"Seed prompt:\n{seed_prompt}\n\n"
            f"Current continuity state:\n{memory_state}\n"
        )

        issues:        List[str]        = []
        candidate_doc: Dict[str, Any]   = {}

        for _ in range(max(2, RL_EPISODES_PER_SCENE)):
            augmentation = ""
            if issues:
                augmentation = "Fix these issues from previous attempt:\n- " + "\n- ".join(issues)
            response      = self._call_llm(system, user_template + "\n" + augmentation)
            candidate_doc = self._normalize_creative_document(self._extract_json_block(response))
            valid, issues = self._self_critique_document(candidate_doc)
            if valid:
                return candidate_doc

        raise RuntimeError(
            f"Failed to generate valid Creative Document after retries. Last issues: {issues}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Refine Scene Prompt
    # ─────────────────────────────────────────────────────────────────────────

    def refine_scene_prompt(
        self,
        scene:             Dict[str, Any],
        creative_document: Dict[str, Any],
        memory_state:      str,
    ) -> str:
        system = (
            "You are a prompt engineer for cinematic text-to-video systems. "
            "Return plain prompt text only."
        )

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

        scene_char_assignments = self._build_scene_character_assignments(scene, characters)

        char_blocks: List[str] = []
        for cid in scene.get("character_ids", []):
            char = characters.get(cid)
            if not char:
                continue
            label = scene_char_assignments.get(cid, cid)
            char_blocks.append(self._format_character_block(char, label))

        location = locations.get(scene.get("location_id"), {})

        char_alias_note = ""
        if scene_char_assignments:
            alias_list = ", ".join(
                f"{label} (= {characters[cid].get('name', cid)})"
                for cid, label in scene_char_assignments.items()
                if cid in characters
            )
            char_alias_note = (
                f"Character aliases for this scene: {alias_list}\n"
                "When writing the prompt, refer to each character by their alias "
                "(Character_1, Character_2, …) so that they remain uniquely identifiable "
                "and do not visually clash with each other.\n"
            )

        user = (
            "Refine this into one high-fidelity scene prompt with continuity lock.\n"
            "Use the exact per-scene character numbers/aliases in the prompt text so every "
            "character is unambiguous and visually distinct from the others.\n\n"
            f"Scene ID: {scene.get('scene_id')}\n"
            f"Narrative: {scene.get('narrative_description', '')}\n"
            f"Base prompt: {scene.get('scene_prompt', '')}\n"
            f"Camera style: {scene.get('camera_style', '')}\n"
            f"Emotional tone: {scene.get('emotional_tone', '')}\n"
            f"Time of day: {scene.get('time_of_day', '')}\n"
            f"Weather: {scene.get('weather', '')}\n"
            f"Continuity constraints: {scene.get('continuity_constraints', '')}\n\n"
            + char_alias_note
            + "── CHARACTER APPEARANCE LOCKS (must be honoured exactly) ──\n"
            + ("\n\n".join(char_blocks) if char_blocks else "None")
            + "\n\n"
            f"Location: {location.get('name', scene.get('location_id', 'unknown'))} | "
            f"{location.get('visual_description', '')}\n"
            f"Cinematic style: {creative_document.get('cinematic_style', '')}\n"
            f"Color grading: {creative_document.get('color_grading', '')}\n"
            f"Memory state: {memory_state}\n"
        )

        refined = self._call_llm(system, user).strip()
        return refined if refined else scene.get("scene_prompt", "")

    # ─────────────────────────────────────────────────────────────────────────
    # Critique Scene Prompt (local LLM JSON critic)
    # ─────────────────────────────────────────────────────────────────────────

    def critique_scene_prompt(
        self,
        scene:             Dict[str, Any],
        scene_prompt:      str,
        creative_document: Dict[str, Any],
        memory_state:      str,
    ) -> Dict[str, Any]:
        system = "You are a strict cinematic continuity critic. Return strict JSON only."
        user = (
            'Return JSON: {"score":0.0-1.0, "accept":bool, "issues":[str], "revised_prompt":"..."}\n'
            "Evaluate this scene prompt for:\n"
            "1. Character physical appearance consistency (skin tone, body type, height, hair, eyes, expression).\n"
            "2. Character identity consistency (each character has a unique numbered alias).\n"
            "3. Location consistency and narrative continuity.\n"
            f"Scene metadata: {json.dumps(scene, ensure_ascii=False)}\n"
            f"Scene prompt: {scene_prompt}\n"
            f"Cinematic style: {creative_document.get('cinematic_style', '')}\n"
            f"Color grading: {creative_document.get('color_grading', '')}\n"
            f"Memory state: {memory_state}\n"
        )
        try:
            response = self._call_llm(system, user)
            parsed   = self._extract_json_block(response)
        except Exception:
            parsed = {"score": 0.5, "accept": True, "issues": [], "revised_prompt": scene_prompt}

        score  = max(0.0, min(1.0, float(parsed.get("score", 0.5))))
        accept = bool(parsed.get("accept", score >= 0.7))
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        revised_prompt = str(parsed.get("revised_prompt", scene_prompt)).strip() or scene_prompt

        return {
            "score":          score,
            "accept":         accept,
            "issues":         [str(i) for i in issues],
            "revised_prompt": revised_prompt,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PPO policy update
    # ─────────────────────────────────────────────────────────────────────────

    def update_policy(
        self,
        scene_state_text:   str,
        action_prompt_text: str,
        reward:             float,
    ) -> Dict[str, Any]:
        query_tensor    = self.tokenizer.encode(scene_state_text,   return_tensors="pt").to(self.device)[0]
        response_tensor = self.tokenizer.encode(action_prompt_text, return_tensors="pt").to(self.device)[0]
        reward_tensor   = torch.tensor(float(reward), dtype=torch.float32).to(self.device)
        try:
            stats = self.ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])
            return {
                "status": "updated",
                "stats": {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in stats.items()
                },
            }
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Enricher
# ──────────────────────────────────────────────────────────────────────────────

class PromptEnricher:
    def __init__(self, director: Optional[DirectorAgent] = None):
        self.director = director

    def enrich(
        self,
        scene:             Dict[str, Any],
        creative_document: Dict[str, Any],
        memory_state:      str,
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

        # Use numbered aliases consistent with refine_scene_prompt
        scene_char_assignments: Dict[str, str] = {}
        if self.director is not None:
            scene_char_assignments = self.director._build_scene_character_assignments(
                scene, characters
            )

        char_blocks: List[str] = []
        for cid in scene.get("character_ids", []):
            char = characters.get(cid)
            if not char:
                continue
            if self.director is not None:
                label = scene_char_assignments.get(cid, cid)
                char_blocks.append(self.director._format_character_block(char, label))
            else:
                # Fallback: compact inline format with all appearance fields
                char_blocks.append(
                    f"{char.get('name', cid)} | "
                    f"age={char.get('age', 'adult')} | "
                    f"gender={char.get('gender', 'unspecified')} | "
                    f"skin={char.get('skin_tone', '')} | "
                    f"body={char.get('body_type', '')} | "
                    f"height={char.get('height', '')} | "
                    f"hair={char.get('hair_color', '')} {char.get('hair_style', '')} | "
                    f"eyes={char.get('eye_color', '')} | "
                    f"expression={char.get('facial_expression', '')} | "
                    f"features={char.get('distinguishing_features', '')} | "
                    f"wardrobe={char.get('wardrobe', 'consistent outfit')} | "
                    f"visual={char.get('visual_description', '')}"
                )

        char_alias_note = ""
        if scene_char_assignments:
            alias_list = ", ".join(
                f"{label} (= {characters[cid].get('name', cid)})"
                for cid, label in scene_char_assignments.items()
                if cid in characters
            )
            char_alias_note = (
                f"Character aliases for this scene: {alias_list}\n"
                "Refer to each character by their alias so they remain uniquely identifiable "
                "and do not visually clash with each other.\n"
            )

        loc           = locations.get(scene.get("location_id"), {})
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
            + char_alias_note
            + "── CHARACTER APPEARANCE LOCKS (must be honoured exactly) ──\n"
            + ("\n\n".join(char_blocks) if char_blocks else "None")
            + "\n\n"
            f"Location: {loc.get('name', scene.get('location_id', 'unknown'))} | {location_text}\n"
            f"Cinematic style: {creative_document.get('cinematic_style', '')}\n"
            f"Color grading: {creative_document.get('color_grading', '')}\n"
            f"Continuity memory: {memory_state}\n"
            "Optimize for Wan2.2 T2V. Emphasize coherent composition, consistent character "
            "appearance, stable scenery, and cinematic motion."
        )

        if self.director is None:
            return base_prompt

        system = (
            "You are a prompt engineer for cinematic text-to-video systems. "
            "Return plain prompt text only."
        )
        user = (
            "Refine this scene spec into a high-fidelity generation prompt with concise, dense details.\n"
            "Use the character aliases exactly as provided so each character is unambiguous.\n\n"
            + base_prompt
        )
        try:
            refined = self.director._call_llm(system, user).strip()
            return refined if refined else base_prompt
        except Exception:
            return base_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Self Critic (CLIP-based visual consistency)
# ──────────────────────────────────────────────────────────────────────────────

class SelfCritic:
    def __init__(
        self,
        threshold: float = CLIP_CONSISTENCY_THRESHOLD,
        device:    Optional[str] = None,
    ):
        self.threshold = threshold
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
            frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            raise RuntimeError(f"Failed to sample frames from {clip_path}")
        return frames

    def _embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensors)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def evaluate(
        self,
        clip_path:                  str,
        character_reference_images: List[str],
        location_reference_images:  List[str],
        current_prompt:             str,
    ) -> CritiqueResult:
        try:
            frames    = self._sample_frames(clip_path, count=5)
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

            ref_emb   = self._embed_images(ref_images)
            score     = float((frame_emb @ ref_emb.T).mean().item())
            retry     = score < self.threshold
            critique  = f"Consistency score={score:.4f}. " + (
                "Below threshold; reinforce character facial traits, wardrobe, and location materials/lighting."
                if retry else "Consistency acceptable."
            )
            suggested = current_prompt
            if retry:
                suggested += (
                    " Maintain strict identity lock: same facial structure, skin tone, eye color, "
                    "hairstyle, clothing textures, and location palette as references."
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


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration  (mirrors run_llm_pipeline exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_rl_dirs(output_root: str) -> Dict[str, str]:
    output_dir      = os.path.join(output_root, OUTPUT_DIR,      "rl_only")
    story_bible_dir = os.path.join(output_root, STORY_BIBLE_DIR, "rl_only")
    memory_dir      = os.path.join(output_root, "memory_rl")
    for path in [output_dir, story_bible_dir, memory_dir]:
        os.makedirs(path, exist_ok=True)
    return {
        "output_dir":      output_dir,
        "story_bible_dir": story_bible_dir,
        "memory_dir":      memory_dir,
    }


def run_rl_pipeline(
    seed_prompt:          str,
    output_root:          str,
    local_model_name:     str,
    resume:               bool,
) -> str:
    print("=" * 80)
    print("RL-ONLY CINEMATIC AGENT PIPELINE  (fully local, RunPod)")
    print("=" * 80)

    dirs        = _ensure_rl_dirs(output_root)
    memory_path = os.path.join(dirs["memory_dir"], "state_rl.json")
    memory      = RLRunMemory(state_path=memory_path)

    if resume and memory.load():
        print("[INFO] Loaded existing RL memory state")
    else:
        print("[INFO] Starting fresh RL run")

    # ── Boot local LLM (loaded once, reused for all calls) ───────────────────
    local_llm = LocalLLM(model_name_or_path=local_model_name)
    agent     = DirectorAgent(local_llm=local_llm)

    creative_doc_path = os.path.join(dirs["story_bible_dir"], "creative_document_rl.json")
    prompts_path      = os.path.join(dirs["output_dir"],      "scene_prompts_rl.json")
    report_path       = os.path.join(dirs["output_dir"],      "rl_pipeline_report.json")

    # ── Step 1: Generate Creative Document ───────────────────────────────────
    print("[STEP] Generate creative document")
    creative_document = agent.generate_creative_document(
        seed_prompt=seed_prompt,
        memory_state=memory.get_state_for_agent(),
    )

    # ── Step 2: Persist full character/location appearance in memory ──────────
    for character in creative_document.get("characters", []):
        cid = str(character.get("id", ""))
        if cid:
            memory.character_registry[cid] = {
                "name":                    character.get("name",                    ""),
                "gender":                  character.get("gender",                  "unspecified"),
                "age":                     character.get("age",                     "adult"),
                "skin_tone":               character.get("skin_tone",               ""),
                "body_type":               character.get("body_type",               ""),
                "height":                  character.get("height",                  ""),
                "hair_color":              character.get("hair_color",              ""),
                "hair_style":              character.get("hair_style",              ""),
                "eye_color":               character.get("eye_color",               ""),
                "facial_expression":       character.get("facial_expression",       ""),
                "distinguishing_features": character.get("distinguishing_features", ""),
                "wardrobe":                character.get("wardrobe",                ""),
                "visual_description":      character.get("visual_description",      ""),
            }
    for location in creative_document.get("locations", []):
        lid = str(location.get("id", ""))
        if lid:
            memory.location_registry[lid] = {
                "name":             location.get("name",             ""),
                "material_palette": location.get("material_palette", ""),
            }
    memory.save()

    with open(creative_doc_path, "w", encoding="utf-8") as handle:
        json.dump(creative_document, handle, indent=2, ensure_ascii=False)

    # ── Step 3: Scene prompt refinement + PPO loop ────────────────────────────
    print("[STEP] Scene prompt refinement + PPO loop")
    characters_map = {
        item["id"]: item
        for item in creative_document.get("characters", [])
        if "id" in item
    }
    refined_scenes: List[Dict[str, Any]] = []

    for scene in creative_document.get("scenes", []):
        scene_id = str(scene.get("scene_id"))
        prompt   = scene.get("scene_prompt", scene.get("narrative_description", ""))

        scene_char_assignments = agent._build_scene_character_assignments(scene, characters_map)

        critique_result: Dict[str, Any] = {
            "score": 0.5, "accept": True, "issues": [], "revised_prompt": prompt,
        }

        for attempt in range(max(2, RL_EPISODES_PER_SCENE)):
            refined = agent.refine_scene_prompt(
                scene=scene,
                creative_document=creative_document,
                memory_state=memory.get_state_for_agent(),
            )
            critique_result = agent.critique_scene_prompt(
                scene=scene,
                scene_prompt=refined,
                creative_document=creative_document,
                memory_state=memory.get_state_for_agent(),
            )
            prompt = critique_result.get("revised_prompt", refined)
            if critique_result.get("accept", False):
                break
            if attempt + 1 < max(2, RL_EPISODES_PER_SCENE):
                prompt += " Keep identity and location details strict and remove contradictions."

        # PPO update — reward = LLM critique score
        ppo_reward = float(critique_result.get("score", 0.5))
        ppo_result = agent.update_policy(
            scene_state_text=memory.get_state_for_agent(),
            action_prompt_text=prompt,
            reward=ppo_reward,
        )

        scene["scene_prompt"]               = prompt
        scene["llm_critique"]               = {
            "score":  float(critique_result.get("score",  0.5)),
            "accept": bool(critique_result.get("accept", False)),
            "issues": critique_result.get("issues", []),
        }
        scene["scene_character_assignments"] = scene_char_assignments
        scene["ppo_update"]                  = ppo_result

        memory.update_after_scene(
            scene_id=scene_id,
            prompt=prompt,
            critique_score=float(critique_result.get("score",  0.5)),
            critique_issues=critique_result.get("issues", []),
            clip_score=0.0,       # populated post-video-generation via SelfCritic
            ppo_reward=ppo_reward,
        )

        refined_scenes.append({
            "scene_id":                   scene_id,
            "prompt":                     prompt,
            "critique":                   scene["llm_critique"],
            "ppo_reward":                 ppo_reward,
            "ppo_update":                 ppo_result,
            "scene_character_assignments": scene_char_assignments,
        })

        alias_str = ", ".join(
            f"{v}={characters_map[k].get('name', k)}"
            for k, v in scene_char_assignments.items()
            if k in characters_map
        )
        print(
            f"[SCENE {scene_id}] score={scene['llm_critique']['score']:.4f} "
            f"accept={scene['llm_critique']['accept']} "
            f"ppo_reward={ppo_reward:.4f} "
            f"chars=[{alias_str}]"
        )

    # ── Step 4: Write outputs ─────────────────────────────────────────────────
    with open(creative_doc_path, "w", encoding="utf-8") as handle:
        json.dump(creative_document, handle, indent=2, ensure_ascii=False)
    with open(prompts_path, "w", encoding="utf-8") as handle:
        json.dump(refined_scenes,    handle, indent=2, ensure_ascii=False)

    scores      = [float(item.get("critique", {}).get("score", 0.0)) for item in refined_scenes]
    mean_score  = sum(scores) / len(scores) if scores else 0.0
    mean_reward = sum(memory.reward_curve) / len(memory.reward_curve) if memory.reward_curve else 0.0

    report = {
        "pipeline":             "rl_only",
        "local_model":          local_model_name,
        "scene_count":          len(refined_scenes),
        "mean_critique_score":  float(mean_score),
        "mean_ppo_reward":      float(mean_reward),
        "reward_curve":         memory.reward_curve,
        "creative_document_path": creative_doc_path,
        "scene_prompts_path":   prompts_path,
        "memory_path":          memory_path,
        "timestamp":            int(time.time()),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"RL-only pipeline completed.  scene_count={len(refined_scenes)}")
    print(f"Mean critique score : {mean_score:.4f}")
    print(f"Mean PPO reward     : {mean_reward:.4f}")
    print(f"Creative document   : {creative_doc_path}")
    print(f"Scene prompts       : {prompts_path}")
    print(f"Memory state        : {memory_path}")
    print(f"Report              : {report_path}")
    print("=" * 80)
    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point  (mirrors LLM pipeline parse_args / main)
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone RL-only cinematic agent pipeline (fully local, RunPod)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic mystery in an old coastal city where a journalist uncovers a hidden archive.",
        help="Seed prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Project root / output base directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PPO_POLICY_MODEL_NAME,
        help=(
            "HuggingFace model name or local path for the director LLM "
            "(e.g. meta-llama/Meta-Llama-3-8B-Instruct, /workspace/models/qwen2.5-7b)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from memory_rl/state_rl.json if available",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report_path = run_rl_pipeline(
            seed_prompt=args.prompt,
            output_root=args.output,
            local_model_name=args.model,
            resume=args.resume,
        )
        print(f"RL pipeline report: {report_path}")
        return 0
    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())