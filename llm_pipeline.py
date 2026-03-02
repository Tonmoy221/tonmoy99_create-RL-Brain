#LLM AGENT BRAIN-Promt Spliting

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_SCENES,
    MIN_SCENES,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT_SECONDS,
    OPENAI_API_KEY_ENV,
    OPENAI_TIMEOUT_SECONDS,
    OUTPUT_DIR,
    RL_EPISODES_PER_SCENE,
    STORY_BIBLE_DIR,
)


@dataclass
class LLMRunMemory:
    state_path: str
    character_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    location_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    continuity_log: List[str] = field(default_factory=list)
    llm_episode_log: List[Dict[str, Any]] = field(default_factory=list)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        payload = {
            "character_registry": self.character_registry,
            "location_registry": self.location_registry,
            "continuity_log": self.continuity_log,
            "llm_episode_log": self.llm_episode_log,
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.character_registry = payload.get("character_registry", {})
        self.location_registry = payload.get("location_registry", {})
        self.continuity_log = payload.get("continuity_log", [])
        self.llm_episode_log = payload.get("llm_episode_log", [])
        return True

    def get_state_for_agent(self) -> str:
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
            "LLMRunMemory State\n"
            f"Characters: {character_summary if character_summary else 'None'}\n"
            f"Locations: {location_summary if location_summary else 'None'}\n"
            f"Recent continuity: {self.continuity_log[-5:] if self.continuity_log else 'None'}"
        )

    def update_after_scene(
        self,
        scene_id: str,
        prompt: str,
        critique_score: float,
        critique_issues: List[str],
    ) -> None:
        self.continuity_log.append(
            f"Scene {scene_id} finalized. score={critique_score:.4f}. prompt={prompt[:220]}"
        )
        self.llm_episode_log.append(
            {
                "scene_id": scene_id,
                "prompt_used": prompt,
                "critique_score": float(critique_score),
                "critique_issues": [str(item) for item in critique_issues],
                "timestamp": int(time.time()),
            }
        )
        self.save()


class LLMDirectorAgent:
    def __init__(self, provider: str = LLM_PROVIDER, model_name: str = LLM_MODEL):
        self.provider = provider
        self.model_name = model_name

    # ── Retry settings for free-tier rate limits ──
    _RATE_MAX_RETRIES = 8
    _RATE_BASE_DELAY = 15  # seconds
    _RATE_MAX_DELAY = 120  # seconds

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        import time
        import random

        try:
            from openai import OpenAI, RateLimitError as _OAIRateLimit

            api_key = os.getenv(OPENAI_API_KEY_ENV)
            if not api_key:
                raise RuntimeError(f"Missing {OPENAI_API_KEY_ENV} environment variable")

            base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
            client_kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "timeout": OPENAI_TIMEOUT_SECONDS,
            }
            if base_url is not None:
                client_kwargs["base_url"] = base_url

            client = OpenAI(**client_kwargs)

            extra_headers: Dict[str, str] = {}
            http_referer = os.getenv("OPENAI_HTTP_REFERER", "").strip()
            x_title = os.getenv("OPENAI_X_TITLE", "").strip()
            if http_referer:
                extra_headers["HTTP-Referer"] = http_referer
            if x_title:
                extra_headers["X-Title"] = x_title

            # Build two possible message lists
            msgs_system = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            merged_user_prompt = (
                "Follow these instructions exactly:\n"
                f"{system_prompt}\n\n"
                "User request:\n"
                f"{user_prompt}"
            )
            msgs_user_only = [{"role": "user", "content": merged_user_prompt}]

            use_system = True  # start by trying system+user

            for attempt in range(1, self._RATE_MAX_RETRIES + 1):
                messages = msgs_system if use_system else msgs_user_only
                try:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        temperature=0.4,
                        messages=messages,
                        extra_headers=extra_headers if extra_headers else None,
                    )
                    return response.choices[0].message.content or ""

                except _OAIRateLimit:
                    # 429 – wait with exponential backoff + jitter then retry
                    delay = min(
                        self._RATE_BASE_DELAY * (2 ** (attempt - 1)),
                        self._RATE_MAX_DELAY,
                    ) + random.uniform(0, 5)
                    print(
                        f"  [RATE-LIMIT] attempt {attempt}/{self._RATE_MAX_RETRIES}, "
                        f"sleeping {delay:.1f}s …"
                    )
                    time.sleep(delay)

                except Exception as exc:
                    text = str(exc).lower()
                    if (
                        "developer instruction is not enabled" in text
                        or ("system" in text and "invalid_argument" in text)
                    ) and use_system:
                        # Switch to user-only messages and retry immediately
                        print("  [FALLBACK] system role rejected → user-only messages")
                        use_system = False
                        continue
                    raise

            raise RuntimeError(
                f"OpenRouter rate-limited after {self._RATE_MAX_RETRIES} retries. "
                "Wait a few minutes and retry."
            )
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

        if not isinstance(doc.get("characters", []), list):
            errors.append("characters must be a list")
        if not isinstance(doc.get("locations", []), list):
            errors.append("locations must be a list")
        if not isinstance(doc.get("scenes", []), list):
            errors.append("scenes must be a list")

        scenes = doc.get("scenes", [])
        if len(scenes) < MIN_SCENES:
            errors.append(
                f"Scene count too low: expected at least {MIN_SCENES}, got {len(scenes)}"
            )
        if len(scenes) > MAX_SCENES:
            errors.append(f"Scene count too high: max {MAX_SCENES}, got {len(scenes)}")

        character_ids = {
            c.get("id") for c in doc.get("characters", []) if isinstance(c, dict)
        }
        location_ids = {
            l.get("id") for l in doc.get("locations", []) if isinstance(l, dict)
        }

        for index, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                errors.append(f"Scene {index} must be an object")
                continue

            for key in [
                "scene_id",
                "narrative_description",
                "scene_prompt",
                "character_ids",
                "location_id",
                "camera_style",
                "emotional_tone",
                "time_of_day",
                "weather",
                "continuity_constraints",
            ]:
                if key not in scene:
                    errors.append(f"Scene {index} missing {key}")

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
                    # ── Detailed physical appearance ──
                    "skin_tone": item.get("skin_tone", "medium"),
                    "body_type": item.get(
                        "body_type", "average"
                    ),  # slim/athletic/average/heavyset/muscular
                    "height": item.get(
                        "height", "average height"
                    ),  # e.g. "5ft 10in, tall"
                    "hair_color": item.get("hair_color", "brown"),
                    "hair_style": item.get("hair_style", "medium length"),
                    "eye_color": item.get("eye_color", "brown"),
                    "facial_expression": item.get(
                        "facial_expression", "neutral"
                    ),  # happy/sad/angry/determined/scared
                    "distinguishing_features": item.get(
                        "distinguishing_features", ""
                    ),  # scars, tattoos, beard, glasses
                    # ── Wardrobe & full visual summary ──
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

    @staticmethod
    def _build_scene_character_assignments(
        scene: Dict[str, Any],
        characters: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Assign a stable per-scene number to each character that appears in a scene.
        Returns a mapping like {"char_1": "Character_1", "char_2": "Character_2"}.
        This prevents multi-character prompt clashes across scenes.
        """
        assignments: Dict[str, str] = {}
        for slot_idx, cid in enumerate(scene.get("character_ids", []), start=1):
            if cid in characters:
                assignments[cid] = f"Character_{slot_idx}"
        return assignments

    @staticmethod
    def _format_character_block(
        char: Dict[str, Any],
        label: str,
    ) -> str:
        """
        Produce a verbose, locked character description block for prompt injection.
        The `label` is the scene-local numbered alias e.g. 'Character_1'.
        """
        lines = [
            f"[{label} | ID={char.get('id')} | Name={char.get('name', label)}]",
            f"  Gender      : {char.get('gender', 'unspecified')}",
            f"  Age         : {char.get('age', 'adult')}",
            f"  Skin tone   : {char.get('skin_tone', 'medium')}",
            f"  Body type   : {char.get('body_type', 'average')}",
            f"  Height      : {char.get('height', 'average height')}",
            f"  Hair        : {char.get('hair_color', 'brown')} | {char.get('hair_style', 'medium length')}",
            f"  Eye color   : {char.get('eye_color', 'brown')}",
            f"  Expression  : {char.get('facial_expression', 'neutral')}",
            f"  Features    : {char.get('distinguishing_features', 'none')}",
            f"  Wardrobe    : {char.get('wardrobe', 'consistent outfit')}",
            f"  Full desc.  : {char.get('visual_description', '')}",
        ]
        return "\n".join(lines)

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

        issues: List[str] = []
        for _ in range(max(2, RL_EPISODES_PER_SCENE)):
            augmentation = ""
            if issues:
                augmentation = (
                    "Fix these issues from previous attempt:\n- " + "\n- ".join(issues)
                )

            response = self._call_llm(system, user_template + "\n" + augmentation)
            normalized = self._normalize_creative_document(
                self._extract_json_block(response)
            )
            valid, issues = self._validate_creative_document(normalized)
            if valid:
                return normalized

        raise RuntimeError(
            f"Failed to generate valid Creative Document after retries. Last issues: {issues}"
        )

    def refine_scene_prompt(
        self,
        scene: Dict[str, Any],
        creative_document: Dict[str, Any],
        memory_state: str,
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

        # ── Assign per-scene numbered labels for every character in this scene ──
        # e.g. Character_1, Character_2 … to avoid confusion when multiple characters exist.
        scene_char_assignments = self._build_scene_character_assignments(
            scene, characters
        )

        char_blocks: List[str] = []
        for cid in scene.get("character_ids", []):
            char = characters.get(cid)
            if not char:
                continue
            label = scene_char_assignments.get(cid, cid)
            char_blocks.append(self._format_character_block(char, label))

        location = locations.get(scene.get("location_id"), {})

        # Build a numbered-character reference note for the LLM
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

    def critique_scene_prompt(
        self,
        scene: Dict[str, Any],
        scene_prompt: str,
        creative_document: Dict[str, Any],
        memory_state: str,
    ) -> Dict[str, Any]:
        system = (
            "You are a strict cinematic continuity critic. Return strict JSON only."
        )
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
            parsed = self._extract_json_block(response)
        except Exception:
            parsed = {
                "score": 0.5,
                "accept": True,
                "issues": [],
                "revised_prompt": scene_prompt,
            }

        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        accept = bool(parsed.get("accept", score >= 0.7))
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        revised_prompt = str(parsed.get("revised_prompt", scene_prompt)).strip()
        if not revised_prompt:
            revised_prompt = scene_prompt

        return {
            "score": score,
            "accept": accept,
            "issues": [str(item) for item in issues],
            "revised_prompt": revised_prompt,
        }


def _ensure_llm_dirs(output_root: str) -> Dict[str, str]:
    output_dir = os.path.join(output_root, OUTPUT_DIR, "llm_only")
    story_bible_dir = os.path.join(output_root, STORY_BIBLE_DIR, "llm_only")
    memory_dir = os.path.join(output_root, "memory_llm")
    for path in [output_dir, story_bible_dir, memory_dir]:
        os.makedirs(path, exist_ok=True)
    return {
        "output_dir": output_dir,
        "story_bible_dir": story_bible_dir,
        "memory_dir": memory_dir,
    }


def run_llm_pipeline(
    seed_prompt: str,
    output_root: str,
    provider: str,
    model_name: str,
    resume: bool,
) -> str:
    print("=" * 80)
    print("LLM-ONLY CINEMATIC AGENT PIPELINE")
    print("=" * 80)

    dirs = _ensure_llm_dirs(output_root)
    memory_path = os.path.join(dirs["memory_dir"], "state_llm.json")
    memory = LLMRunMemory(state_path=memory_path)

    if resume and memory.load():
        print("[INFO] Loaded existing LLM memory state")
    else:
        print("[INFO] Starting fresh LLM-only run")

    agent = LLMDirectorAgent(provider=provider, model_name=model_name)

    creative_doc_path = os.path.join(
        dirs["story_bible_dir"], "creative_document_llm.json"
    )
    prompts_path = os.path.join(dirs["output_dir"], "scene_prompts_llm.json")
    report_path = os.path.join(dirs["output_dir"], "llm_pipeline_report.json")

    print("[STEP] Generate creative document")
    creative_document = agent.generate_creative_document(
        seed_prompt=seed_prompt,
        memory_state=memory.get_state_for_agent(),
    )

    # ── Persist full character appearance in memory registry ──
    for character in creative_document.get("characters", []):
        cid = str(character.get("id", ""))
        if cid:
            memory.character_registry[cid] = {
                "name": character.get("name", ""),
                "gender": character.get("gender", "unspecified"),
                "age": character.get("age", "adult"),
                "skin_tone": character.get("skin_tone", ""),
                "body_type": character.get("body_type", ""),
                "height": character.get("height", ""),
                "hair_color": character.get("hair_color", ""),
                "hair_style": character.get("hair_style", ""),
                "eye_color": character.get("eye_color", ""),
                "facial_expression": character.get("facial_expression", ""),
                "distinguishing_features": character.get("distinguishing_features", ""),
                "wardrobe": character.get("wardrobe", ""),
                "visual_description": character.get("visual_description", ""),
            }
    for location in creative_document.get("locations", []):
        lid = str(location.get("id", ""))
        if lid:
            memory.location_registry[lid] = {
                "name": location.get("name", ""),
                "material_palette": location.get("material_palette", ""),
            }
    memory.save()

    with open(creative_doc_path, "w", encoding="utf-8") as handle:
        json.dump(creative_document, handle, indent=2, ensure_ascii=False)

    print("[STEP] Scene prompt refinement loop")
    refined_scenes: List[Dict[str, Any]] = []
    for scene in creative_document.get("scenes", []):
        scene_id = str(scene.get("scene_id"))
        prompt = scene.get("scene_prompt", scene.get("narrative_description", ""))

        # ── Build per-scene character number assignment for logging ──
        characters_map = {
            item["id"]: item
            for item in creative_document.get("characters", [])
            if "id" in item
        }
        scene_char_assignments = agent._build_scene_character_assignments(
            scene, characters_map
        )

        critique_result: Dict[str, Any] = {
            "score": 0.5,
            "accept": True,
            "issues": [],
            "revised_prompt": prompt,
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
                prompt = (
                    prompt
                    + " Keep identity and location details strict and remove contradictions."
                )

        scene["scene_prompt"] = prompt
        scene["llm_critique"] = {
            "score": float(critique_result.get("score", 0.5)),
            "accept": bool(critique_result.get("accept", False)),
            "issues": critique_result.get("issues", []),
        }
        # Store which numbered alias was assigned to each character in this scene
        scene["scene_character_assignments"] = scene_char_assignments

        memory.update_after_scene(
            scene_id=scene_id,
            prompt=prompt,
            critique_score=float(critique_result.get("score", 0.5)),
            critique_issues=critique_result.get("issues", []),
        )

        refined_scenes.append(
            {
                "scene_id": scene_id,
                "prompt": prompt,
                "critique": scene["llm_critique"],
                # Numbered character aliases for downstream consumers
                "scene_character_assignments": scene_char_assignments,
            }
        )
        alias_str = ", ".join(
            f"{v}={characters_map[k].get('name', k)}"
            for k, v in scene_char_assignments.items()
            if k in characters_map
        )
        print(
            f"[SCENE {scene_id}] score={scene['llm_critique']['score']:.4f} "
            f"accept={scene['llm_critique']['accept']} "
            f"chars=[{alias_str}]"
        )

    with open(creative_doc_path, "w", encoding="utf-8") as handle:
        json.dump(creative_document, handle, indent=2, ensure_ascii=False)

    with open(prompts_path, "w", encoding="utf-8") as handle:
        json.dump(refined_scenes, handle, indent=2, ensure_ascii=False)

    scores = [
        float(item.get("critique", {}).get("score", 0.0)) for item in refined_scenes
    ]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    report = {
        "pipeline": "llm_only",
        "provider": provider,
        "model": model_name,
        "scene_count": len(refined_scenes),
        "mean_critique_score": float(mean_score),
        "creative_document_path": creative_doc_path,
        "scene_prompts_path": prompts_path,
        "memory_path": memory_path,
        "timestamp": int(time.time()),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"LLM-only pipeline completed. scene_count={len(refined_scenes)}")
    print(f"Mean critique score: {mean_score:.4f}")
    print(f"Creative document: {creative_doc_path}")
    print(f"Scene prompts: {prompts_path}")
    print(f"Memory state: {memory_path}")
    print(f"Report: {report_path}")
    print("=" * 80)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone LLM-only cinematic agent pipeline"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic mystery in an old coastal city where a journalist uncovers a hidden archive.",
        help="Seed prompt",
    )
    parser.add_argument(
        "--output", type=str, default=".", help="Project root/output base directory"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=LLM_PROVIDER,
        help="LLM provider (openai or ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help="LLM model name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from memory_llm/state_llm.json if available",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report_path = run_llm_pipeline(
            seed_prompt=args.prompt,
            output_root=args.output,
            provider=args.provider,
            model_name=args.model,
            resume=args.resume,
        )
        print(f"LLM pipeline report: {report_path}")
        return 0
    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
