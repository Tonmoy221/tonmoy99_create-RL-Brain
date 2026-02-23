import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

from agent import DirectorAgent
from config import (
    CLIPS_DIR,
    MEMORY_DIR,
    MIN_VRAM_GB,
    OUTPUT_DIR,
    STORY_BIBLE_DIR,
)
from memory import ConsistencyMemory
from preproduction import ReferenceGenerator
from reward_model import RewardModel
from scene_pipeline import SceneGenerator
from stitcher import VideoStitcher


def _check_vram() -> float:
    if not torch.cuda.is_available():
        print(
            "[WARN] CUDA is not available. Cinematic mode requires a CUDA GPU for practical runtime."
        )
        return 0.0
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / (1024**3)
    if total_gb < MIN_VRAM_GB:
        print(
            f"[WARN] Detected VRAM: {total_gb:.2f}GB, recommended minimum is {MIN_VRAM_GB}GB for cinematic mode."
        )
    else:
        print(f"[INFO] Detected VRAM: {total_gb:.2f}GB")
    return total_gb


def _ensure_dirs(output_root: str) -> Dict[str, str]:
    output_dir = os.path.join(output_root, OUTPUT_DIR)
    clips_dir = os.path.join(output_root, CLIPS_DIR)
    memory_dir = os.path.join(output_root, MEMORY_DIR)
    story_bible_dir = os.path.join(output_root, STORY_BIBLE_DIR)
    references_dir = os.path.join(story_bible_dir, "references")

    for path in [output_dir, clips_dir, memory_dir, story_bible_dir, references_dir]:
        os.makedirs(path, exist_ok=True)

    return {
        "output_dir": output_dir,
        "clips_dir": clips_dir,
        "memory_dir": memory_dir,
        "story_bible_dir": story_bible_dir,
        "references_dir": references_dir,
    }


def _collect_refs(scene: Dict, creative_document: Dict) -> Tuple[List[str], List[str]]:
    char_map = {
        item.get("id"): item for item in creative_document.get("characters", [])
    }
    loc_map = {item.get("id"): item for item in creative_document.get("locations", [])}

    character_refs = []
    for char_id in scene.get("character_ids", []):
        character_refs.extend(
            char_map.get(char_id, {}).get("reference_image_paths", [])
        )

    location_refs = loc_map.get(scene.get("location_id"), {}).get(
        "reference_image_paths", []
    )
    return list(character_refs), list(location_refs)


def run_cinematic_mode(seed_prompt: str, output_root: str) -> str:
    print("=" * 80)
    print("RL CINEMATIC VIDEO GENERATION SYSTEM (WAN2.2)")
    print("=" * 80)

    _check_vram()
    dirs = _ensure_dirs(output_root)
    start_time = time.time()

    memory = ConsistencyMemory(output_root=output_root)

    print("[STEP] Memory initialization")
    state_exists = os.path.exists(os.path.join(dirs["memory_dir"], "state.json"))
    resume = False
    if state_exists:
        choice = (
            input(
                "Found existing memory state. Resume from last completed scene? [y/N]: "
            )
            .strip()
            .lower()
        )
        resume = choice in {"y", "yes"}
    if resume:
        memory.load()
        print("[INFO] Resuming from saved state")
    else:
        print("[INFO] Starting fresh run")

    director = DirectorAgent()
    reward_model = RewardModel()
    scene_generator = SceneGenerator(output_root=output_root)
    reference_generator = ReferenceGenerator(output_root=output_root)
    stitcher = VideoStitcher(output_root=output_root)

    creative_doc_path = os.path.join(dirs["story_bible_dir"], "creative_document.json")

    print("[STEP] Generate Creative Document")
    try:
        if resume and os.path.exists(creative_doc_path):
            with open(creative_doc_path, "r", encoding="utf-8") as handle:
                creative_document = json.load(handle)
        else:
            creative_document = director.generate_creative_document(
                seed_prompt, memory.get_state_for_agent()
            )
            with open(creative_doc_path, "w", encoding="utf-8") as handle:
                json.dump(creative_document, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise RuntimeError(f"Creative document stage failed: {exc}") from exc

    print("[STEP] Generate reference assets")
    try:
        creative_document = reference_generator.generate_all(creative_document)
        with open(creative_doc_path, "w", encoding="utf-8") as handle:
            json.dump(creative_document, handle, indent=2, ensure_ascii=False)

        for character in creative_document.get("characters", []):
            cid = character.get("id")
            if cid is None:
                continue
            memory.character_registry[str(cid)] = {
                "name": character.get("name", ""),
                "age": character.get("age", "adult"),
                "gender": character.get("gender", "unspecified"),
                "wardrobe": character.get("wardrobe", "consistent outfit"),
                "text_description": character.get("visual_description", ""),
                "reference_image_paths": character.get("reference_image_paths", []),
                "clip_embedding": None,
            }

        for location in creative_document.get("locations", []):
            lid = location.get("id")
            if lid is None:
                continue
            memory.location_registry[str(lid)] = {
                "name": location.get("name", ""),
                "scenery_type": location.get("scenery_type", "environment"),
                "material_palette": location.get("material_palette", "neutral tones"),
                "text_description": location.get("visual_description", ""),
                "reference_image_paths": location.get("reference_image_paths", []),
                "clip_embedding": None,
            }
        memory.save()
    except Exception as exc:
        raise RuntimeError(f"Reference generation stage failed: {exc}") from exc

    scenes = creative_document.get("scenes", [])
    if not scenes:
        raise RuntimeError("Creative document has no scenes")

    completed_scene_ids = (
        {str(item.get("scene_id")) for item in memory.rl_episode_log}
        if resume
        else set()
    )
    clip_paths: List[str] = []

    print("[STEP] Scene loop")
    for scene in scenes:
        scene_id = str(scene.get("scene_id"))
        if scene_id in completed_scene_ids:
            existing_clip = os.path.join(dirs["clips_dir"], f"scene_{scene_id}.mp4")
            if os.path.exists(existing_clip):
                clip_paths.append(existing_clip)
            print(f"[INFO] Skipping completed scene {scene_id}")
            continue

        try:
            clip_path = scene_generator.generate_scene(scene, memory, creative_document)
            character_refs, location_refs = _collect_refs(scene, creative_document)
            used_prompt = scene.get(
                "scene_prompt", scene.get("narrative_description", seed_prompt)
            )

            reward = reward_model.compute(
                scene_id=scene_id,
                clip_path=clip_path,
                prompt=used_prompt,
                character_refs=character_refs,
                location_refs=location_refs,
                continuity_log=memory.continuity_log,
            )

            policy_result = director.update_policy(
                memory.get_state_for_agent(), used_prompt, reward
            )
            if policy_result.get("status") != "updated":
                print(
                    f"[WARN] PPO update issue for scene {scene_id}: {policy_result.get('error', 'unknown')}"
                )

            memory.update_after_scene(
                scene_id=scene_id,
                clip_path=clip_path,
                prompt=used_prompt,
                reward=reward,
            )
            if memory.rl_episode_log:
                memory.rl_episode_log[-1]["reward_breakdown"] = dict(
                    reward_model.latest_breakdown
                )
                memory.rl_episode_log[-1]["clip_path"] = clip_path
                memory.rl_episode_log[-1]["timestamp"] = time.time()
                memory.save()

            clip_paths.append(clip_path)

            breakdown = reward_model.latest_breakdown
            print(
                f"[SCENE {scene_id}] reward={breakdown.get('final', reward):.4f} "
                f"(visual={breakdown.get('visual', 0.0):.4f}, "
                f"character={breakdown.get('character', 0.0):.4f}, "
                f"location={breakdown.get('location', 0.0):.4f}, "
                f"narrative={breakdown.get('narrative', 0.0):.4f})"
            )
        except Exception as exc:
            print(f"[ERROR] Scene {scene_id} failed: {exc}")
            raise

    print("[STEP] Stitch final video")
    final_video_path = os.path.join(dirs["output_dir"], "final_video.mp4")
    try:
        final_video_path = stitcher.stitch(clip_paths, final_video_path)
    except Exception as exc:
        raise RuntimeError(f"Video stitching failed: {exc}") from exc

    print("[STEP] Generate production report")
    try:
        report_path = stitcher.generate_production_report(
            creative_document, memory, final_video_path
        )
    except Exception as exc:
        raise RuntimeError(f"Production report generation failed: {exc}") from exc

    duration_min = (time.time() - start_time) / 60.0
    reward_curve = [
        f"{item.get('reward_received', 0.0):.4f}" for item in memory.rl_episode_log
    ]
    print("=" * 80)
    print(f"Completed cinematic generation in {duration_min:.2f} minutes")
    print(f"Reward curve: {reward_curve}")
    print(f"Final video: {final_video_path}")
    print(f"Production report: {report_path}")
    print("=" * 80)
    return final_video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.2 video launcher in RL-cinematic mode"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic wide shot of a lone traveler crossing a stormy sea at golden hour.",
        help="Seed prompt",
    )
    parser.add_argument(
        "--output", type=str, default=".", help="Project root/output base directory"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out = run_cinematic_mode(seed_prompt=args.prompt, output_root=args.output)
        print(f"Cinematic mode output: {out}")
        return 0
    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
