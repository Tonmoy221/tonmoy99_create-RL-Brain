import json
import os
import time
from typing import Dict, List

from moviepy import VideoFileClip, concatenate_videoclips

from config import CROSSFADE_DURATION, OUTPUT_DIR, PRODUCTION_REPORT_NAME


class VideoStitcher:
    def __init__(self, output_root: str = "."):
        self.output_root = output_root
        self.default_output_dir = os.path.join(self.output_root, OUTPUT_DIR)
        os.makedirs(self.default_output_dir, exist_ok=True)

    def stitch(self, clip_paths: List[str], output_path: str) -> str:
        valid_paths = [path for path in clip_paths if os.path.exists(path)]
        if not valid_paths:
            raise RuntimeError("No valid clip paths available to stitch")

        clips = []
        try:
            for path in valid_paths:
                clips.append(VideoFileClip(path))

            final_clip = concatenate_videoclips(
                clips, method="compose", padding=-CROSSFADE_DURATION
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_clip.write_videofile(
                output_path, codec="libx264", audio_codec="aac", fps=24
            )
            final_clip.close()
            return output_path
        finally:
            for clip in clips:
                try:
                    clip.close()
                except Exception:
                    pass

    def generate_production_report(
        self, creative_document: Dict, memory, output_path: str
    ) -> str:
        report_path = os.path.join(self.default_output_dir, PRODUCTION_REPORT_NAME)

        rewards = [
            float(item.get("reward_received", 0.0)) for item in memory.rl_episode_log
        ]
        retries = sum(int(item.get("retry_count", 0)) for item in memory.rl_episode_log)

        per_scene_breakdown = []
        for item in memory.rl_episode_log:
            per_scene_breakdown.append(
                {
                    "scene_id": item.get("scene_id"),
                    "reward": float(item.get("reward_received", 0.0)),
                    "visual": float(
                        item.get("reward_breakdown", {}).get("visual", 0.0)
                    ),
                    "character": float(
                        item.get("reward_breakdown", {}).get("character", 0.0)
                    ),
                    "location": float(
                        item.get("reward_breakdown", {}).get("location", 0.0)
                    ),
                    "narrative": float(
                        item.get("reward_breakdown", {}).get("narrative", 0.0)
                    ),
                    "final": float(
                        item.get("reward_breakdown", {}).get(
                            "final", item.get("reward_received", 0.0)
                        )
                    ),
                    "prompt_used": item.get("prompt_used", ""),
                    "clip_path": item.get("clip_path", ""),
                }
            )

        all_assets = {
            "character_refs": [
                path
                for character in creative_document.get("characters", [])
                for path in character.get("reference_image_paths", [])
            ],
            "location_refs": [
                path
                for location in creative_document.get("locations", [])
                for path in location.get("reference_image_paths", [])
            ],
            "last_frame_path": memory.last_frame_path,
            "final_video": output_path,
        }

        start_ts = None
        end_ts = None
        if memory.rl_episode_log:
            start_ts = memory.rl_episode_log[0].get("timestamp")
            end_ts = memory.rl_episode_log[-1].get("timestamp")

        elapsed_minutes = 0.0
        if start_ts is not None and end_ts is not None:
            elapsed_minutes = max(0.0, (float(end_ts) - float(start_ts)) / 60.0)
        elif rewards:
            elapsed_minutes = len(rewards) * 1.5

        report = {
            "creative_document": creative_document,
            "per_scene_reward_breakdown": per_scene_breakdown,
            "rl_reward_improvement_curve": rewards,
            "total_retries": retries,
            "total_generation_time_minutes": float(elapsed_minutes),
            "paths_to_generated_assets": all_assets,
            "generated_at_unix": time.time(),
        }

        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

        return report_path
