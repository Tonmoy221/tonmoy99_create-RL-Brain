import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from agent import DirectorAgent, PromptEnricher, SelfCritic
from config import (
    CLIPS_DIR,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
    IP_ADAPTER_SCALE,
    RL_EPISODES_PER_SCENE,
    VIDEO_FPS,
)
from utils.data import save_video
from wan_video_new import ModelConfig, WanVideoPipeline


class SceneGenerator:
    def __init__(self, output_root: str = ".", device: Optional[str] = None):
        self.output_root = output_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clips_dir = os.path.join(self.output_root, CLIPS_DIR)
        os.makedirs(self.clips_dir, exist_ok=True)

        self.director_proxy = DirectorAgent()
        self.prompt_enricher = PromptEnricher(director=self.director_proxy)
        self.self_critic = SelfCritic()

    def _build_model_configs(self) -> Tuple[List[ModelConfig], ModelConfig]:
        model_root = os.getenv("WAN_MODEL_ROOT", "./models")
        diffusion_model = os.getenv(
            "WAN_DIFFUSION_MODEL",
            os.path.join(model_root, "PAI", "Wan2.1-Fun-V1.1-1.3B-InP"),
        )
        t5_path = os.getenv(
            "WAN_T5_PATH",
            os.path.join(
                model_root,
                "Wan-AI",
                "Wan2.1-T2V-1.3B",
                "models_t5_umt5-xxl-enc-bf16.pth",
            ),
        )
        vae_path = os.getenv(
            "WAN_VAE_PATH",
            os.path.join(model_root, "Wan-AI", "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"),
        )
        clip_path = os.getenv(
            "WAN_CLIP_PATH",
            os.path.join(
                model_root,
                "Wan-AI",
                "Wan2.1-I2V-14B-480P",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
        )
        tokenizer_path = os.getenv(
            "WAN_TOKENIZER_PATH",
            os.path.join(model_root, "Wan-AI", "Wan2.1-T2V-1.3B", "google", "umt5-xxl"),
        )

        configs = [
            ModelConfig(
                model_id=diffusion_model,
                origin_file_pattern="diffusion_pytorch_model.safetensors",
            ),
            ModelConfig(path=t5_path),
            ModelConfig(path=vae_path),
            ModelConfig(path=clip_path),
        ]
        tokenizer = ModelConfig(path=tokenizer_path)
        return configs, tokenizer

    def _load_wan_pipeline(self) -> WanVideoPipeline:
        model_configs, tokenizer_config = self._build_model_configs()
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.float16,
            device=self.device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        return pipe

    @staticmethod
    def _load_image_safe(path: Optional[str]) -> Optional[Image.Image]:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    @staticmethod
    def _pick_first_existing(paths: List[str]) -> Optional[str]:
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def _prepare_ip_adapter_condition(
        self, character_refs: List[str]
    ) -> Optional[Image.Image]:
        selected_path = self._pick_first_existing(character_refs)
        if selected_path is None:
            return None

        ip_model = None
        try:
            from diffusers import CLIPVisionModelWithProjection

            ip_model = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter", subfolder="models/image_encoder"
            )
            _ = IP_ADAPTER_SCALE
            return self._load_image_safe(selected_path)
        except Exception:
            return self._load_image_safe(selected_path)
        finally:
            if ip_model is not None:
                del ip_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _prepare_controlnet_condition(
        self, location_refs: List[str], num_frames: int
    ) -> Optional[List[Image.Image]]:
        selected_path = self._pick_first_existing(location_refs)
        if selected_path is None:
            return None

        controlnet_model = None
        try:
            from diffusers import ControlNetModel

            controlnet_model = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            )
            base = self._load_image_safe(selected_path)
            if base is None:
                return None
            return [base.copy() for _ in range(num_frames)]
        except Exception:
            base = self._load_image_safe(selected_path)
            if base is None:
                return None
            return [base.copy() for _ in range(num_frames)]
        finally:
            if controlnet_model is not None:
                del controlnet_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _collect_refs(
        scene: Dict, creative_document: Dict
    ) -> Tuple[List[str], List[str]]:
        char_map = {
            item.get("id"): item for item in creative_document.get("characters", [])
        }
        loc_map = {
            item.get("id"): item for item in creative_document.get("locations", [])
        }

        character_refs: List[str] = []
        for char_id in scene.get("character_ids", []):
            item = char_map.get(char_id, {})
            character_refs.extend(item.get("reference_image_paths", []))

        loc_item = loc_map.get(scene.get("location_id"), {})
        location_refs = list(loc_item.get("reference_image_paths", []))
        return character_refs, location_refs

    def generate_scene(self, scene: Dict, memory, creative_document: Dict) -> str:
        scene_id = str(scene.get("scene_id", "unknown"))
        clip_path = os.path.join(self.clips_dir, f"scene_{scene_id}.mp4")

        character_refs, location_refs = self._collect_refs(scene, creative_document)
        memory_state = memory.get_state_for_agent()
        prompt = self.prompt_enricher.enrich(scene, creative_document, memory_state)

        retry_count = 0
        while retry_count < RL_EPISODES_PER_SCENE:
            pipe = None
            try:
                first_frame = self._load_image_safe(memory.last_frame_path)
                ip_adapter_image = self._prepare_ip_adapter_condition(character_refs)
                controlnet_video = self._prepare_controlnet_condition(
                    location_refs, num_frames=DEFAULT_NUM_FRAMES
                )

                pipe = self._load_wan_pipeline()
                generated_frames = pipe(
                    prompt=prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    input_image=first_frame,
                    reference_image=ip_adapter_image,
                    control_video=controlnet_video,
                    num_frames=DEFAULT_NUM_FRAMES,
                    height=DEFAULT_HEIGHT,
                    width=DEFAULT_WIDTH,
                    seed=0,
                    tiled=True,
                )

                os.makedirs(os.path.dirname(clip_path), exist_ok=True)
                save_video(generated_frames, clip_path, fps=VIDEO_FPS, quality=5)

                critique = self.self_critic.evaluate(
                    clip_path=clip_path,
                    character_reference_images=character_refs,
                    location_reference_images=location_refs,
                    current_prompt=prompt,
                )

                if critique.retry and retry_count + 1 < RL_EPISODES_PER_SCENE:
                    prompt = critique.suggested_prompt
                    retry_count += 1
                    continue

                return clip_path
            except Exception as exc:
                if retry_count + 1 >= RL_EPISODES_PER_SCENE:
                    raise RuntimeError(
                        f"Scene generation failed for scene {scene_id}: {exc}"
                    ) from exc
                retry_count += 1
                prompt = (
                    prompt + " Ensure clean composition and strict identity continuity."
                )
            finally:
                if pipe is not None:
                    del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        raise RuntimeError(f"Scene generation exhausted retries for scene {scene_id}")
