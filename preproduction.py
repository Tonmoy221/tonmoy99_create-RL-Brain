import os
from typing import Dict, List, Optional

import torch
from PIL import Image

from config import (
    REFERENCE_IMAGE_MODEL,
    REFERENCE_IMAGES_PER_CHARACTER,
    REFERENCE_IMAGES_PER_LOCATION,
    REFERENCES_DIR,
)


class ReferenceGenerator:
    def __init__(self, output_root: str = ".", device: Optional[str] = None):
        self.output_root = output_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.references_root = os.path.join(self.output_root, REFERENCES_DIR)

    def _build_character_prompts(self, character: Dict) -> List[str]:
        desc = character.get("visual_description", "")
        return [
            f"Ultra-detailed cinematic studio portrait front-facing of {character.get('name', 'character')}. {desc}. sharp details, realistic skin texture",
            f"Ultra-detailed 3/4 angle cinematic portrait of {character.get('name', 'character')}. {desc}. dramatic soft lighting",
            f"Full-body cinematic fashion shot of {character.get('name', 'character')}. {desc}. clear clothing silhouette, realistic anatomy",
        ]

    def _build_location_prompts(self, location: Dict) -> List[str]:
        desc = location.get("visual_description", "")
        return [
            f"Cinematic wide establishing shot of location {location.get('name', 'location')}. {desc}. high detail environment",
            f"Cinematic texture and detail shot of location {location.get('name', 'location')}. {desc}. emphasis on materials and atmosphere",
        ]

    def _load_image_pipeline(self):
        from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline

        if REFERENCE_IMAGE_MODEL.lower() == "flux":
            model_id = "black-forest-labs/FLUX.1-dev"
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
        else:
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )

        pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        return pipe

    @staticmethod
    def _save_image(image: Image.Image, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

    def _generate_images(
        self, prompts: List[str], out_dir: str, prefix: str, expected_count: int
    ) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        generated_paths: List[str] = []
        pipe = None

        try:
            pipe = self._load_image_pipeline()
            for index, prompt in enumerate(prompts[:expected_count]):
                result = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=4.0)
                image = result.images[0]
                out_path = os.path.join(out_dir, f"{prefix}_{index + 1}.png")
                self._save_image(image, out_path)
                generated_paths.append(out_path)
        except Exception:
            fallback_path = os.path.join(out_dir, f"{prefix}_fallback.png")
            Image.new("RGB", (768, 768), color=(40, 40, 40)).save(fallback_path)
            generated_paths.append(fallback_path)
        finally:
            if pipe is not None:
                del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return generated_paths

    def generate_character_references(self, character: Dict) -> List[str]:
        character_id = character.get("id", "unknown_character")
        out_dir = os.path.join(self.references_root, "characters", str(character_id))
        prompts = self._build_character_prompts(character)
        return self._generate_images(
            prompts=prompts,
            out_dir=out_dir,
            prefix="character",
            expected_count=REFERENCE_IMAGES_PER_CHARACTER,
        )

    def generate_location_references(self, location: Dict) -> List[str]:
        location_id = location.get("id", "unknown_location")
        out_dir = os.path.join(self.references_root, "locations", str(location_id))
        prompts = self._build_location_prompts(location)
        return self._generate_images(
            prompts=prompts,
            out_dir=out_dir,
            prefix="location",
            expected_count=REFERENCE_IMAGES_PER_LOCATION,
        )

    def generate_all(self, creative_document: Dict) -> Dict:
        characters = creative_document.get("characters", [])
        locations = creative_document.get("locations", [])

        for character in characters:
            refs = self.generate_character_references(character)
            character["reference_image_paths"] = refs

        for location in locations:
            refs = self.generate_location_references(location)
            location["reference_image_paths"] = refs

        return creative_document
