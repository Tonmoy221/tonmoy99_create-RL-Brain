<!-- markdownlint-disable MD022 MD032 MD031 MD040 -->

# RL-Powered Cinematic Video Generation System

## 1) SYSTEM OVERVIEW

This system generates multi-scene cinematic videos from a seed prompt using Wan2.1 as the video engine and an RL-driven director loop to improve scene quality and continuity over time.

### Pipeline (Step-by-step)
1. User runs `inference.py` with `--mode cinematic` and a seed prompt.
2. System checks VRAM, creates folders, and loads/initializes `ConsistencyMemory`.
3. `DirectorAgent` creates a Creative Document JSON (characters, locations, scenes, style, arc).
4. `ReferenceGenerator` produces character/location reference images.
5. For each scene:
   - `PromptEnricher` builds a Wan2.1-optimized prompt.
   - `SceneGenerator` calls existing Wan2.1 pipeline wrapper to generate clip.
   - `SelfCritic` evaluates CLIP consistency and triggers retries if needed.
   - `RewardModel` computes composite reward.
   - `DirectorAgent` performs PPO policy update.
   - `ConsistencyMemory` updates continuity state and persists to disk.
6. `VideoStitcher` concatenates clips with 0.5s crossfade.
7. Production report is written with reward curve and asset paths.

### ASCII Data Flow
```
Seed Prompt
   |
   v
DirectorAgent (LLM + PPO)
   |
   v
Creative Document (JSON) -----> ReferenceGenerator -----> Story Bible Refs
   |                                                      |
   |                                                      v
   +------------> SceneGenerator (Wan2.1 Wrapper + Conditioning)
                              |
                              v
                        Scene Clip (.mp4)
                              |
                              +--> SelfCritic (CLIP)
                              +--> RewardModel (ImageReward + CLIP + LLM)
                              |
                              v
                     PPO Update + Memory Update
                              |
                              v
                    VideoStitcher + Production Report
                              |
                              v
                        Final Video (.mp4)
```

---

## 2) HOW THE RL AGENT WORKS

### RL State
The policy state combines:
- User seed prompt
- Creative document context
- Continuity log from completed scenes
- Character/location registries
- Last frame path from previous clip
- Recent reward history

### RL Action
The action is the prompt decision for each scene:
- Prompt structure
- Visual detail emphasis
- Continuity constraints
- Cinematic style injection

### Reward Components
Final reward:
- Visual quality (`0.25`): ImageReward over sampled frames
- Character consistency (`0.35`): CLIP similarity between frames and character refs
- Location consistency (`0.20`): CLIP similarity between frames and location refs
- Narrative coherence (`0.20`): LLM continuity scoring against memory log

These weights favor identity continuity first (character highest), then scene-level quality and story consistency.

### PPO Policy Update
- A lightweight local value-head LM is used for PPO updates via `trl`.
- Scene state text is query; scene prompt text is response; scalar composite reward is reward signal.
- PPO update happens once per generated scene attempt.

### Improvement Across Scenes
- Reward curve is tracked in memory.
- Low-consistency outputs trigger retries with stricter prompt constraints.
- Continuity log and reference assets increasingly constrain downstream scene generation.

---

## 3) HOW CONSISTENCY IS MAINTAINED

### Reference Image Generation
- Character refs: front portrait, 3/4 portrait, full-body.
- Location refs: wide establishing shot, detail/texture shot.
- Saved under `story_bible/references/...`.

### Character Consistency (IP-Adapter Strategy)
- Scene pipeline prepares a character conditioning image from refs.
- IP-Adapter scale target is `0.7`.
- Same identity descriptors are reinforced in enriched prompts.

### Location Consistency (ControlNet Strategy)
- Scene pipeline prepares location conditioning frames from reference image(s).
- Control guidance is fed as conditioning video for Wan wrapper path.

### Continuity Log
- After each scene, memory appends summary + reward + clip path.
- Last frame is extracted and reused as first-frame context for next scene.
- Narrative coherence score uses continuity log to detect contradictions.

---

## 4) FILE STRUCTURE

### Top-level runtime and orchestration
- `inference.py`: Main launcher with `simple` and `cinematic` modes and full pipeline orchestration.
- `config.py`: Centralized runtime, RL, reward, path, and generation configuration constants.
- `agent.py`: `DirectorAgent`, `PromptEnricher`, `SelfCritic` implementations.
- `reward_model.py`: Composite reward computation for each scene clip.
- `memory.py`: JSON + `.npy` persisted continuity and episode state.
- `preproduction.py`: Character/location reference generation.
- `scene_pipeline.py`: Scene-level generation wrapper over existing Wan2.1 pipeline.
- `stitcher.py`: Clip stitching and production report generation.
- `requirements.txt`: Pinned dependencies for environment setup.
- `DOCS.md`: This documentation file.

### Existing model and pipeline internals
- `wan_video_new.py`: Core Wan2.1 pipeline and unit execution graph.
- `save_video.py`: Video/image save and ffmpeg merge utilities.
- `lora.py`: Root-level LoRA helper utilities.
- `xdit_context_parallel.py`: Context-parallel utility for distributed transformer blocks.
- `make_spatialvid_metadata.py`: Spatial metadata generation utility.
- `train.py`: Training launch script.

### Loader
- `loader/config.py`: External model path/config loading helper.

### Models
- `models/downloader.py`: Model download helper logic.
- `models/longcat_video_dit.py`: LongCat video transformer implementation.
- `models/lora.py`: LoRA merge/load helpers for model components.
- `models/model_config.py`: Model type detection and config registry.
- `models/model_manager.py`: Model loading, hashing, fetch, and management.
- `models/utils.py`: Shared model helper functions.
- `models/wan_video_animate_adapter.py`: Animate adapter model block.
- `models/wan_video_camera_controller.py`: Camera control embeddings/utilities.
- `models/wan_video_dit_s2v.py`: Speech-to-video related DiT utilities.
- `models/wan_video_dit.py`: Core Wan diffusion transformer model.
- `models/wan_video_image_encoder.py`: Image encoder module.
- `models/wan_video_mot.py`: Motion-oriented transformer component.
- `models/wan_video_motion_controller.py`: Motion controller module.
- `models/wan_video_text_encoder.py`: Text encoder module.
- `models/wan_video_vace.py`: VACE-related model component.
- `models/wan_video_vae.py`: VAE encode/decode model.
- `models/wav2vec.py`: Audio processing model helpers.

### Prompters
- `prompters/base_prompter.py`: Base prompt processing abstractions.
- `prompters/wan_prompter.py`: Wan-specific prompt tokenization and embedding.

### Scheduler
- `schedulers/flow_match.py`: Flow-match scheduler utilities.

### Trainers
- `trainers/unified_dataset.py`: Unified dataset definitions.
- `trainers/utils.py`: Training utility functions and CLI parser.

### Utilities
- `utils/__init__.py`: Shared base classes/utilities (`BasePipeline`, units, converters).
- `utils/data/__init__.py`: Video/image IO and processing helper functions.

### VRAM management
- `vram_management/gradient_checkpointing.py`: Gradient checkpointing helpers.
- `vram_management/layers.py`: Memory-efficient layer implementations.

### Shell helper
- `Wan2.1-Fun-V1.1-1.3B-InP.sh`: Shell script for Wan model-related setup/run commands.

---

## 5) HOW TO RUN ON RUNPOD

### Recommended GPU
- Primary: `A100 80GB` (best for cinematic mode with multiple heavy models).
- Secondary: `RTX 4090` (works with stricter generation settings and retries).

### Step-by-step setup
1. Create a RunPod with CUDA 12.1-capable image and Python 3.10.
2. Clone/upload this project.
3. Create environment and install deps:
   - `python3.10 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`
4. Set LLM credentials (if OpenAI):
   - `export OPENAI_API_KEY="your_key"`
5. Set Wan model paths (if different from defaults):
   - `export WAN_MODEL_ROOT=./models`
   - or set individual `WAN_DIFFUSION_MODEL`, `WAN_T5_PATH`, `WAN_VAE_PATH`, `WAN_CLIP_PATH`, `WAN_TOKENIZER_PATH`

### Run simple mode
- `python inference.py --mode simple --prompt "..." --image test_image.png --output .`

### Run cinematic mode
- `python inference.py --mode cinematic --prompt "..." --output .`

### Resume interrupted run
- If `memory/state.json` exists, launcher asks to resume.
- Reply `y` to continue from remaining scenes.

### Read production report
- Open `output/production_report.json`.
- Inspect reward breakdown, retry counts, reward curve, and final asset paths.

---

## 6) CONFIGURATION REFERENCE

`config.py` variables:

### RL
- `RL_ALGORITHM="PPO"`: RL method.
- `RL_LEARNING_RATE=1e-5`: PPO optimizer LR.
- `RL_KL_PENALTY=0.1`: PPO KL pressure.
- `RL_GAMMA=0.99`: Discount factor.
- `RL_EPISODES_PER_SCENE=3`: Max retries per scene.
- `RL_REWARD_THRESHOLD=0.75`: Acceptance threshold.

### Reward Weights
- `REWARD_VISUAL_QUALITY=0.25`
- `REWARD_CHARACTER_CONSISTENCY=0.35`
- `REWARD_LOCATION_CONSISTENCY=0.20`
- `REWARD_NARRATIVE_COHERENCE=0.20`

### Consistency
- `CLIP_CONSISTENCY_THRESHOLD=0.65`: Retry trigger threshold.
- `IP_ADAPTER_SCALE=0.7`: Character conditioning scale.

### Model Settings
- `LLM_PROVIDER="openai"`: LLM backend (`openai`/`ollama`).
- `LLM_MODEL="gpt-4o"`: LLM model name.
- `REFERENCE_IMAGE_MODEL="flux"`: Ref image model (`flux`/`sdxl`).
- `VIDEO_MODEL="wan2.1"`: Video model family.

### Paths
- `OUTPUT_DIR="./output"`
- `STORY_BIBLE_DIR="./story_bible"`
- `MEMORY_DIR="./memory"`
- `CLIPS_DIR="./output/clips"`
- `REFERENCES_DIR="./story_bible/references"`

### Hardware
- `MIN_VRAM_GB=24`: VRAM warning floor.
- `TARGET_GPU="A100"`: Suggested target GPU.

### Generation
- `REFERENCE_IMAGES_PER_CHARACTER=3`
- `REFERENCE_IMAGES_PER_LOCATION=2`
- `VIDEO_FPS=24`
- `CROSSFADE_DURATION=0.5`
- `FRAMES_TO_SAMPLE_FOR_REWARD=5`
- `DEFAULT_NUM_FRAMES=150`
- `DEFAULT_HEIGHT=480`
- `DEFAULT_WIDTH=832`
- `DEFAULT_NEGATIVE_PROMPT=...`

### API Runtime
- `OPENAI_API_KEY_ENV="OPENAI_API_KEY"`
- `OLLAMA_BASE_URL="http://localhost:11434"`
- `OLLAMA_TIMEOUT_SECONDS=120`
- `OPENAI_TIMEOUT_SECONDS=120`

### PPO Backbone
- `PPO_POLICY_MODEL_NAME="sshleifer/tiny-gpt2"`
- `PPO_BATCH_SIZE=1`
- `PPO_MINI_BATCH_SIZE=1`
- `PPO_INIT_KL_COEF=0.1`
- `PPO_LEARNING_RATE=RL_LEARNING_RATE`

### Safety / Limits
- `MAX_SCENES=12`
- `MAX_CHARACTERS=8`
- `MAX_LOCATIONS=8`

### Report / persistence
- `PRODUCTION_REPORT_NAME="production_report.json"`
- `MEMORY_STATE_FILE="state.json"`
- `REWARD_BREAKDOWN_FILE="reward_breakdown.json"`

---

## 7) TROUBLESHOOTING

### OOM / CUDA out-of-memory
- Reduce `DEFAULT_NUM_FRAMES`.
- Reduce output resolution (`DEFAULT_HEIGHT`, `DEFAULT_WIDTH`).
- Lower scene retries (`RL_EPISODES_PER_SCENE`).
- Use A100 80GB for cinematic mode.

### CLIP consistency score too low
- Increase detail in character/location descriptions.
- Add stronger continuity constraints in seed prompt.
- Increase retries per scene.

### LLM not responding
- Verify `OPENAI_API_KEY` and network egress.
- Switch provider to Ollama in `config.py` and ensure Ollama server is running.

### Wan2.1 generation failure
- Check model path environment variables.
- Ensure all required model files exist under configured directories.
- Test `--mode simple` first to validate base pipeline.
- Verify `pip install -r requirements.txt` completed successfully.
