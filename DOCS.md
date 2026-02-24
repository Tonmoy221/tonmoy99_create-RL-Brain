<!-- markdownlint-disable MD022 MD032 -->

# RL-Powered Cinematic Video Generation System (Wan2.2)

## 1) SYSTEM OVERVIEW

This project is a cinematic-only autonomous video generation pipeline built on Wan2.2.
A user provides one seed text prompt, then the system:

1. Plans full story structure (characters, locations, scenes).
2. Extracts character identity attributes (age, gender, wardrobe).
3. Extracts scenery attributes (type, material palette, visual description).
4. Splits the story into multiple scene prompts (10-12s target per clip).
5. Generates references and scene clips with consistency constraints.
6. Scores each scene and performs PPO policy updates.
7. Stitches clips into final video and writes a production report.

## 2) RL AGENT DESIGN

### State
- Seed prompt
- Character/location registries
- Continuity log
- Last frame path
- Recent reward history

### Action
- Scene prompt formulation and refinement for Wan2.2

### Reward (weighted)
- Visual quality
- Character consistency
- Location consistency
- Narrative coherence

### Learning loop
For each scene clip:
- Generate clip -> evaluate -> reward -> PPO update -> memory update

## 3) CONSISTENCY STRATEGY

### Character consistency
- Character profile includes age, gender, wardrobe, and visual description
- Reference images generated and reused across scenes
- Prompt constraints lock identity and wardrobe details

### Scenery consistency
- Location profile includes scenery type, material palette, and visual description
- Location references generated and reused across scenes
- Prompt constraints lock environment attributes

### Temporal continuity
- Last frame of previous scene is reused as initial context
- Continuity log tracks scene outcomes and guides next prompt

## 4) CORE FILES

- `inference.py`: main cinematic orchestration entrypoint
- `config.py`: centralized hyperparameters and path/model settings
- `agent.py`: `DirectorAgent`, `PromptEnricher`, `SelfCritic`
- `preproduction.py`: character/location reference generation
- `scene_pipeline.py`: Wan2.2 scene generation wrapper + retries
- `reward_model.py`: multi-component reward computation
- `memory.py`: persistent continuity state and resume support
- `stitcher.py`: final clip stitching and production report

## 5) HOW TO RUN

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (high VRAM recommended)

### Install
- `python3.10 -m venv .venv`
- `source .venv/bin/activate`
- `pip install --upgrade pip`
- `pip install -r requirements.txt`

### Required env for OpenAI mode
- `export OPENAI_API_KEY="your_key"`

### Optional Wan model paths
- `export WAN_MODEL_ROOT=./models`
- or set `WAN_DIFFUSION_MODEL`, `WAN_T5_PATH`, `WAN_VAE_PATH`, `WAN_CLIP_PATH`, `WAN_TOKENIZER_PATH`

### Run
- `python inference.py --prompt "Your cinematic story prompt" --output .`

### Run LLM-only pipeline (new, separate)
This is a separate pipeline that keeps the RL/WAN pipeline untouched. It uses an LLM agent to:
- Generate creative document
- Refine scene prompts
- Critique and revise prompts
- Save planning artifacts for downstream generation

Run:
- `python llm_pipeline.py --prompt "Your cinematic story prompt" --output .`

Optional:
- `python llm_pipeline.py --prompt "Your cinematic story prompt" --output . --provider openai --model gpt-4o`
- `python llm_pipeline.py --prompt "Your cinematic story prompt" --output . --resume`

Artifacts:
- `output/llm_only/scene_prompts_llm.json`
- `output/llm_only/llm_pipeline_report.json`
- `story_bible/llm_only/creative_document_llm.json`
- `memory_llm/state_llm.json`

### Kaggle smoke test (before WAN generation)
Use this to verify the agent/planner/PPO/memory loop is functioning before spending GPU time on WAN video generation.

- Install and setup as above
- Set API key if using OpenAI mode: `export OPENAI_API_KEY="your_key"`
- Run smoke test only:
  - `python kaggle_smoke_test.py --output-root /kaggle/working`
- If you only want PPO + memory checks without LLM planning:
  - `python kaggle_smoke_test.py --output-root /kaggle/working --skip-creative-doc`

Expected artifacts:
- `/kaggle/working/memory/state.json`
- `/kaggle/working/kaggle_smoke_test_report.json`

### Resume
If `memory/state.json` exists, the launcher asks whether to resume.

## 6) IMPORTANT CONFIGURATION

In `config.py`:

- `VIDEO_MODEL="wan2.2"`
- `RL_EPISODES_PER_SCENE`: retry/learning attempts per scene
- `MIN_SCENES`, `MAX_SCENES`: scene decomposition limits
- `SCENE_SECONDS_TARGET`: target seconds per generated scene clip
- Reward weights: visual/character/location/narrative
- `CLIP_CONSISTENCY_THRESHOLD`: retry trigger sensitivity

## 7) TROUBLESHOOTING

### OOM
- Lower frame count / resolution
- Lower retry count
- Use higher VRAM GPU

### Weak consistency
- Increase detail in seed prompt
- Ensure references are generated correctly
- Tighten continuity constraints in prompts

### LLM/API failures
- Verify API key and network
- Switch provider configuration if needed

### Model loading failures
- Check Wan2.2 model paths and files
- Verify dependency installation with `pip install -r requirements.txt`
