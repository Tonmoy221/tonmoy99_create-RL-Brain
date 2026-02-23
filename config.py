# RL Settings
RL_ALGORITHM = "PPO"  # Reinforcement learning algorithm used for policy updates
RL_LEARNING_RATE = 1e-5  # Learning rate for PPO optimizer
RL_KL_PENALTY = 0.1  # KL divergence penalty coefficient for PPO stability
RL_GAMMA = 0.99  # Discount factor for future rewards
RL_EPISODES_PER_SCENE = 3  # Maximum retries / RL episodes allowed per scene
RL_REWARD_THRESHOLD = 0.75  # Reward threshold to accept scene without further retries

# Reward Weights
REWARD_VISUAL_QUALITY = 0.25  # Weight for ImageReward visual quality score
REWARD_CHARACTER_CONSISTENCY = 0.35  # Weight for character consistency score
REWARD_LOCATION_CONSISTENCY = 0.20  # Weight for location consistency score
REWARD_NARRATIVE_COHERENCE = 0.20  # Weight for LLM-based narrative coherence score

# Consistency Thresholds
CLIP_CONSISTENCY_THRESHOLD = 0.65  # Minimum CLIP similarity to avoid retry
IP_ADAPTER_SCALE = 0.7  # Conditioning scale for IP-Adapter image guidance

# Model Settings
LLM_PROVIDER = "openai"  # "openai" or "ollama"
LLM_MODEL = "gpt-4o"  # LLM model name (e.g., "gpt-4o" or "llama3")
REFERENCE_IMAGE_MODEL = "flux"  # "flux" or "sdxl" for reference generation
VIDEO_MODEL = "wan2.1"  # Video model family used in this project

# Paths
OUTPUT_DIR = "./output"  # Root directory for generated outputs
STORY_BIBLE_DIR = "./story_bible"  # Directory for story bible artifacts
MEMORY_DIR = "./memory"  # Directory for persisted continuity and RL state
CLIPS_DIR = "./output/clips"  # Directory for per-scene clip files
REFERENCES_DIR = (
    "./story_bible/references"  # Directory for character/location references
)

# VRAM
MIN_VRAM_GB = 24  # Minimum recommended VRAM for cinematic mode
TARGET_GPU = "A100"  # Preferred target GPU class ("A100" or "RTX4090")

# Generation Settings
REFERENCE_IMAGES_PER_CHARACTER = 3  # Number of generated character reference images
REFERENCE_IMAGES_PER_LOCATION = 2  # Number of generated location reference images
VIDEO_FPS = 24  # Frames-per-second for final output video
CROSSFADE_DURATION = 0.5  # Transition duration (seconds) between scene clips
FRAMES_TO_SAMPLE_FOR_REWARD = 5  # Number of clip frames sampled for reward calculation

# Inference Defaults
DEFAULT_NUM_FRAMES = 150  # Default number of frames per generated scene
DEFAULT_HEIGHT = 480  # Default output height
DEFAULT_WIDTH = 832  # Default output width
DEFAULT_NEGATIVE_PROMPT = (
    "overly saturated, overexposed, static, blurry, subtitles, low quality, "
    "jpeg artifacts, deformed, extra limbs, cluttered background"
)  # Shared fallback negative prompt

# Runtime / API
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"  # Environment variable containing OpenAI API key
OLLAMA_BASE_URL = "http://localhost:11434"  # Local Ollama endpoint URL
OLLAMA_TIMEOUT_SECONDS = 120  # Timeout for Ollama generation requests
OPENAI_TIMEOUT_SECONDS = 120  # Timeout for OpenAI responses

# PPO Policy Backbone
PPO_POLICY_MODEL_NAME = (
    "sshleifer/tiny-gpt2"  # Lightweight local trainable policy for PPO updates
)
PPO_BATCH_SIZE = 1  # PPO batch size per update step
PPO_MINI_BATCH_SIZE = 1  # PPO mini-batch size
PPO_INIT_KL_COEF = 0.1  # PPO initial KL coefficient
PPO_LEARNING_RATE = RL_LEARNING_RATE  # PPO learning rate alias used by trainer setup

# Story / generation constraints
MAX_SCENES = 12  # Safety cap for number of scenes in a creative document
MAX_CHARACTERS = 8  # Safety cap for number of characters in a creative document
MAX_LOCATIONS = 8  # Safety cap for number of locations in a creative document

# Report & persistence
PRODUCTION_REPORT_NAME = "production_report.json"  # Production report file name
MEMORY_STATE_FILE = "state.json"  # Persistent memory state filename
REWARD_BREAKDOWN_FILE = "reward_breakdown.json"  # Optional reward debug filename
