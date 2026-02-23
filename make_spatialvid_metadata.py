import os
import json
import pandas as pd
from pathlib import Path

# --- INPUT (read-only) ---
INPUT_DATASET_PATH = "/kaggle/input/group1-first50/SmallDataset"
# --- OUTPUT (writable) ---
OUTPUT_CSV_PATH = "/kaggle/working/SpatialVID_HQ_metadata.csv"

videos_dir = Path(INPUT_DATASET_PATH) / "videos"
rows = []

print("Scanning videos and loading captions...")
for video_path in videos_dir.rglob("*.mp4"):
    rel_path = video_path.relative_to(INPUT_DATASET_PATH)
    
    # Get group and uuid
    group = video_path.parent.name      # group_0001, group_0002, etc.
    uuid = video_path.stem               # video UUID
    
    caption_file = Path(INPUT_DATASET_PATH) / "annotations" / group / uuid / "caption.json"
    
    if caption_file.exists():
        try:
            with open(caption_file, "r", encoding="utf-8") as f:
                caption_data = json.load(f)
            
            # Extract scene description (prioritize SceneSummary, fallback to SceneDescription)
            scene_desc = caption_data.get("SceneSummary", caption_data.get("SceneDescription", ""))
            camera_motion = caption_data.get("CameraMotion", "")
            
            # Build prompt similar to the dataset loader
            if camera_motion:
                prompt = f"{scene_desc} Camera: {camera_motion}"
            else:
                prompt = scene_desc
            
            # Use as-is without enhancement
            prompt = prompt.strip() if prompt else "A detailed 3D object rotating on a turntable"
                
        except Exception as e:
            print(f"Warning: Could not read {caption_file}: {e}")
            prompt = "A detailed 3D object rotating on a turntable"
    else:
        print(f"Warning: Caption not found at {caption_file}")
        prompt = "A detailed 3D object rotating on a turntable"
    
    rows.append({
        "video": str(rel_path),
        "prompt": prompt
    })

# Save CSV to writable directory
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n✓ Success! Created metadata with {len(df)} videos")
print(f"→ Saved to: {OUTPUT_CSV_PATH}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSample prompts:")
for i in range(min(3, len(df))):
    print(f"\n{i+1}. {df.iloc[i]['video']}")
    print(f"   {df.iloc[i]['prompt'][:150]}...")