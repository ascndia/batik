from huggingface_hub import HfApi

# --- Configuration ---
YOUR_LOCAL_DATA_PATH = "/opt/data/batik-sd-vae/" # The local folder with images/ and vae-sd/
YOUR_HF_USERNAME = "ruwwww"
YOUR_HF_DATASET_NAME = "batik-processed"
# ---------------------

api = HfApi()
repo_id = f"{YOUR_HF_USERNAME}/{YOUR_HF_DATASET_NAME}"

print(f"Uploading folder {YOUR_LOCAL_DATA_PATH} to {repo_id}...")

# This will upload every file and folder inside YOUR_LOCAL_DATA_PATH
# to the root of your dataset repository.
api.upload_folder(
    folder_path=YOUR_LOCAL_DATA_PATH,
    repo_id=repo_id,
    repo_type="dataset",
    # You can ignore .git, .DS_Store, etc.
    # Add any other files/folders you want to skip.
    ignore_patterns=["*.git*", "*.DS_Store*"], 
)

print(f"Upload complete! View your dataset at: https://huggingface.co/datasets/{repo_id}")