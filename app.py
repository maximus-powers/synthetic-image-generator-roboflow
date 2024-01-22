from fastapi import FastAPI, HTTPException
from generate import generate_entire_dataset  # Import your main function
from diffusers import StableDiffusionPipeline
import os
import torch

app = FastAPI()

# model path stuff
model_name = 'runwayml/stable-diffusion-v1-5'
access_token = os.getenv('huggingface_access_token')
target_dir = './stable-diffusion-v1-5_pretrained-model'

def download_model(model_name, token, target_dir):
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name, token=token, torch_dtype=torch.float16, revision="fp16"
    )
    pipeline.save_pretrained(target_dir)

def model_exists(directory):
    return os.path.exists(directory) and os.listdir(directory)

if not model_exists(target_dir):
    print("Downloading model...")
    download_model(model_name, access_token, target_dir)
else:
    print("Model already downloaded.")

@app.post('/generate-images')
async def generate_images(data: dict):
    roboflow_api_key = data['roboflow_api_key']
    roboflow_project_id = data['roboflow_project_id']
    roboflow_workspace_id = data['roboflow_workspace_id']
    openai_api_key = data['openai_api_key']
    image_subjects_list = data['image_subjects_list']
    total_dataset_size = data.get('total_dataset_size', 100)
    guidance_scale = data.get('guidance_scale', 8)

    response = generate_entire_dataset(
        target_dir, roboflow_api_key, roboflow_workspace_id, 
        roboflow_project_id, openai_api_key, image_subjects_list, 
        total_dataset_size, guidance_scale
    )
    
    return response
