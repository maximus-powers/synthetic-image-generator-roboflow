import os
import torch
from diffusers import StableDiffusionPipeline

class StableDiffusionObject:
    # make sure to pass in access token for huggingface
    def __init__(self, model_path, output_dir='images'): # do we even need to pass in a folder name?
        # config stable diffusion pipeline, pass pretrained model if you want
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, revision="fp16"
        ).to("cuda") # won't work on most laptops, containerize and send to sever w GPU

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)



    def generate_images_per_prompt(self, prompt, start_index, end_index, num_images, subject, guidance_scale=8):
        print(f'Generating {num_images} images of {prompt}')

        # create the subject directory if it doesn't exist
        subject_dir = os.path.join(self.output_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)

        # run the generation
        images = self.pipeline(prompt, num_images_per_prompt=num_images, guidance_scale=guidance_scale)

        # save each image to their output folder (each object has a folder)
        for i in range(start_index, end_index):
            image_name = f"{subject_dir}/{i}.png"
            images.images[i-start_index].save(image_name)  # Adjusted index for images list
            print(f'Saved image {subject} {image_name}')

        # display images would go here but I'm gonna run it on a server with no display stream

