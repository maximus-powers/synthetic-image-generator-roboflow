# from huggingface_hub import login
from roboflowObject import RoboflowObject
from llmObject import LLMObject
from stableDiffusionObject import StableDiffusionObject
import math

def generate_entire_dataset(model_path, roboflow_api_key, roboflow_workspace_id, roboflow_project_id, openai_api_key, image_subjects_list, total_dataset_size, guidance_scale):
    ############################# CONFIG #############################
    model_path = 'stabledif_model' # path to the model, should be in the same directory as this file
    # init the objects
    robo_obj = RoboflowObject(roboflow_api_key, roboflow_workspace_id, roboflow_project_id, roboflow_project_id)
    sd_obj = StableDiffusionObject(model_path, output_dir=roboflow_project_id) # sets image folder on server as roboflow project name
    llm_obj = LLMObject(openai_api_key)


    ############################# PROMPT GENERATION #############################

    all_prompts = [] # init for a list of tuples, (subject, prompt_list)
    total_num_prompts = 1 # starts at 1 to avoid division by 0 error, used to calculate num images per prompt
    for subject in image_subjects_list:
        # generate prompts for the subject
        prompts_json = llm_obj.generate_prompts_for_subject(subject)
        total_num_prompts += len(prompts_json[subject])
        all_prompts.append(prompts_json)

    total_num_prompts -= 1 # subtract 1 to account for the initial +1

    num_images_per_prompt = math.ceil(total_dataset_size / total_num_prompts) # number of images per prompt rounded up, rather overdeliver


    ############################# IMAGE GENERATION #############################

    image_index = 0 # iterator for naming
    # generate images per prompt
    for subject, prompt_list in all_prompts:
        for prompt in prompt_list:
            # set the start and end index, could be done in the args but just incase someone doesn't get it
            start_index = image_index
            end_index = image_index + num_images_per_prompt

            # generates and stores the images locally. Directory structure is (project_name)/(subject)/(image_index).png
            sd_obj.generate_images_per_prompt(prompt, start_index, end_index, num_images_per_prompt, subject, guidance_scale)

            image_index += num_images_per_prompt # update the index
        

    ############################# UPLOAD TO ROBOFLOW #############################
        
    robo_obj.upload_images_to_roboflow() # uploads all images in the project folder to roboflow

    return {"status": "success", "message": "Images generated and uploaded"}
