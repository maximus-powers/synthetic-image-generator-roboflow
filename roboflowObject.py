from roboflow import Roboflow
import glob
import os
import shutil

class RoboflowObject():
    def __init__(self, robo_api_key, robo_workspace_id, robo_project_id, images_dir):
        self.rf = Roboflow(api_key=robo_api_key)
        self.project = self.rf.workspace(workspace_id=robo_workspace_id).project(project_id=robo_project_id)
        self.images_dir = images_dir

    def upload_images_to_roboflow(self):
        # iterate over each category directory in the images directory
        subject_dirs = glob.glob(os.path.join(self.images_dir, '*'))

        for subject_dir in subject_dirs:
            if os.path.isdir(subject_dir):
                subject = os.path.basename(subject_dir)
                image_paths = glob.glob(os.path.join(subject, '*'))  # adjusted to get all files in the category folder
                for image in image_paths:
                    self.project.upload(image, tag=subject, num_retry_uploads=3)
                    print(f'Uploading image {image} of category {subject}')
                
                # delete the folder after uploading its contents
                shutil.rmtree(subject_dir)
                print(f'Deleted folder {subject_dir}')

        print('Finished uploading images to Roboflow')
        shutil.rmtree(self.images_dir) # delete the images folder after uploading all images
        print(f'Deleted folder {self.images_dir}')