# come up with 10 descriptive image generation prompts with the subject of a "[object]", 
from openai import OpenAI
import os
import json

class LLMObject:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)

    def generate_prompts_for_subject(self, subject):
        json_template = '''
        {
            "<subject>": [<string prompt>, <string prompt>, <string prompt>, ...]
        }
        '''
        instructions = "Please create a list of 25 descriptive image generation prompts for the given subject. The prompts should depict the subject at all relevant angles, distances, and contexts. Return a JSON dictionary containing all the prompts, in of the form:"

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{instructions} {json_template}"},
                {"role": "user", "content": subject},
                {"role": "system", "content": "Tip: make sure the key name matches the given subject exactly, and that the prompts are in a list."}
            ],
            temperature=1,
            seed=1001
        )

        json_response = json.loads(response.choices[0].message.content)


        return json_response


