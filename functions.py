import ast
import openai
import re
import torch
import threading
from diffusers import StableDiffusionPipeline
import random
import math
from PIL import Image
from tqdm import tqdm
import uuid

tqdm.monitor_interval = 0

global_model_pipe = None

def load_model_async():
    global global_model_pipe
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32  # Default to float32 on CPU
    global_model_pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', torch_dtype=dtype).to(device)
   

# Start loading the model asynchronously at the start of the script or application
model_loading_thread = threading.Thread(target=load_model_async)
model_loading_thread.start()

def initialize_conversation():
    # Use a delimiter to split the prompt into parts
    delimiter = "####"

    # Describe the chatbot's role, a chain of thoughts, and few shot examples
    tags_prompt = f"""You are a digital marketing professional who is responsible for designing a marketing campaign for a product targeted for a specific target group.
    You need to interact with the user to extract information about the following things:
    1. Description of product that is being marketed
    2. Description of the target audience

    {delimiter}
    You will go through the following chain of thoughts to extract information and save it in the format given:
    Thought 1. Start with a message introducing yourself and saying that you will ask two questions to understand the product and the target audience. Ask the user to say something to continue the chat.
    Thought 2. If the user says yes, ask the user to describe the product.
    Thought 3. Ask the user about the target audience.
    Thought 4. Thank the user for providing the information and ask them to wait for their call to action and marketing image.
    {delimiter}

    {delimiter}
    Here is a sample conversation for you to learn from.
    {delimiter}

    Sample conversation 1:
    Assistant: Hello, I am a marketing professional, I can help you with the creation of a marketing campaign. I will need to ask you a few questions for that. Would you like to continue?
    User: Yes, continue.
    Assistant: Please provide a short description of the product that you would like to market.
    User: I would like to market a savings bank account.
    Assistant: Thank you for providing that information! Please provide a short description of the audience for which you would like to target the marketing campaign.
    User: I want to target this product to people of the age group 18-35 who live in tier-1 cities in India.
    Assistant: Thank you for providing this information. Please wait for your call to action text and marketing image.
    {delimiter}

    {delimiter}
    Sample conversation 2:

    Assistant: Hello, I am a marketing professional, I can help you with the creation of a marketing campaign. I will need to ask you a few questions for that. Would you like to continue?
    User: Yes, continue.
    Assistant: Please provide a short description of the product that you would like to market.
    User: I would like to market a home loan.
    Assistant: Thank you for providing that information! Please provide a short description of the audience for which you would like to target the marketing campaign.
    User: I want to target this product to people of the age group 35-50 who live in tier-2 cities in India.
    Assistant: Thank you for providing this information. Please wait for your call to action text and marketing image.
    {delimiter}

    Start with just the first message.
    """

    # Save the system message in the correct format
    chat_initializer = [{'role': 'system', 'content': tags_prompt}]

    return chat_initializer


def get_chat_model_completions(messages,token=3000):
    response = openai.chat.completions.create(
        model ="gpt-4-turbo",
        messages = messages,
        temperature = 0, # this is the degree of randomness of the model's output
        max_tokens = token
    )
    return response.choices[0].message.content

def moderation_check(user_input):
    response = openai.moderations.create(input=user_input)
    moderation_output = response.results[0]
    if moderation_output.flagged == True:
        return "Flagged"
    else:
        return "Not Flagged"

def get_response(prompt, conversation):
    user_response = {'role': 'user', 'content': prompt}
    conversation.append(user_response)
    response = get_chat_model_completions(conversation)
    conversation.append(response)
    return conversation
    
def intent_confirmation_layer(response_assistant):
    prompt = f"""
    You are a seasoned marketing proffessional. Your job is to review a given conversation and find out if input has product description and the target audience that a customer wants to aim.
    Output a string 'Yes' if the input contains the product description and the target audience. Value of target audience can't be null or empty, in such case return 'No'
    Otherwise out the string 'No'. Value of both the keys is important, so cross check the input if both keys have crossponding values before providing reponse.
    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No, cross check for values for both keys. There should be target audience and product description. If any of these is not present return 'No'
    """

    messages = [{"role": "system", "content": prompt}];
    confirmation = get_chat_model_completions(messages,20)
    return confirmation




def dictionary_present(response):
    delimiter = "####"
    prompt = f"""You are a seasoned marketing proffessional. Your job is to review a given conversation and extract information about the product description and the target audience that a customer wants to aim.
            Return that information in the JSON format with the keys "product_description" and "target_audience". Please don't provide any text beside this information.
            {delimiter}

            Here is the input {response}

            """
    system_response = [{'role': 'system', 'content': prompt}]
    response = get_chat_model_completions(system_response, 1000)
    return response


def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"
    dictionary_matches = re.findall(regex_pattern, string)
    dictionary = {}

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        product_description = ast.literal_eval(dictionary_string)['product_description']
        target_audience = ast.literal_eval(dictionary_string)['target_audience']
    return product_description, target_audience

def generate_tags(target_audience):
    '''Returns a list of tags relevant to the target audience'''

    # Define a format string to illustrate the expected tag format
    format = "['tag1', 'tag2', 'tag3', ...]"

    # Define a delimiter for formatting purposes
    delimiter = '####'

    # Create a prompt for the chatbot to provide tags for a specific target audience
    tags_prompt = f"""Your task is to provide 10 tags for people from the target group described in {target_audience}

    {delimiter}
    Provide your response in the following format: {format}"""

    # Use a chat model to get completions for the tag prompt
    tag_string = get_chat_model_completions([{'role': 'system', 'content': tags_prompt}])

    # Convert the response from the chat model into a Python list using ast.literal_eval
    # The response content is assumed to be a string representation of a list
    tags = ast.literal_eval(tag_string)

    # Return the extracted tags
    return tags

def generate_descriptions(product, tags):
    '''Takes in information about the product and tags about the target audience
    and returns a call to action and an image description that can later be used as a prompt.'''

    # Provide a format for the output
    format = '{"cta": "call to action", "image": "image description"}'

    delimiter = "####"

    # Provide a template for our prompt and few-shot examples
    descriptions_prompt = f"""You are an expert Marketing Manager.
    Develop a marketing campaign for {product}, targeting the specified audience {tags}. The task includes crafting:

    A call-to-action (CTA) phrase, concise yet compelling, under 200 characters, to motivate the audience to engage with {product}.
    An image description for AI-generated visual content, adhering to the following criteria: no depiction of humans, faces, hands, or text, and succinctly encapsulate the product's appeal in under 100 words.
    Format your response as specified in {format}, ensuring both elements are concise and effectively communicate the campaign
    """

    # Get a response from the ChatCompletion API
    marketing_descriptions = get_chat_model_completions([{'role': 'system', 'content': descriptions_prompt}])

    # Convert the response into a dictionary
    descriptions = ast.literal_eval(marketing_descriptions)
    return descriptions

def gen_sd_img(prompt, negative_prompt, seed = None):
    global global_model_pipe
    # Ensure the model is loaded before proceeding
    model_loading_thread.join()
    rnum = math.floor(random.random() * 1_000_000_000)
    fseed = rnum if seed is None else seed
    generator = torch.Generator('cuda').manual_seed(fseed)

    image = global_model_pipe(prompt = prompt,
                 negative_prompt = negative_prompt,
                 num_inference_steps = 20,
                 guidance_scale = 9,
                 height = 768,
                 width = 768,
                 num_images_per_prompt = 1,
                 generator = generator).images[0]
    return image


def generate_image(response):
    product_description, target_audience = extract_dictionary_from_string(response)
    tags = generate_tags(target_audience)
    descriptions = generate_descriptions(product_description, tags)
    gen_image = gen_sd_img(prompt = descriptions['image'], negative_prompt = "text, face, ugly, low-quality, hand", seed = 23)
    return gen_image, descriptions['cta']

def is_valid_image(image_object):
    # Check if the object is an instance of PIL.Image.Image
    if isinstance(image_object, Image.Image):
        try:
            # Attempt to access a basic attribute of the image
            image_object.verify()  # This will check for some errors in the image
            return True
        except Exception as e:
            return False
    else:
        return False
    

def save_image(image):
    unique_id = uuid.uuid4()
    image_path = f"static/gen_image/{unique_id}.jpg"
    image.save(image_path)
    return image_path