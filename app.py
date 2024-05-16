from flask import Flask, redirect, url_for, render_template, request
from functions import initialize_conversation, get_chat_model_completions, moderation_check,intent_confirmation_layer,dictionary_present,generate_image,is_valid_image,save_image
from datetime import datetime


import openai
import ast
import re
import pandas as pd
import json

openai.api_key = open("api_key.txt", "r").read().strip()


app = Flask(__name__)
app.secret_key = '12345'


camp_req = None
@app.route("/")
def default_func():
    global conversation_bot, conversation, camp_req,conversation_reco
    conversation_bot = []
    conversation = initialize_conversation()
    introduction = get_chat_model_completions(conversation)
    conversation_bot.append({'bot':introduction})
    conversation.append({"role": "assistant", "content": introduction})
    now = datetime.now()
    user_time = now.strftime("%H:%M")
    return render_template("chat.html", name_xyz = conversation_bot, user_time = user_time)

@app.route("/end_conv", methods = ['POST'])
def end_conv():
    global conversation_bot, conversation, camp_req, conversation_reco
    conversation_bot = []
    conversation=[]
    conversation_reco=[]
    camp_req = None
    return redirect(url_for('default_func'))

@app.route("/getresponse", methods = ['POST'])
def getresponse():
    global conversation_bot, conversation, camp_req, conversation_reco
    conversation_bot = []
    user_input = request.form["user_input_message"]
    prompt = 'Remember your system message and that you are an experienced digital marketing professional. So, you only help with questions around digital marketing campaign.'
    moderation = moderation_check(user_input)
    if moderation == 'Flagged':
        return redirect(url_for('end_conv'))

    if camp_req is None:
        conversation.append({"role": "user", "content": user_input + prompt})
        response_assistant = get_chat_model_completions(conversation)
        conversation.append({"role": "assistant", "content": response_assistant})
        moderation = moderation_check(response_assistant)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        confirmation = intent_confirmation_layer(conversation)
        moderation = moderation_check(confirmation)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        if "No" in confirmation:
            conversation_bot.append({'bot':response_assistant})
        else:
            response = dictionary_present(conversation)
            moderation = moderation_check(response)
            if moderation == 'Flagged':
                return redirect(url_for('end_conv'))

            conversation_bot.append({'bot':"Thank you for providing all the information. Kindly wait, while I fetch the products: \n"})
            camp_req, cta = generate_image(response)
            gen_image = is_valid_image(camp_req)
            if not gen_image:
                camp_req = None    
                conversation_bot.append({'bot':"Sorry, We failed to generate a valid image., But here is a catch CTA for your Campaign:"+ cta})
            else:
                image_path = save_image(camp_req)  # Assume this function saves the image and returns the path
                conversation_bot.append({'bot': "Here is your campaign image: " + image_path + " and CTA: " + cta})
                conversation_reco = response

    else:
        camp_req, cta = generate_image(conversation_reco)
        gen_image = is_valid_image(camp_req)
        if not gen_image:
            camp_req = None    
            conversation_bot.append({'bot':"Sorry, We failed to generate a valid image., But here is a catch CTA for your Campaign:"+ cta +". Please Retry"})
        else:
            image_path = save_image(camp_req)  # Assume this function saves the image and returns the path
            conversation_bot.append({'bot': "Here is your campaign image: " + image_path + " and CTA: " + cta})
    return conversation_bot




if __name__ == '__main__':
    app.run(debug=True)