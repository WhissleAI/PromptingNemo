# from transformers import AutoModelForCausalLM, AutoTokenizer
from ctransformers import AutoModelForCausalLM,AutoConfig
import time
import json
import re
import sys

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", 
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
    model_type="mistral", 
    gpu_layers=50, 
    context_length=4096,
    max_new_tokens=4096
    )



    
start_time = time.time()

#text = """<s>[INST] A slot values start is marked by slot name NER_*, where * is NER type and value end is marked by END tag. If there is a token with EMOTION_$, then also give emotion label which is $ [/INST]
#restaurants of NER_DATE the present day END that approach nearest the old NER_NORP Bohemian END restaurants of pre-fyire days of the NER_NORP French END class are NER_PERSON Jacks END in in NER_FAC Sacramento Street END between NER_GPE Montgomery END and NER_PERSON Kearney END , Felix END in NER_FAC Montgomery Street END betweenEND between Cay END and NER_GPE Washington END and the poodleog Burges Franks END , in NER_FAC Bus StreetEND between NER_FAC Kearney END and NER_FAC Grant Avenue END . EMOTION_HAPPY </s>
#[INST] give me slot name and values for text above? if emotion token, then give emotion label [/INST]"""


text = """<s>[INST] Below is a conversation between caller and agent from Vangaurd Company. Vanguard Agent: marks start of an agent turn, while Caller: marks gives text of caller turn. [/INST]
Vanguard Agent: Good morning! Thank you for calling Vanguard. This is Sarah speaking. How can I assist you today?

Caller: Hi Sarah, I'm considering investing in Vanguard funds, particularly for my European trip. Could you tell me more about the services offered?

Vanguard Agent: Of course! Our funds cover a diverse range of investment goals. Are you looking for personalized advice or specific tools for your European plans?

Caller: I'm interested in personalized advice, especially concerning investments while on the trip. How does that work?

Vanguard Agent: Our personalized advice services cater to individual financial situations. As for your trip, we can explore investments suitable for that scenario. For instance, have you considered Paris or Rome for investment opportunities?

Caller: Hmm, I haven't thought about those locations. Could you provide more insights?

Vanguard Agent: Absolutely, Paris offers some promising investment avenues, especially in the real estate and tourism sectors. Rome, on the other hand, has notable opportunities in heritage preservation and luxury markets. These might align well with your travel plans.

Caller: That sounds intriguing. Do your retirement tools factor in international investments too?

Vanguard Agent: Yes, our retirement planning tools are versatile. They can help you map out and achieve your retirement goals, even if you're considering overseas investments.

Caller: And these market insights you mentioned—do they cover European markets as well?

Vanguard Agent: Definitely! Our platform provides insights into various markets, including Europe. This way, you're kept informed about market trends and updates across different regions.

Caller: That's great to know. How do I get started, particularly for investments related to the trip?

Vanguard Agent: Let's begin by understanding your investment goals and risk tolerance for this European venture. I can guide you through setting up an account and assist in selecting suitable investment options tailored for the trip.

Caller: That sounds comprehensive. Thank you, Sarah, for the information. I'll definitely consider Vanguard for my European investment plans.

Vanguard Agent: You're welcome! If you have more questions or need further assistance, feel free to reach out. We're here to help you make the most of your financial future.</s>
[INST] give me a summary of the call in less than 200 words. [/INST]"""


reply = llm(text, temperature=0.01)
print("Answer", reply)
print("Time taken: ", time.time() - start_time)
    

# device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# model.to(device)

# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])

