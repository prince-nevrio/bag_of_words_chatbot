import random
import json

import torch

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

def userChat(userchat):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('Bag_word_chatbot\intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "Bag_word_chatbot\data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Neural"
    print("Let's chat! (type 'quit' to exit)")

    
    # sentence = "do you use credit cards?"
    sentence = userchat
    
        

    sentence = tokenize(sentence)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.90:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                return(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...For more information please contact us Email- id - preeti@nevrio.tech, deepak@nevrio.tech or call us - 9041959799.") 
        return (f"{bot_name}: I do not understand...For more information please contact us Email- id - preeti@nevrio.tech, deepak@nevrio.tech or call us - 9041959799.")  

userChat("hey")