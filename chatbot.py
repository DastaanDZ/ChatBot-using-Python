from matplotlib.pyplot import flag
import numpy as np
import nltk
import random
import string # to process standard python strings

f = open('chatbot.txt', 'r', errors = 'ignore')
raw = f.read()
raw = raw.lower() # converts to lowercase
nltk.download('punkt') # first time use only
nltk.download('wordnet') # first time use only
sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences (list of strings)
word_tokens = nltk.word_tokenize(raw) # converts to list of words (list of strings)

lemmer = nltk.stem.WordNetLemmatize()
#WordNet is a semantic graph of English included in NLTK
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREET_RESPONSES = ["'sup, bro", "hey, how are you?", "hi, I'm glad to see you"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def reponses(user_response):
    robot_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        robot_response = "I am sorry! I don't understand you"
        return robot_response
    else:
        robot_response = robot_response+sent_tokens[idx]
        return robot_response

flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("ROBO: You are welcome..")
        else:
            if(greet(user_response) != None):
                print("ROBO: "+greet(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens = word_tokens+nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("ROBO: ", end="")
                print(reponses(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! take care..")