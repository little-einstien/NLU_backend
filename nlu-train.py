from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from pymongo import MongoClient
import pprint
import datetime
import json
client = MongoClient('localhost', 27017)
db = client.chatbot_nlu_training
t_data = {'rasa_nlu_data': {'regex_features':[{
        "name": "zipcode",
        "pattern": "[0-9]{5}"
      }],'entity_synonyms':[],'common_examples' : []}}
collection = db.training_data	
for intent in collection.find():
    for text in intent['text']:
        # data = t_data['rasa_nlu_data']
        t_data['rasa_nlu_data']['common_examples'].append({'intent' : intent['intent'],'text':text['value'],'entities': text['entities']})

pprint.pprint(t_data)	
f= open("t_data.json","w+")
f.write(json.dumps(t_data))
f.close()

t1_data = load_data('t_data.json')
trainer = Trainer(RasaNLUConfig("nlu_model_config.json"))
trainer.train(t1_data)
model_directory = trainer.persist('models/nlu/', fixed_model_name="current")
