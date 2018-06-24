from flask import Flask, flash, redirect, render_template, request, session, abort
from rasa_nlu.model import Metadata, Interpreter
from flask import jsonify
app = Flask(__name__)
from flask import request
from weather import Weather, Unit
weather = Weather(unit=Unit.CELSIUS)
from flask_cors import CORS
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from pymongo import MongoClient
import pprint
import datetime
import json
CORS(app)
@app.route("/")
def index():
    return "Flask App!"

@app.route("/api/<string:projid>/<string:query>/")
def responseGenerator(projid,query):
    db = getMongoDBConnection('chatbot_nlu_training')
    collection = db['training_data_'+projid]
    #interpreter = Interpreter.load("F://chatbots/"+projid+"/models/nlu/default/current")
    interpreter = Interpreter.load("./chatbots/"+projid+"/models/nlu/default/current")
    response = interpreter.parse(query)
    # pprint.pprint(response)
    for data in collection.find({'intent' : response['intent']['name']}):
        pprint.pprint({'rasa_resp' : response })
        return jsonify({'rasa_resp' : response})

@app.route("/api", methods=['POST'])
def responseGeneratorWithLogging():
    parameters = request.get_json()
    
    pprint.pprint("Request recived with parameters")
    pprint.pprint(parameters)

    pid = parameters['pid']
    query = parameters['q']

    db = getMongoDBConnection('chatbot_nlu_training')
    collection = db['training_data_'+pid]
    #interpreter = Interpreter.load("F://chatbots/"+pid+"/models/nlu/default/current")
    interpreter = Interpreter.load("./chatbots/"+pid+"/models/nlu/default/current")
    response = interpreter.parse(query)
    pprint.pprint(response)
    #if bot fails the log the failure
    if 'intent' not in response or 'name' not in response['intent'] or response['intent']['confidence'] < .01 :
        logFailure(query,response,'hvfhjdv',pid)
    pprint.pprint(response)
    responseFromMongo = collection.find({'intent' : response['intent']['name']})
    fianlResponse = {}
    for data in responseFromMongo:
        fianlResponse['payload'] = response
        fianlResponse['res'] = data['response']
        pprint.pprint(fianlResponse)
    
    logchat(query,fianlResponse['res'],'hvfhjdv',pid)
    return jsonify(fianlResponse)

def logchat(usermsg,botmsg,sid,pid):
    db = getMongoDBConnection('chatbot_nlu_training')
    collection = db['chatlog_'+pid]
    data = {'$push':{'chat' : {'u_m':usermsg,'b_m':botmsg,'tm' : datetime.datetime.now()}}}
    where = {'pid' : pid,'sid' : sid}
    result = collection.update(where,data,upsert = True)
    pprint.pprint(result)

def logFailure(usermsg,botmsg,sid,pid):
    db = getMongoDBConnection('chatbot_nlu_training')
    collection = db['failure_'+pid]
    data = {'$push':{'chat' : {'u_m':usermsg,'b_m':botmsg,'tm' : datetime.datetime.now()}}}
    where = {'pid' : pid,'sid' : sid}
    result = collection.update(where,data,upsert = True)
    pprint.pprint(result)

@app.route("/train/<string:projid>/")
def train(projid):
    return trainbot(projid)
@app.route("/w")
def w():
    location = weather.lookup_by_location('gurgaon')
    condition = location.condition
    return jsonify(condition.text)
def trainbot(projid):
    db = getMongoDBConnection('chatbot_nlu_training')
    t_data = {'rasa_nlu_data': {'regex_features':[],'entity_synonyms':[],'common_examples' : []}}
    collection = db['training_data_'+projid]
    for intent in collection.find():
        for text in intent['text']:
            # data = t_data['rasa_nlu_data']
            t_data['rasa_nlu_data']['common_examples'].append({'intent' : intent['intent'],'text':text['value'],'entities': text['entities']})

    pprint.pprint(t_data)
    #f= open("F://chatbots/"+projid+"/training_data.json","w+")
    f= open("./chatbots/"+projid+"/training_data.json","w+")
    f.write(json.dumps(t_data))
    f.close()

    #t1_data = load_data("F://chatbots/"+projid+"/training_data.json")
    t1_data = load_data("./chatbots/"+projid+"/training_data.json")
    trainer = Trainer(config.load("./chatbots/config.yaml"))
    # trainer = Trainer(RasaNLUConfig("F://chatbots/"+projid+"/config.yaml"))
    trainer.train(t1_data)
    #trainer.persist('F://chatbots/'+projid+'/models/nlu/', fixed_model_name="current")
    trainer.persist('./chatbots/'+projid+'/models/nlu/', fixed_model_name="current")
    return jsonify({'status':'sucess'})

    
def getMongoDBConnection(db):
    client = MongoClient('localhost', 27017)
    db = client[db]
    return db

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4203,threaded=True)
