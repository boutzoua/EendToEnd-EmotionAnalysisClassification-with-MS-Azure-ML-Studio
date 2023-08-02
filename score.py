import os
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text
from azureml.core.model import Model
from azureml.core import Workspace, Datastore, Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_TRT_DISABLE'] = '1'

def init():
    global model
    global data
    global max_len
    global token
    global sent_to_id
    
    ws = Workspace(workspace_name="TrainMLModel",
               subscription_id="591ab441-3f7d-4a10-8641-161749bf9f82",
               resource_group="MustaphaGroupResources",
               _location="francecentral")
    datastore = ws.get_default_datastore()
    data_path = [(datastore, 'data/aug_emotion_data.csv')]
    dataset = Dataset.Tabular.from_delimited_files(path=data_path)
    data = dataset.to_pandas_dataframe()
    # Load the model from the registered model path
    model_path = Model.get_model_path(_workspace=ws,model_name="model_LSTM_emotion_BiLSTM")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
   
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data.sentiment_id)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)

    X_train, X_test, y_train, y_test = train_test_split(data.clean_content,Y, random_state=1995, test_size=0.2, shuffle=True)

    token = text.Tokenizer(num_words=None)
    token.fit_on_texts(list(X_train) + list(X_test))
    max_len = 160
    sent_to_id  = {"worry":0,"happy":1,"sad":2, "love":3, "surprise":4}

# def get_sentiment(text):
#     # text = clean_text(text)
#     #tokenize
#     twt = token.texts_to_sequences([text])
#     twt = pad_sequences(twt, maxlen=max_len, dtype='int32')
#     sentiment = model.predict(twt,batch_size=1,verbose = 2)
#     sent = np.round(np.dot(sentiment,100).tolist(),0)[0]
#     result = pd.DataFrame([sent_to_id.keys(),sent]).T
#     result.columns = ["sentiment","percentage"]
#     result=result[result.percentage !=0]
#     result = result.sort_values('percentage', ascending=False)
#     return result
   

 

def run(raw_data):
    try:
        # Convert the input to a numpy array
        data = json.loads(raw_data)['text']
        # features = np.array(data)

        #Tokenize
        twt = token.texts_to_sequences([data])
        twt = pad_sequences(twt, maxlen=max_len, dtype='int32')
        sentiment = model.predict(twt,batch_size=1,verbose = 2)
        sent = np.round(np.dot(sentiment,100).tolist(),0)[0]
        result = pd.DataFrame([sent_to_id.keys(),sent]).T
        result.columns = ["sentiment","percentage"]
        result=result[result.percentage !=0]
        result = result.sort_values('percentage', ascending=False)
        
        
        # Perform inference using the loaded model
        # prediction = model.predict(features)
        #prediction = get_sentiment(data)

        prediction = result.to_json()
        # Return the prediction as a JSON object
        return json.dumps(prediction)
    
    except Exception as e:
        error = str(e)
        return json.dumps(error)