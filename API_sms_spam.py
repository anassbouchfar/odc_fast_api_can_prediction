from crypt import methods
import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get('/',response_class=HTMLResponse)
def get_root():
    return """
        <html>
            <head>
                <title>Some HTML in here</title>
            </head>
            <body>
                <form action='/spam_detection_query' method='GET'>
  
                        <div className="form-group">
                            <label for="">Home Team</label>
                            <select name='message'  className="form-control" id="">
                            <option>Claim your prize now</option>
                            <option>Hello it's me</option>
                            <option>Algeria</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label for="">Away Team</label>
                            <select name='awayTeam'  class="form-control" id="">
                            <option>Morocco</option>
                            <option>Egypt</option>
                            <option>Algeria</option>
                            </select>
                        </div>
                        <button type="submit" class="btn btn-success">Predicter</button>
                        </form>
            </body>
        </html>"""

model = joblib.load('spam_classifier.joblib')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text
def classify_message(model, message):
    message = preprocessor(message)
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])
    return {'message':message,'label': label, 'spam_probability': spam_prob[0][1]}

@app.get('/spam_detection_query/')
async def detect_spam_query(message: str):
   return classify_message(model, message)

@app.get('/spam_detection_path/{message}')
async def detect_spam_path(message: int):
   return classify_message(model, message)