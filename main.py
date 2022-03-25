from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

import numpy as np
import pandas as pd
import joblib
import pickle

meta_model = joblib.load('model_rf__ODC_football_outcome_predictor_20220325__meta.model')
    
model = meta_model['model']
cols = meta_model['columns']

df = pd.DataFrame(np.zeros((1,len(cols)), dtype=int), columns=cols)

ranking = pd.read_csv(meta_model['ranking'])
    
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/welcome", response_class=HTMLResponse)
def welcome_page():
    return """
    <html>
            <head>
                <title>Some HTML in here</title>
                <style>
                * {
  box-sizing: border-box;
}

input[type=text], select, textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  resize: vertical;
}

label {
  padding: 12px 12px 12px 0;
  display: inline-block;
}

input[type=submit] {
  background-color: #04AA6D;
  color: white;
  padding: 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  float: right;
}

input[type=submit]:hover {
  background-color: #45a049;
}

.container {
  border-radius: 5px;
  background-color: #f2f2f2;
  padding: 20px;
}
                </style>
            </head>
            <body>
                <form action='/predict' method='GET'>
  
                        <div className="form-group">
                            <label for="">Home Team</label>
                            <select name='home_team'  className="form-control" id="">
                            <option>Morocco</option>
                            <option>Egypt</option>
                            <option>Algeria</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label for="">Away Team</label>
                            <select name='away_team'  class="form-control" id="">
                            <option>Morocco</option>
                            <option>Egypt</option>
                            <option>Algeria</option>
                            </select>
                        </div>
                        <input type="submit" class="btn btn-success" value="Predicter"/>
                        </form>
            </body>
        </html>
    """
def predict_outcome(home_team: str, away_team: str):
    
    
    
    try:
        df.home_rank.iloc[0] = ranking[((ranking.rank_date == '2021-05-27') & (ranking.country_full == away_team))]['rank'].values[0]
    except:
        df.home_rank.iloc[0] = 155

    try:
        df.away_rank.iloc[0] = ranking[((ranking.rank_date == '2021-05-27') & (ranking.country_full == home_team))]['rank'].values[0]
    except:
        df.away_rank.iloc[0] = 155
        
    df['home_team_'+home_team].iloc[0] = 1
    df['away_team_'+away_team].iloc[0] = 1
    
    #outcome = model.predict(df)
    
    proba = model.predict_proba(df)
    outcome = model.predict(df)

    msg = ''

    if outcome == 'draw':
        msg = '{0} will draw with {1} with {2:.0%} chance'.format(home_team, away_team, proba[0][0])
    elif outcome == 'lose':
        msg = '{0} will lose to {1} with {2:.0%} chance'.format(home_team, away_team, proba[0][1])
    elif outcome == 'win':
        msg = '{0} will win versus {1} with {2:.0%} chance'.format(home_team, away_team, proba[0][2])
    else:
        msg = 'NA'
        
    return {
        'home_team' : home_team,
        'away_team' : away_team,
        'draw' : '{0:.0%}'.format(proba[0][0]),
        'lose' : '{0:.0%}'.format(proba[0][1]),
        'win' : '{0:.0%}'.format(proba[0][2]),
        'message' : msg
    }
        
@app.get('/predict', response_class=HTMLResponse)
async def predictWithResponseHTML(home_team: str, away_team: str):
    res = predict_outcome(home_team, away_team)
    print(res)
    return """
        <html>
                <head>
                    <title></title>
                    <style>
#customers {
  font-family: Arial, Helvetica, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

#customers td, #customers th {
  border: 1px solid #ddd;
  padding: 8px;
}

#customers tr:nth-child(even){background-color: #f2f2f2;}

#customers tr:hover {background-color: #ddd;}

#customers th {
  padding-top: 12px;
  padding-bottom: 12px;
  text-align: left;
  background-color: #04AA6D;
  color: white;
}
</style>
                </head>
                <body>
                    <table id="customers">
                        <tr>
                            <th>home team</th>
                            <th>away team</th>
                            <th>draw</th>
                            <th>lose</th>
                            <th>win</th>
                            <th>message</th>
                        </tr>
                        <td>"""+res['home_team']+"""</td>
                        <td>"""+res['away_team']+"""</td>
                        <td>"""+res['draw']+"""</td>
                        <td>"""+res['lose']+"""</td>
                        <td>"""+res['win']+"""</td>
                        <td>"""+res['message']+"""</td>
                    </table>
                </body>
            </html>
        """

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}