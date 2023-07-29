from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

import pandas as pd
import os
import joblib

app = Flask(__name__)
api = Api(app)

class Predicts(Resource):
    def get(self):
        return {'message': 'Is Alive!'}, 200  

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('FEBRE', required=True)
        parser.add_argument('MIALGIA', required=True)
        parser.add_argument('CEFALEIA', required=True)
        parser.add_argument('EXANTEMA', required=True)
        parser.add_argument('VOMITO', required=True)
        parser.add_argument('NAUSEA', required=True)
        parser.add_argument('DOR_COSTAS', required=True)
        parser.add_argument('ARTRITE', required=True)
        parser.add_argument('ARTRALGIA', required=True)
        parser.add_argument('PETEQUIA_N', required=True)
        parser.add_argument('DOR_RETRO', required=True)
        
        args = parser.parse_args()  # parse arguments to dictionary
        model = joblib.load("./random_forest.joblib")

        # create new dataframe containing new values
        X = pd.DataFrame({
            'FEBRE': [args.FEBRE],
            'MIALGIA': [args.MIALGIA],
            'CEFALEIA': [args.CEFALEIA],
            'EXANTEMA': [args.EXANTEMA],
            'VOMITO': [args.VOMITO],
            'NAUSEA': [args.NAUSEA],
            'DOR_COSTAS': [args.DOR_COSTAS],
            'ARTRITE': [args.ARTRITE],
            'ARTRALGIA': [args.ARTRALGIA],
            'PETEQUIA_N': [args.PETEQUIA_N],
            'DOR_RETRO': [args.DOR_RETRO],
        })

        print(X.head())
        result = model.predict_proba(X.to_numpy())
        print(result)

        return {'result': {'chikv': result[0][0], 'denv': result[0][1]}}, 200  # return data with 200 OK
    
api.add_resource(Predicts, '/predicts')  # '/predicts' is our entry point for Predicts

if __name__ == '__main__':
    CORS(app, support_credentials=True)
    app.run()  # run our Flask app
