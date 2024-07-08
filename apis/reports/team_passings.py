from flask import Blueprint, jsonify, current_app, request
from bson.json_util import dumps

import requests

team_passings = Blueprint('team_passings', __name__, url_prefix='/api/reports')

@team_passings.route("/team_passings", methods=['GET'])
def get_team_passings():
    try:
        filter_ids = request.args.get('filterIds', '').split(',')
        filter_ids = set(filter_ids) if filter_ids[0] else set()

        
        response = requests.get('http://localhost:5000/api/reports/passings', params={'filterIds': ','.join(filter_ids)})
        if response.status_code != 200:
            
            return jsonify({'error': "Error fetching data from player_passings"}), response.status_code

        player_passings_data = response.json()
        team1_passings = player_passings_data['team1_passings']
        team2_passings = player_passings_data['team2_passings']

        cumulative_passings_team1 = []
        cumulative_passings_team2 = []

        total_team1_passings = 0
        total_team2_passings = 0

        max_len = max(len(team1_passings), len(team2_passings))

        for i in range(max_len):
            if i < len(team1_passings):
                total_team1_passings += team1_passings[i]['passings']
            cumulative_passings_team1.append(total_team1_passings)

            if i < len(team2_passings):
                total_team2_passings += team2_passings[i]['passings']
            cumulative_passings_team2.append(total_team2_passings)



        return dumps({'team1_passings': cumulative_passings_team1, 'team2_passings': cumulative_passings_team2}), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the team passings"}), 500
