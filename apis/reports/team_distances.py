from flask import Blueprint, jsonify, current_app, request
from bson.json_util import dumps

import math

team_distances = Blueprint('team_distances', __name__, url_prefix='/api/reports')

@team_distances.route("/team_distances", methods=['GET'])
def get_team_distances():
    try:
        filter_ids = request.args.get('filterIds', '').split(',')
        filter_ids = set(filter_ids) if filter_ids[0] else set()

        db = current_app.config['Mongo_db']
        latest_data = db.analysis.find_one({}, {'analysis.players': 1}, sort=[('date', -1)])
        if latest_data is None:
            
            return jsonify({'error': "No data found"}), 404

        team1_distances = []
        team2_distances = []
        team_colors = {}

        for player in latest_data['analysis']['players']:
            if player['id'] in filter_ids:
                continue

            player_color = tuple(player.get('color'))
            if player_color not in team_colors:
                if len(team_colors) == 0:
                    team_colors[player_color] = 'team1'
                elif len(team_colors) == 1:
                    team_colors[player_color] = 'team2'
                else:
                    continue  

            distance = calculate_distance(player['positions'])
            if distance < 1:
                continue 

            if team_colors[player_color] == 'team1':
                team1_distances.append(distance)
            elif team_colors[player_color] == 'team2':
                team2_distances.append(distance)



        return dumps({'team1_distances': team1_distances, 'team2_distances': team2_distances}), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the team distances"}), 500

def calculate_distance(positions):
    total_distance = 0
    for i in range(1, len(positions)):
        x1, y1 = positions[i-1]
        x2, y2 = positions[i]
        total_distance += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return total_distance
