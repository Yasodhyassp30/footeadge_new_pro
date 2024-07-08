

from flask import Blueprint, jsonify, current_app, request
from bson.json_util import dumps


player_passings = Blueprint('player_passings', __name__, url_prefix='/api/reports')

@player_passings.route("/passings", methods=['GET'])
def get_player_passings():
    try:
        filter_ids = request.args.get('filterIds', '').split(',')
        filter_ids = set(filter_ids) if filter_ids[0] else set()

        db = current_app.config['Mongo_db']
        latest_data = db.analysis.find_one({}, {'passings': 1}, sort=[('date', -1)])
        if latest_data is None:
            
            return jsonify({'error': "No data found"}), 404

        team1_passings = {}
        team2_passings = {}
        team_colors = {}

        for passing in latest_data['passings']:
            tracker_id = str(passing['tracker_id'])
            if tracker_id in filter_ids:
                continue

            player_color = tuple(passing.get('color'))
            if player_color not in team_colors:
                if len(team_colors) == 1:
                    team_colors[player_color] = 'team1'
                elif len(team_colors) == 0:
                    team_colors[player_color] = 'team2'
                else:
                    continue  

            if team_colors[player_color] == 'team1':
                if tracker_id not in team1_passings:
                    team1_passings[tracker_id] = {
                        'id': tracker_id,
                        'name': f'Player {tracker_id}',
                        'passings': 0
                    }
                team1_passings[tracker_id]['passings'] += 1
            elif team_colors[player_color] == 'team2':
                if tracker_id not in team2_passings:
                    team2_passings[tracker_id] = {
                        'id': tracker_id,
                        'name': f'Player {tracker_id}',
                        'passings': 0
                    }
                team2_passings[tracker_id]['passings'] += 1

        return dumps({
            'team1_passings': list(team1_passings.values()),
            'team2_passings': list(team2_passings.values())
        }), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the player passings"}), 500
