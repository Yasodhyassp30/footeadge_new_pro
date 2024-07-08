from flask import Blueprint, jsonify, current_app, request
from bson.json_util import dumps


team_pie_passings = Blueprint('team_pie_passings', __name__, url_prefix='/api/reports')

@team_pie_passings.route("/team_pie_passings", methods=['GET'])
def get_team_pie_passings():
    try:
        filter_ids = request.args.get('filterIds', '').split(',')
        filter_ids = set(filter_ids) if filter_ids[0] else set()

        db = current_app.config['Mongo_db']
        latest_data = db.analysis.find_one({}, {'passings': 1}, sort=[('date', -1)])
        if latest_data is None:
            
            return jsonify({'error': "No data found"}), 404

        team1_correct_passes = 0
        team1_wrong_passes = 0
        team2_correct_passes = 0
        team2_wrong_passes = 0

        team_colors = {}

        for i, passing in enumerate(latest_data['passings']):
            if i == 0:
                continue

            prev_passing = latest_data['passings'][i-1]
            tracker_id = str(passing['tracker_id'])
            prev_tracker_id = str(prev_passing['tracker_id'])

            if tracker_id in filter_ids or prev_tracker_id in filter_ids:
                continue

            player_color = tuple(passing.get('color'))
            prev_player_color = tuple(prev_passing.get('color'))

            if player_color not in team_colors:
                if len(team_colors) == 0:
                    team_colors[player_color] = 'team1'
                elif len(team_colors) == 1:
                    team_colors[player_color] = 'team2'
                else:
                    continue  

            if prev_player_color not in team_colors:
                if len(team_colors) == 0:
                    team_colors[prev_player_color] = 'team1'
                elif len(team_colors) == 1:
                    team_colors[prev_player_color] = 'team2'
                else:
                    continue  

            if team_colors[player_color] == team_colors[prev_player_color]:
                if team_colors[player_color] == 'team1':
                    team1_correct_passes += 1
                else:
                    team2_correct_passes += 1
            else:
                if team_colors[prev_player_color] == 'team1':
                    team1_wrong_passes += 1
                else:
                    team2_wrong_passes += 1

        

        return dumps({
            'team1_passes': {
                'correct': team1_correct_passes,
                'wrong': team1_wrong_passes
            },
            'team2_passes': {
                'correct': team2_correct_passes,
                'wrong': team2_wrong_passes
            }
        }), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the team pie passings"}), 500
