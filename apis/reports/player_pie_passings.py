from flask import Blueprint, jsonify, current_app, request
from bson.json_util import dumps


player_pie_passings = Blueprint('player_pie_passings', __name__, url_prefix='/api/reports')

@player_pie_passings.route("/player_pie_passings", methods=['GET'])
def get_player_pie_passings():
    try:
        filter_ids = request.args.get('filterIds', '').split(',')
        filter_ids = set(filter_ids) if filter_ids[0] else set()

        db = current_app.config['Mongo_db']
        latest_data = db.analysis.find_one({}, {'passings': 1}, sort=[('date', -1)])
        if latest_data is None:
           
            return jsonify({'error': "No data found"}), 404

        player_passings = {}
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

            if prev_tracker_id not in player_passings:
                player_passings[prev_tracker_id] = {'correct': 0, 'wrong': 0, 'color': prev_player_color}

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
                player_passings[prev_tracker_id]['correct'] += 1
            else:
                player_passings[prev_tracker_id]['wrong'] += 1

        team1_passings = []
        team2_passings = []

        for player_id, passes in player_passings.items():
            data = {
                'id': player_id,
                'correct': passes['correct'],
                'wrong': passes['wrong'],
                'color': passes['color']
            }
            if team_colors[tuple(passes['color'])] == 'team1':
                team1_passings.append(data)
            else:
                team2_passings.append(data)

        return dumps({'team1_passings': team1_passings, 'team2_passings': team2_passings}), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the player pie passings"}), 500
