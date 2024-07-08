from flask import Blueprint, jsonify, current_app, request
from bson.json_util import dumps


team_timeline = Blueprint('team_timeline', __name__, url_prefix='/api/reports')

@team_timeline.route("/team_timeline", methods=['GET'])
def get_team_timeline():
    try:
        filter_ids = request.args.get('filterIds', '').split(',')
        filter_ids = set(filter_ids) if filter_ids[0] else set()

        db = current_app.config['Mongo_db']
        latest_data = db.analysis.find_one({}, {'passings': 1}, sort=[('date', -1)])
        if latest_data is None:
            
            return jsonify({'error': "No data found"}), 404

        team1_timeline = []
        team2_timeline = []
        team_colors = {}
        current_team = None
        start_time = 0

        for i, passing in enumerate(latest_data['passings']):
            tracker_id = str(passing['tracker_id'])
            if tracker_id in filter_ids:
                continue

            player_color = tuple(passing.get('color'))
            if player_color not in team_colors:
                if len(team_colors) == 0:
                    team_colors[player_color] = 'team1'
                elif len(team_colors) == 1:
                    team_colors[player_color] = 'team2'
                else:
                    continue  

            if current_team is None:
                current_team = team_colors[player_color]
                start_time = i
            elif current_team != team_colors[player_color]:
                end_time = i
                if current_team == 'team1':
                    team1_timeline.append({'start': start_time, 'end': end_time})
                else:
                    team2_timeline.append({'start': start_time, 'end': end_time})
                current_team = team_colors[player_color]
                start_time = i

       
        end_time = len(latest_data['passings'])
        if current_team == 'team1':
            team1_timeline.append({'start': start_time, 'end': end_time})
        else:
            team2_timeline.append({'start': start_time, 'end': end_time})

        

        return dumps({'team1_timeline': team1_timeline, 'team2_timeline': team2_timeline}), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the team timeline"}), 500
