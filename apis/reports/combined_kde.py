from flask import Blueprint, jsonify, current_app, request
from bson.json_util import dumps


combined_kde = Blueprint('combined_kde', __name__, url_prefix='/api/reports')

@combined_kde.route("/combined_kde", methods=['GET'])
def get_combined_kde_data():
    try:
        filter_ids = request.args.get('filterIds', '').split(',')
        filter_ids = set(filter_ids) if filter_ids[0] else set()

        db = current_app.config['Mongo_db']
        latest_data = db.analysis.find_one({}, {'analysis.players': 1}, sort=[('date', -1)])
        if latest_data is None:
            
            return jsonify({'error': "No data found"}), 404

        combined_team1_data = []
        combined_team2_data = []
        team_colors = {}

        for player in latest_data['analysis']['players']:
            player_id = str(player['id'])
            if player_id in filter_ids:
                continue

            player_color = tuple(player.get('color'))
            if player_color not in team_colors:
                if len(team_colors) == 0:
                    team_colors[player_color] = 'team1'
                elif len(team_colors) == 1:
                    team_colors[player_color] = 'team2'
                else:
                    continue  

            if team_colors[player_color] == 'team1':
                combined_team1_data.extend(player['positions'])
            elif team_colors[player_color] == 'team2':
                combined_team2_data.extend(player['positions'])

        return dumps({'team1_positions': combined_team1_data, 'team2_positions': combined_team2_data}), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the analysis"}), 500
