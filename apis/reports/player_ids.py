

from flask import Blueprint, jsonify, current_app
from bson.json_util import dumps


player_ids = Blueprint('player_ids', __name__, url_prefix='/api/reports')

@player_ids.route("/player_ids", methods=['GET'])
def get_player_ids():
    try:
        db = current_app.config['Mongo_db']
        latest_data = db.analysis.find_one({}, {'analysis.players.id': 1, 'analysis.players.name': 1}, sort=[('date', -1)])
        if latest_data is None:
            
            return jsonify({'error': "No data found"}), 404

        player_ids = []
        for player in latest_data['analysis']['players']:
            player_ids.append({
                'id': player['id'],
                'name': player['name']
            })

        return dumps({'player_ids': player_ids}), 200

    except Exception as e:
        
        return jsonify({'error': "Error in getting the player ids"}), 500
