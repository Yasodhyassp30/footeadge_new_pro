import os
import base64
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.json_util import dumps
from bson import ObjectId

player_profile = Blueprint("player_profile", __name__, url_prefix="/api/players")

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}
UPLOAD_FOLDER = "player_data_uploads/"

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@player_profile.route("/add_player", methods=["POST"])
def add_player():
    data = request.form.to_dict()
    files = request.files

    player_data = {
        "name": data.get("name"),
        "age": data.get("age"),
        "gender": data.get("gender"),
        "height": data.get("height"),
        "weight": data.get("weight"),
        "club": data.get("club"),
        "position": data.get("position"),
        "nationality": data.get("nationality"),
        "dob": data.get("dob"),
        "preferredFoot": data.get("preferredFoot"),
        "playerNumber": data.get("playerNumber"),
        "email": data.get("email"),
        "phone": data.get("phone"),
        "physicalEndurance": data.get("physicalEndurance"),
        "speed": data.get("speed"),
        "strength": data.get("strength"),
        "injuryHistory": data.get("injuryHistory"),
        "matchesPlayed": data.get("matchesPlayed"),
        "minutesPlayed": data.get("minutesPlayed"),
        "goalsScored": data.get("goalsScored"),
        "assists": data.get("assists"),
        "shotsOnTarget": data.get("shotsOnTarget"),
        "passAccuracy": data.get("passAccuracy"),
        "dribbles": data.get("dribbles"),
        "tackles": data.get("tackles"),
        "interceptions": data.get("interceptions"),
        "foulsCommitted": data.get("foulsCommitted"),
        "yellowCards": data.get("yellowCards"),
        "redCards": data.get("redCards"),
        "averageRating": data.get("averageRating"),
        "trainingSessions": data.get("trainingSessions"),
        "trainingHours": data.get("trainingHours"),
        "skillImprovement": data.get("skillImprovement"),
        "coachFeedback": data.get("coachFeedback"),
        "awards": data.get("awards"),
        "trophies": data.get("trophies"),
        "records": data.get("records"),
        "education": data.get("education"),
        "languagesSpoken": data.get("languagesSpoken"),
        "hobbies": data.get("hobbies"),
        "ballControl": data.get("ballControl"),
        "passing": data.get("passing"),
        "shooting": data.get("shooting"),
        "defending": data.get("defending"),
        "tacticalAwareness": data.get("tacticalAwareness"),
        "mentalToughness": data.get("mentalToughness"),
        "leadership": data.get("leadership"),
        "teamwork": data.get("teamwork"),
        "decisionMaking": data.get("decisionMaking"),
        "socialMediaHandles": data.get("socialMediaHandles"),
        "publicAppearances": data.get("publicAppearances"),
        "fanEngagement": data.get("fanEngagement"),
        "heatmaps": data.get("heatmaps"),
        "passMaps": data.get("passMaps"),
        "shotMaps": data.get("shotMaps"),
        "keyMoments": data.get("keyMoments"),
    }

    # Handling image upload
    image_file = files.get("image")
    if image_file and allowed_file(image_file.filename):
        image_filename = secure_filename(image_file.filename)
        image_path = os.path.join("player_data_uploads", image_filename)
        image_file.save(image_path)
        player_data["image"] = image_path

    # Handling training videos upload
    training_videos = files.getlist("trainingVideos")
    video_paths = []
    for video_file in training_videos:
        if video_file and allowed_file(video_file.filename):
            video_filename = secure_filename(video_file.filename)
            video_path = os.path.join("player_data_uploads", video_filename)
            video_file.save(video_path)
            video_paths.append(video_path)
    player_data["trainingVideos"] = video_paths

    # Insert player data into the database
    db = current_app.config["Mongo_db"]
    db.players.insert_one(player_data)

    return jsonify({"message": "Player profile added successfully"}), 201


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@player_profile.route("/players", methods=["GET"])
def get_all_players():
    db = current_app.config["Mongo_db"]
    players = db.players.find()
    players_list = []
    for player in players:
        player["_id"] = str(player["_id"])
        if "image" in player:
            player["image_base64"] = convert_image_to_base64(
                os.path.join(player["image"])
            )
        players_list.append(player)
    return dumps(players_list), 200


@player_profile.route("/players/<id>", methods=["GET"])
def get_player(id):
    # try:
    print(id)
    db = current_app.config["Mongo_db"]
    player = db.players.find_one({"_id": ObjectId(id)})
    if not player:
        return jsonify({"error": "Player not found"}), 404
    player["_id"] = str(player["_id"])
    if "image" in player:
        player["image_base64"] = convert_image_to_base64(os.path.join(player["image"]))
    return dumps(player), 200
    # except Exception as e:
    #     return jsonify({"error": "Error in getting player"}), 500


# Ensure the blueprint is registered in the main app file (e.g., main.py)
# from apis.player_profile.add_player import player_profile
# app.register_blueprint(player_profile)
