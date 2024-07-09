from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, join_room
from dotenv import load_dotenv


from apis.tactical_analysis.blueprint import tactical_analysis, tactical_analysis_video
from utils.logger import configure_logger
from database.databaseConnection import get_db
from apis.authentication.blueprint import user
from apis.reports.reports import register_reports_blueprints
from apis.player_profile.register_players import register_players_blueprints


load_dotenv()
app = Flask(__name__)

CORS(app)


logger = configure_logger()

app.register_blueprint(tactical_analysis)
app.register_blueprint(user)
register_reports_blueprints(app)
register_players_blueprints(app)


app.config["Mongo_db"] = get_db()
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("join_room")
def handle_join_room(data):
    room_id = data.get("roomId")
    if room_id:
        join_room(room_id)
        print(f"Client {request.sid} joined room {room_id}")
    else:
        print("Invalid room ID provided")


@socketio.on("start_processing")
def start_processing(data):
    logger.info("Start processing request.")
    video_path = data["video_path"]

    ret = tactical_analysis_video(video_path, socketio=socketio)
    if ret == 0:
        print(f"Processing of video {video_path} completed")
    logger.info("End processing request.")
    return None


@socketio.on("disconnect")
def handle_disconnect():
    print(f"Client {request.sid} disconnected")


if __name__ == "__main__":
    socketio.run(app, debug=True)
