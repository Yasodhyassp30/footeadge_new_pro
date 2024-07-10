import base64
import uuid
from flask import Blueprint,request,jsonify,current_app
from flask_socketio import  close_room, join_room
import cv2
import os
from apis.tactical_analysis.objects.soccerfield import Soccerfield
from apis.tactical_analysis.objects.players import Players
from bson.json_util import dumps
from bson import ObjectId
import supervision as sv

tactical_analysis= Blueprint('analysis',__name__,url_prefix='/api')

def tactical_analysis_video(file_path,socketio):
    cap = cv2.VideoCapture(f"uploads/{file_path}")
    byte_tracker = sv.ByteTrack(frame_rate=24,track_thresh=0.65,match_thresh=0.85,track_buffer=30)
    count = 0;
    field = Soccerfield()
    players = Players(byte_tracker)
    counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break
        try:
            frame_copied = frame.copy()

            frame_preprocess = frame.copy()
            frame_copied = field.frame_preprocess(frame_preprocess)
            field.obtain_detections(frame_copied)
            field.organize_detections()
            field.restructured_segment(frame=frame_copied)
            field.determine_points()
            field.first_find_intersection()
            field.second_find_intersection()
            field.first_parallel()
            field.second_parallel()
            field.estimate_homography()
            players.detect_players(frame,homography=field.homography)
            players.clustering()
            #players.mark_players(top_view_marking,error)

            if counter ==12:
                frame_resized = cv2.resize(frame, (1280, 720))
                _, encoded_image = cv2.imencode('.jpg', frame_resized)
                image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                players.ball = [entry for entry in players.ball if entry.get("tracker_id") is not None]
                possesions = {
                    "ball":players.ball,
                    "frame":count
                }
                socketio.emit('message_from_server', {'info':players.details,'frame':image_base64,'ball':possesions, 'count':count}, room=file_path)
                players.ball = []
                counter =0
                count+=1
            counter+=1
        except Exception as e:
            print(e)

    cap.release()
    
    socketio.emit('final_message_from_server', {}, room=file_path)
    byte_tracker.reset()
    close_room(file_path)
    players = None
    return 0

@tactical_analysis.route('/upload',methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    unique_filename = str(uuid.uuid4()) + '.mp4'
    file_path = 'uploads/' + unique_filename
    file.save(file_path)
    return jsonify({'id': str(unique_filename)})

@tactical_analysis.route("/analysis",methods=['POST'])
def save_match():
    try:
        data = request.json
        if data is None:
            return jsonify({'error': "No data provided"})
        db= current_app.config['Mongo_db']

        db.analysis.insert_one(data)

        return jsonify({'message': "Analysis saved successfully"}), 200

    except Exception as e:
        return jsonify({'error': "Error in saving the analysis"}), 500


@tactical_analysis.route("/analysis/players/<id>",methods=['GET'])
def get_match(id):
    try:
        db= current_app.config['Mongo_db']
        data = db.analysis.find_one({'_id':ObjectId(id)},{'analysis.players':1})
        if data is None:
            return jsonify({'error': "No data found"}), 404
        return dumps(data), 200

    except Exception as e:
        return jsonify({'error': "Error in getting the analysis"}), 500
@tactical_analysis.route("/analysis/<id>/players",methods=['GET'])
def get_match_details(id):
    try:
        db= current_app.config['Mongo_db']
        data = db.analysis.find_one({'_id':ObjectId(id)})
        if data is None:
            return jsonify({'error': "No data found"}), 404
        return dumps(data), 200

    except Exception as e:
        return jsonify({'error': "Error in getting the analysis"}), 500
    
@tactical_analysis.route("/analysis/users/<id>",methods=['GET'])
def get_user_matches(id):
    try:
        db= current_app.config['Mongo_db']
        data = db.analysis.find({'id':id},{ '_id': 1, 'name': 1 })
        if data is None:
            return jsonify({'error': "No data found"}), 404
        
        return dumps(data), 200

    except Exception as e:
        return jsonify({'error': "Error in getting the analysis"}), 500
    
@tactical_analysis.route("/analysis/<id>",methods=['GET'])
def get_user_matches_details(id):
    try:
        db= current_app.config['Mongo_db']
        data = db.analysis.find({'id':id},{ '_id': 1, 'name': 1 , 'date':1,'analysis': {
            'colors':1,
            'teams':1,
        }})
        if data is None:
            return jsonify({'error': "No data found"}), 404
        
        return dumps(data), 200

    except Exception as e:
        return jsonify({'error': "Error in getting the analysis"}), 500
