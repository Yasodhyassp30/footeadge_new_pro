import os
import uuid
import base64
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.json_util import dumps
from bson import ObjectId

annotation = Blueprint("annotation", __name__, url_prefix="/api")

# Configuration for allowed file extensions and upload folder
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}
UPLOAD_FOLDER = "uploads/analysis/"

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file, folder):
    filename = secure_filename(file.filename)
    filepath = os.path.join(folder, filename)
    file.save(filepath)
    return filepath


def save_base64_file(data, folder, file_extension):
    try:
        if "," in data:
            _, encoded = data.split(",", 1)
        else:
            encoded = data
        filename = f"{uuid.uuid4()}.{file_extension}"
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as file:
            file.write(base64.b64decode(encoded))  # Decode base64 and save as file
        return filepath
    except Exception as e:
        raise ValueError(f"Error saving base64 file: {str(e)}")


@annotation.route("/save_annotations", methods=["POST"])
def save_annotations():
    data = request.json

    # Create a unique folder for this analysis using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder = os.path.join(UPLOAD_FOLDER, timestamp)
    if not os.path.exists(unique_folder):
        os.makedirs(unique_folder)

    # Save video file
    video_src = data.get("videoSrc")
    video_filename = None
    if video_src:
        video_filename = save_base64_file(video_src, unique_folder, "mp4")

    # Save snapshots
    snapshots = data.get("snapshots", [])
    snapshot_paths = []
    for snapshot in snapshots:
        snapshot_filename = save_base64_file(snapshot, unique_folder, "png")
        snapshot_paths.append(snapshot_filename)

    # Save annotation data to MongoDB
    annotations_data = {
        "annotations": data.get("annotations", []),
        "customButtons": data.get("customButtons", []),
        "analysis_result": data.get("analysisResult", ""),
        "video_path": video_filename,
        "snapshot_paths": snapshot_paths,
        "created_at": datetime.now(),
    }

    db = current_app.config["Mongo_db"]
    result = db.annotations.insert_one(annotations_data)
    return (
        jsonify(
            {"message": "Annotations saved successfully", "id": str(result.inserted_id)}
        ),
        201,
    )


def convert_file_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


@annotation.route("/retrieve_analysis/<id>", methods=["GET"])
def retrieve_analysis(id):
    try:
        db = current_app.config["Mongo_db"]
        analysis = db.annotations.find_one({"_id": ObjectId(id)})
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404

        # Convert snapshot paths to base64
        snapshot_paths = analysis.get("snapshot_paths", [])
        snapshots_base64 = []
        for path in snapshot_paths:
            if os.path.exists(path):
                snapshots_base64.append(convert_file_to_base64(path))

        # Convert video path to base64 if it exists
        video_base64 = None
        video_path = analysis.get("video_path")
        if video_path and os.path.exists(video_path):
            video_base64 = convert_file_to_base64(video_path)

        analysis["snapshot_base64"] = snapshots_base64
        analysis["video_base64"] = video_base64

        return dumps(analysis), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@annotation.route("/list_analyses", methods=["GET"])
def list_analyses():
    try:
        db = current_app.config["Mongo_db"]
        analyses = db.annotations.find()
        analyses_list = []
        for analysis in analyses:
            analysis_summary = {
                "id": str(analysis["_id"]),
                "created_at": (
                    analysis.get("created_at").strftime("%Y-%m-%d %H:%M:%S")
                    if analysis.get("created_at")
                    else None
                ),
                "video_path": analysis.get("video_path"),
                "snapshot_count": len(analysis.get("snapshot_paths", [])),
                "annotation_count": len(analysis.get("annotations", [])),
            }
            analyses_list.append(analysis_summary)

        return dumps(analyses_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Ensure the blueprint is registered in the main app file (e.g., main.py)
# from apis.annotation import annotation
# app.register_blueprint(annotation)
