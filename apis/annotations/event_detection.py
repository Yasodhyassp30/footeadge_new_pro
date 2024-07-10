# from openai import OpenAI
# import os
# import cv2
# from moviepy.editor import VideoFileClip
# import time
# import base64


# open_ai_api_key = "sk-proj-TQ0fJXcevC4Suxj8rJJOT3BlbkFJ5AQaMcGlBRik0M93uPOn"
# model = "gpt-4o"
# client = OpenAI(api_key=open_ai_api_key)
# VIDEO_PATH = "./6a4bd98f-17c1-4876-b6df-5d69fcfe20f0.mp4"
# def process_video(video_path, seconds_per_frame=2):
#     # Create a folder named 'frames' if it doesn't exist
#     frames_folder = "frames"
#     if not os.path.exists(frames_folder):
#         os.makedirs(frames_folder)

#     base64frames = []
#     base_video_path, _ = os.path.splitext(video_path)
#     print(base_video_path)

#     video = cv2.VideoCapture(video_path)
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frames_to_skip = int(fps * seconds_per_frame)

#     curr_frame = 0
#     frame_count = 0

#     while curr_frame < total_frames - 1:
#         video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
#         success, frame = video.read()

#         if not success:
#             break

#         frame_filename = os.path.join(frames_folder, f"frame_{frame_count:05d}.jpg")
#         cv2.imwrite(frame_filename, frame)

#         _, buffer = cv2.imencode(".jpg", frame)
#         base64frames.append(base64.b64encode(buffer).decode("utf-8"))

#         curr_frame += frames_to_skip
#         frame_count += 1

#     video.release()

#     return base64frames


# base64_frames = process_video(VIDEO_PATH)
# base64_frames = base64_frames[::20]
# response = client.chat.completions.create(
#     model=model,
#     messages=[
#         {
#             "role": "system",
#             "content": "You are generating a detailed soccer event analysis with advanced KPIs and metrics. Please provide an in-depth analysis of the soccer game, respond in Markdown.",
#         },
#         {
#             "role": "user",
#             "content": [
#                 "These are the frames from the soccer video. Analyze the events and provide insights on key performance indicators (KPIs) such as passing accuracy, shot accuracy, possession, player movement heatmaps, and other advanced metrics. Include details on notable events like goals, fouls, assists, and player performance evaluations.",
#                 *map(
#                     lambda x: {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpg;base64,{x}",
#                             "detail": "low",
#                         },
#                     },
#                     base64_frames,
#                 ),
#             ],
#         },
#     ],
#     temperature=0,
# )
# print(response.choices[0].message.content)

import os
import cv2
import base64
import time
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from datetime import datetime
from openai import OpenAI

event_detection = Blueprint(
    "event-detection", __name__, url_prefix="/api/event-detection"
)

# OpenAI Configuration
open_ai_api_key = "sk-proj-TQ0fJXcevC4Suxj8rJJOT3BlbkFJ5AQaMcGlBRik0M93uPOn"
model = "gpt-4o"
client = OpenAI(api_key=open_ai_api_key)

# Configuration for allowed file extensions and upload folder
ALLOWED_EXTENSIONS = {"mp4"}
UPLOAD_FOLDER = "uploads/event_detection/"

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


def process_video(video_path, seconds_per_frame=2):
    # Create a folder named 'frames' if it doesn't exist
    frames_folder = "frames"
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    base64frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)

    curr_frame = 0
    frame_count = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()

        if not success:
            break

        frame_filename = os.path.join(frames_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        _, buffer = cv2.imencode(".jpg", frame)
        base64frames.append(base64.b64encode(buffer).decode("utf-8"))

        curr_frame += frames_to_skip
        frame_count += 1

    video.release()

    return base64frames


@event_detection.route("/analyze_video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(video_file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Create a unique folder for this analysis using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder = os.path.join(UPLOAD_FOLDER, timestamp)
    if not os.path.exists(unique_folder):
        os.makedirs(unique_folder)

    # Save video file
    video_path = save_file(video_file, unique_folder)

    # Process video to extract frames
    base64_frames = process_video(video_path)
    base64_frames = base64_frames[::20]  # Reduce number of frames sent to OpenAI

    # try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are generating a detailed soccer event analysis with advanced KPIs and 
                metrics only from the provided set of frames which were extracted 
                from a short video. Please provide an in-depth analysis of the soccer 
                game, respond in Markdown. 
                DO NOT MENTION AS THIS A FRAME BY FRAME ANALYSIS AND DO NOT 
                BUILD UP COLLECTIVE INSIGHTS FROM COMBINING ALL THE FRAMES.""",
            },
            {
                "role": "user",
                "content": [
                    """These are the frames from the soccer video. Analyze the events 
                    and provide insights on key performance indicators (KPIs) such as 
                    passing accuracy, shot accuracy, possession, player movement heatmaps, 
                    and other advanced metrics that can be highlighted from the given set of frames. 
                    Include details on notable events 
                    like goals, fouls, assists, and player performance evaluations.
                    DO NOT MENTION AS THIS A FRAME BY FRAME ANALYSIS AND DO NOT 
                    BUILD UP COLLECTIVE INSIGHTS FROM COMBINING ALL THE FRAMES.""",
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64_frames,
                    ),
                ],
            },
        ],
        temperature=0,
    )

    analysis_result = response.choices[0].message.content
    return jsonify({"analysis": analysis_result}), 200


@event_detection.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(video_file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Create a unique folder for this analysis using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder = os.path.join(UPLOAD_FOLDER, timestamp)
    if not os.path.exists(unique_folder):
        os.makedirs(unique_folder)

    # Save video file
    video_path = save_file(video_file, unique_folder)

    # Process video to extract frames
    base64_frames = process_video(video_path)
    base64_frames = base64_frames[::20]  # Reduce number of frames sent to OpenAI

    annotations = request.form.get("annotations")
    custom_buttons = request.form.get("customButtons")

    # Prepare the prompt for OpenAI API
    prompt_system = f"""You are generating a detailed soccer event analysis with advanced KPIs and
        metrics based on the provided annotations and video snapshots.
        Please provide an in-depth analysis of the soccer game, respond in Markdown.
        Your analysis should include insights on key performance indicators (KPIs) such as
        passing accuracy, shot accuracy, possession, player movement heatmaps, and other advanced metrics.
        Highlight notable events like goals, fouls, assists, and player performance evaluations.
        Provide detailed, structured insights and recommendations for future practice sessions. 
        Include snapshots from the video where necessary to support your analysis. 
        Ensure your report is structured to be actionable for the team's next practice session.
        Annotations: {annotations}
        """

    prompt_user = [
        """These are the frames from the soccer video. Analyze the events 
        and provide insights on key performance indicators (KPIs) such as 
        passing accuracy, shot accuracy, possession, player movement heatmaps, 
        and other advanced metrics that can be highlighted from the given set of frames. 
        Include details on notable events like goals, fouls, assists, and player performance evaluations.""",
        *map(
            lambda x: {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpg;base64,{x}",
                    "detail": "low",
                },
            },
            base64_frames,
        ),
    ]

    # Call the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are generating a detailed soccer event analysis with advanced KPIs and
        metrics based on the provided annotations and video snapshots.
        Please provide an in-depth analysis of the soccer game, respond in Markdown.
        Your analysis should include insights on key performance indicators (KPIs) such as
        passing accuracy, shot accuracy, possession, player movement heatmaps, and other advanced metrics.
        Highlight notable events like goals, fouls, assists, and player performance evaluations.
        Provide detailed, structured insights and recommendations for future practice sessions. 
        Include snapshots from the video where necessary to support your analysis. 
        Ensure your report is structured to be actionable for the team's next practice session.
        Annotations: {annotations}
        """,
            },
            {
                "role": "user",
                "content": [
                    """These are the frames from the soccer video. Analyze the events 
        and provide insights on key performance indicators (KPIs) such as 
        passing accuracy, shot accuracy, possession, player movement heatmaps, 
        and other advanced metrics that can be highlighted from the given set of frames. 
        Include details on notable events like goals, fouls, assists, and player performance evaluations.""",
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64_frames,
                    ),
                ],
            },
        ],
        temperature=0,
    )

    analysis_result = response.choices[0].message.content
    return jsonify({"analysis": analysis_result}), 200


# Ensure the blueprint is registered in the main app file (e.g., main.py)
# app.register_blueprint(annotation)

# if __name__ == "__main__":
#     app.run(debug=True)
