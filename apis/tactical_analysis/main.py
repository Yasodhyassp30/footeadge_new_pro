import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from objects.soccerfield import Soccerfield
from objects.players import Players
import torch
from mplsoccer.pitch import Pitch


device = torch.device("cpu")



cap = cv2.VideoCapture(f"clips\\9a97dae4_1.mp4")
homographies = []
top_view = cv2.imread("assets\\top_view_no_grass.jpg")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = 25
frame_skip = int(cap.get(cv2.CAP_PROP_FPS) / fps)
cap.set(cv2.CAP_PROP_POS_MSEC, 1000*60*25)

field = Soccerfield()
players = Players()

size = (1280, 1440) 
result = cv2.VideoWriter(f'output_videos/9a97dae4_1.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        25, size) 

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_copied = frame.copy()
    top_view_marking = top_view.copy()
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
    homographies.append(field.homography)
    error = field.error_calculation()

    players.detect_players(frame)
    players.clustering(homography=field.homography)
    players.mark_players(top_view_marking,error)
    
    for point in field.determined_frame_points:
        cv2.circle(frame, point, 6, (0, 0, 0) , -1)
    #for point in field.determined_topview_points:
        #cv2.circle(top_view_marking, point, 6, (0, 0, 0) , -1)
    #print(field.determined_topview_points)
    #print(field.determined_frame_points)
    #field.box_annotator.annotate(scene=frame, detections=field.field_detections)
    #field.box_annotator.annotate(scene=frame, detections=players.players)
    warped_partial_field = cv2.warpPerspective(top_view, field.homography, (frame.shape[1], frame.shape[0]))
    wrapped_result = cv2.add(frame, warped_partial_field)
    frame_resized = cv2.resize(frame, (1280, 720))
    top_view_marking_resized = cv2.resize(top_view_marking, (1280, 720))
    combined_image = np.zeros((1440, 1280, 3), dtype=np.uint8)
    combined_image[:720, :] = frame_resized
    combined_image[720:1440, :] = top_view_marking_resized
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 640, 720)
    cv2.imshow("image",combined_image)
    result.write(combined_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    for _ in range(frame_skip - 1):
        cap.read() 

cap.release()
result.release()
cv2.destroyAllWindows()


homographies_array = np.array(homographies)
fig, ax = plt.subplots(figsize=(10, 6))
for k in range(homographies_array.shape[2]):
    for i in range(3):
        ax.plot(range(len(homographies_array)), homographies_array[:, i, k], label=f'Value {k*3 + i + 1}')
ax.set_xlabel('Homography Index')
ax.set_ylabel('Homography Value')
ax.legend()
plt.show()


pitch = Pitch(line_color='gray', line_zorder=2)
fig, axs = plt.subplots(1, 2, figsize=(16, 9))

pitch.kdeplot(players.teams["team_1_x"], players.teams["team_1_y"], ax=axs[0], cmap='Reds', fill=True, n_levels=10)
axs[0].set_title('Team 1')
pitch.draw(axs[0])
pitch.kdeplot(players.teams["team_2_x"], players.teams["team_2_y"], ax=axs[1], cmap='Blues', fill=True, n_levels=10)
axs[1].set_title('Team 2')
pitch.draw(axs[1])
plt.show()
