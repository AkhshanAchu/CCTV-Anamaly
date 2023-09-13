import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO
# Path to the video file

def running(path):
    video_path = path

    # Output video path
    output_path = r"C:\Users\akhsh\Desktop\Project\Fun\Funny\out.mp4"

    # Read the video
    video = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the distance between frames dynamically
    distance_between_frames = min(frame_width, frame_height) // 10

    # Calculate the speed thresholds dynamically
    vehicle_speed_threshold = frame_width // 100
    person_running_threshold = frame_height // 200
    model = YOLO()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize variables for speed estimation and object tracking
    prev_frame = None
    prev_detections = {}
    person_ids = {}
    vehicle_ids = {}
    frame_count=0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % 7 == 0:
            result = model.predict(frame)
            for i in result:
                boxes = [j.xyxy[0] for j in i.boxes]
                classes = [j.cls for j in i.boxes]
            # Filter detections to keep only person and vehicle classes
            person_vehicle_bbox = [boxes[i].detach().cpu().numpy() for i in range(len(classes)) if classes[i] in [0,1,2,3,4,5,7]]

            # Draw bounding boxes and labels on the frame
            output_frame = frame.copy()
            for i, bbox in enumerate(person_vehicle_bbox):
                x, y, w, h = bbox
                cv2.rectangle(output_frame, (round(x), round(y)), (round(x+w), round(y+h)), (0, 255, 0), 2)
                #cv2.putText(output_frame, classes[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Track person IDs
                if classes[i] == 0:
                    centroid_x = int((x + x + w) / 2)
                    centroid_y = int((y + y + h) / 2)
                    centroid = (centroid_x, centroid_y)

                    if i not in person_ids:
                        # Assign a new ID to the person
                        person_ids[i] = len(person_ids) + 1

                    person_id = person_ids[i]

                    # Display the person ID on the frame
                    cv2.putText(output_frame, f"Person ID: {person_id}", (round(x), round(y) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Calculate speed and distance only if there are previous detections available
                    if prev_frame is not None:
                        if i in prev_detections:
                            # Person ID exists in previous frame, calculate speed and distance
                            prev_centroid_x, prev_centroid_y = prev_detections[i]
                            distance = ((centroid_x - prev_centroid_x) ** 2 + (centroid_y - prev_centroid_y) ** 2) ** 0.5

                            # Calculate the speed (pixels per second)
                            speed = distance * fps / distance_between_frames

                            # Display the speed and distance on the frame
                            cv2.putText(output_frame, f"Person ID: {person_id}, Speed: {speed:.2f} pixels/sec, Distance: {distance:.2f} pixels", (centroid_x, centroid_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            # Check for running
                            if speed > person_running_threshold:
                                cv2.putText(output_frame, f"Person ID: {person_id}, Running", (centroid_x, centroid_y + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                print(f"Person ID: {person_id} is running.")

                    # Update the person's centroid in the previous detections
                    prev_detections[i] = centroid

                # Track vehicle IDs
                if classes[i] in [1,2,3,4,5,7]:
                    centroid_x = int((x + x + w) / 2)
                    centroid_y = int((y + y + h) / 2)
                    centroid = (centroid_x, centroid_y)

                    if i not in vehicle_ids:
                        # Assign a new ID to the vehicle
                        vehicle_ids[i] = len(vehicle_ids) + 1

                    vehicle_id = vehicle_ids[i]

                    # Display the vehicle ID on the frame
                    cv2.putText(output_frame, f"Vehicle ID: {vehicle_id}", (round(x), round(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Calculate speed and distance only if there are previous detections available
                    if prev_frame is not None:
                        if i in prev_detections:
                            # Vehicle ID exists in previous frame, calculate speed and distance
                            prev_centroid_x, prev_centroid_y = prev_detections[i]
                            distance = ((centroid_x - prev_centroid_x) ** 2 + (centroid_y - prev_centroid_y) ** 2) ** 0.5

                            # Calculate the speed (pixels per second)
                            speed = distance * fps / distance_between_frames

                            # Display the speed and distance on the frame
                            cv2.putText(output_frame, f"Vehicle ID: {vehicle_id}, Speed: {speed:.2f} pixels/sec, Distance: {distance:.2f} pixels", (centroid_x, centroid_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            # Check for overspeeding
                            if speed > vehicle_speed_threshold:
                                cv2.putText(output_frame, f"Vehicle ID: {vehicle_id}, Overspeeding", (centroid_x, centroid_y + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                print(f"Vehicle ID: {vehicle_id} is overspeeding.")

                    # Update the vehicle's centroid in the previous detections
                    prev_detections[i] = centroid
        # # Perform object detection
        # bbox, label, conf = cv.detect_common_objects(frame)

        # # Filter detections to keep only person and vehicle classes
        # person_vehicle_bbox = [bbox[i] for i in range(len(bbox)) if label[i] in ['person', 'car', 'motorbike', 'bus', 'truck']]

        # # Draw bounding boxes and labels on the frame
        # output_frame = frame.copy()
        # for i, bbox in enumerate(person_vehicle_bbox):
        #     x, y, w, h = bbox
        #     cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     cv2.putText(output_frame, label[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #     # Track person IDs
        #     if label[i] == 'person':
        #         centroid_x = int((x + x + w) / 2)
        #         centroid_y = int((y + y + h) / 2)
        #         centroid = (centroid_x, centroid_y)

        #         if i not in person_ids:
        #             # Assign a new ID to the person
        #             person_ids[i] = len(person_ids) + 1

        #         person_id = person_ids[i]

        #         # Display the person ID on the frame
        #         cv2.putText(output_frame, f"Person ID: {person_id}", (x, y - 40),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #         # Calculate speed and distance only if there are previous detections available
        #         if prev_frame is not None:
        #             if i in prev_detections:
        #                 # Person ID exists in previous frame, calculate speed and distance
        #                 prev_centroid_x, prev_centroid_y = prev_detections[i]
        #                 distance = ((centroid_x - prev_centroid_x) ** 2 + (centroid_y - prev_centroid_y) ** 2) ** 0.5

        #                 # Calculate the speed (pixels per second)
        #                 speed = distance * fps / distance_between_frames

        #                 # Display the speed and distance on the frame
        #                 cv2.putText(output_frame, f"Person ID: {person_id}, Speed: {speed:.2f} pixels/sec, Distance: {distance:.2f} pixels", (centroid_x, centroid_y - 10),
        #                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #                 # Check for running
        #                 if speed > person_running_threshold:
        #                     cv2.putText(output_frame, f"Person ID: {person_id}, Running", (centroid_x, centroid_y + 30),
        #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        #                     print(f"Person ID: {person_id} is running.")

        #         # Update the person's centroid in the previous detections
        #         prev_detections[i] = centroid

        #     # Track vehicle IDs
        #     if label[i] in ['car', 'motorbike', 'bus', 'truck']:
        #         centroid_x = int((x + x + w) / 2)
        #         centroid_y = int((y + y + h) / 2)
        #         centroid = (centroid_x, centroid_y)

        #         if i not in vehicle_ids:
        #             # Assign a new ID to the vehicle
        #             vehicle_ids[i] = len(vehicle_ids) + 1

        #         vehicle_id = vehicle_ids[i]

        #         # Display the vehicle ID on the frame
        #         cv2.putText(output_frame, f"Vehicle ID: {vehicle_id}", (x, y - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #         # Calculate speed and distance only if there are previous detections available
        #         if prev_frame is not None:
        #             if i in prev_detections:
        #                 # Vehicle ID exists in previous frame, calculate speed and distance
        #                 prev_centroid_x, prev_centroid_y = prev_detections[i]
        #                 distance = ((centroid_x - prev_centroid_x) ** 2 + (centroid_y - prev_centroid_y) ** 2) ** 0.5

        #                 # Calculate the speed (pixels per second)
        #                 speed = distance * fps / distance_between_frames

        #                 # Display the speed and distance on the frame
        #                 cv2.putText(output_frame, f"Vehicle ID: {vehicle_id}, Speed: {speed:.2f} pixels/sec, Distance: {distance:.2f} pixels", (centroid_x, centroid_y - 10),
        #                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #                 # Check for overspeeding
        #                 if speed > vehicle_speed_threshold:
        #                     cv2.putText(output_frame, f"Vehicle ID: {vehicle_id}, Overspeeding", (centroid_x, centroid_y + 30),
        #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        #                     print(f"Vehicle ID: {vehicle_id} is overspeeding.")

        #         # Update the vehicle's centroid in the previous detections
        #         prev_detections[i] = centroid

        # Write the frame to the output video
        output_video.write(output_frame)

        # Update the previous frame
        prev_frame = frame
        frame_count+=1
        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and output video
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

running(r"C:\Users\akhsh\Desktop\Project\Fun\Funny\testcase\17.mp4")