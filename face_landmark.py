



import argparse
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None


def run(model: str, num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:



    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    row_size = 50
    left_margin = 24
    text_color = (0, 0, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(result: vision.FaceLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT


        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1


    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,
        result_callback=save_result)
    detector = vision.FaceLandmarker.create_from_options(options)


    EXPRESSION_MAP = {

    'happy': ['smile', 'cheekRaise', 'lipCornerPull', 'eyesBright', 'mouthOpen', 'eyebrowLift',
                  'headTilt','eyeBlinkLeft', 'eyeBlinkRight'],
    'sad': ['browDownLeft', 'browDownRight', 'mouthFrown'],
    'neutral': ['browInnerUp','mouthClosed', 'eyebrowNeutral'],
    'surprise': ['mouthOpen', 'jawOpen', 'eyeWideLeft', 'eyeWideRight']



    }

    def get_expression(face_blendshapes):

        expression_scores = {'happy': 0, 'sad': 0, 'neutral': 0, 'surprise': 0}


        for blendshape in face_blendshapes:
            for expression, blendshape_names in EXPRESSION_MAP.items():
                if blendshape.category_name in blendshape_names:
                    expression_scores[expression] += blendshape.score


        dominant_expression = max(expression_scores, key=expression_scores.get)
        return dominant_expression, expression_scores[dominant_expression]


    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape


        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)


        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:

            for face_landmarks, face_blendshapes in zip(DETECTION_RESULT.face_landmarks,
                                                        DETECTION_RESULT.face_blendshapes):

                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                    y=landmark.y,
                                                    z=landmark.z) for
                    landmark in face_landmarks
                ])
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())


                dominant_expression, expression_score = get_expression(face_blendshapes)
                dominant_expression_text = f"{dominant_expression.capitalize()} ({expression_score:.2f})"


                forehead_landmark = face_landmarks[10]
                forehead_x = int(forehead_landmark.x * image_width)
                forehead_y = int(forehead_landmark.y * image_height - 20)


                cv2.putText(current_frame, dominant_expression_text,
                            (forehead_x, forehead_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('face_landmarker', current_frame)

        if cv2.waitKey(1) == 27:
            break


    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Name of face landmarker model.', required=False, default='face_landmarker.task')
    parser.add_argument('--numFaces', help='Max number of faces that can be detected by the landmarker.', required=False, default=2)
    parser.add_argument('--minFaceDetectionConfidence', help='The minimum confidence score for face detection.', required=False, default=0.5)
    parser.add_argument('--minFacePresenceConfidence', help='The minimum confidence score of face presence score in the landmark detection.', required=False, default=0.5)
    parser.add_argument('--minTrackingConfidence', help='The minimum confidence score for tracking.', required=False, default=0.5)
    parser.add_argument('--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, default=1280)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, default=960)
    args = parser.parse_args()

    run(args.model, int(args.numFaces), float(args.minFaceDetectionConfidence),
        float(args.minFacePresenceConfidence), float(args.minTrackingConfidence),
        int(args.cameraId), int(args.frameWidth), int(args.frameHeight))


if __name__ == '__main__':
    main()
