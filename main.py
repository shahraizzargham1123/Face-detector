import cv2
import mediapipe as mp
import os

MEMES_DIR = "memes"

EMOTION_TO_MEME = {
    "happy": "happy",
    "angry": "angry",
    "neutral": "neutral",
}

def load_memes():
    memes = {}
    for emotion in ["happy", "angry", "neutral"]:
        for ext in ["jpg", "jpeg", "png"]:
            path = os.path.join(MEMES_DIR, f"{emotion}.{ext}")
            if os.path.exists(path):
                memes[emotion] = cv2.imread(path)
                break
    return memes

def get_y(landmarks, idx, h):
    return landmarks[idx].y * h

def detect_emotion(landmarks, h):
    corner_avg_y = (get_y(landmarks, 61, h) + get_y(landmarks, 291, h)) / 2
    upper_lip_y = get_y(landmarks, 13, h)
    chin_y = get_y(landmarks, 152, h)
    nose_y = get_y(landmarks, 1, h)
    face_height = chin_y - nose_y or 1

    smile_score = (upper_lip_y - corner_avg_y) / face_height

    left_brow_dist = get_y(landmarks, 159, h) - get_y(landmarks, 107, h)
    right_brow_dist = get_y(landmarks, 386, h) - get_y(landmarks, 336, h)
    brow_score = ((left_brow_dist + right_brow_dist) / 2) / face_height

    if smile_score > 0.025:
        return "happy"
    elif brow_score < 0.20:
        return "angry"
    return "neutral"

def show_meme(memes, emotion):
    if emotion in memes and memes[emotion] is not None:
        meme = cv2.resize(memes[emotion], (400, 400))
        cv2.imshow("Meme", meme)

def main():
    memes = load_memes()
    if not memes:
        print(f"No memes found in '{MEMES_DIR}/' folder.")

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    current_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            mp_drawing.draw_landmarks(
                frame, result.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            emotion = detect_emotion(landmarks, h)

            if emotion != current_emotion:
                current_emotion = emotion
                show_meme(memes, EMOTION_TO_MEME[emotion])

            if emotion is not None:
                cv2.putText(frame, emotion.upper(), (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No face detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Face Meme Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
