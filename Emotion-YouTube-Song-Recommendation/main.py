# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import cv2
# import numpy as np
# import webbrowser

# emotion_to_query = {
#     'Happy': 'latest hindi happy songs',
#     'Sad': 'latest hindi sad songs',
#     'Angry': 'latest hindi motivational songs',
#     'Fear': 'latest hindi inspirational songs',
#     'Disgust': 'latest hindi soothing songs',
#     'Surprise': 'latest hindi party songs',
#     'Neutral': 'latest hindi chill songs'
# }

# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# classifier = load_model('Z:/Chinmay Rane/MCA/SYMCA/Mini_Project_2/model.h5')

# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# cap = cv2.VideoCapture(0)

# detected_emotion = ""  

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    
#     cv2.putText(frame, f'Emotion: {detected_emotion}', (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('Emotion Detector (Press A to Analyze)', frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('a'):
#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y + h, x:x + w]
#             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#             if np.sum(roi_gray) != 0:
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)
#                 roi = np.expand_dims(roi, axis=-1)

#                 prediction = classifier.predict(roi)[0]
#                 label = emotion_labels[prediction.argmax()]
#                 detected_emotion = label
#                 print("Detected Emotion:", label)

#                 if label in emotion_to_query:
#                     query = emotion_to_query[label]
#                     url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
#                     webbrowser.open(url)

#     elif key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from collections import Counter
import time
import webbrowser

# Emotion to YouTube search query mapping
emotion_to_query = {
    'Happy': 'latest hindi happy songs',
    'Sad': 'latest hindi sad songs',
    'Angry': 'latest hindi motivational songs',
    'Fear': 'latest hindi inspirational songs',
    'Disgust': 'latest hindi soothing songs',
    'Surprise': 'latest hindi party songs',
    'Neutral': 'latest hindi chill songs'
}

# Load face detector and emotion model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('Z:/Chinmay Rane/MCA/SYMCA/Mini_Project_2/model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

detected_emotion = ""
current_emotion = ""
collected_emotions = []
collecting = False
start_time = None
collection_duration = 5  # seconds

print("Press 'A' to start emotion detection for 5 seconds.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = classifier.predict(roi, verbose=0)[0]
            current_emotion = emotion_labels[prediction.argmax()]

            # Draw face box and emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, current_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if collecting:
                collected_emotions.append(current_emotion)

    # Show detected emotion in real-time
    cv2.putText(frame, f'Real-Time Emotion: {current_emotion}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # If collection is ongoing
    if collecting:
        elapsed = time.time() - start_time
        remaining = collection_duration - elapsed
        cv2.putText(frame, f'Collecting... {int(remaining)}s left', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

        if elapsed >= collection_duration:
            collecting = False
            if collected_emotions:
                # Determine most common emotion
                most_common = Counter(collected_emotions).most_common(1)[0][0]
                detected_emotion = most_common
                print("Detected Emotion (after 5 seconds):", detected_emotion)

                # Open YouTube with corresponding query
                if detected_emotion in emotion_to_query:
                    query = emotion_to_query[detected_emotion]
                    url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                    webbrowser.open(url)
            collected_emotions = []

    cv2.imshow('Emotion Detector (Press A to Analyze)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') and not collecting:
        print("Started 5-second emotion collection...")
        collected_emotions = []
        start_time = time.time()
        collecting = True
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
