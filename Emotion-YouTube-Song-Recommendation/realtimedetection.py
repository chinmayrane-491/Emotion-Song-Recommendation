import cv2
face_cap = cv2.CascadeClassifier("c:/Users/ranea/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)
while True:
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    face = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in face:
        cv2.rectangle(video_data,(x,y),(x+w,x+h),(0,0,255),2)
    cv2.imshow("Video Live",video_data)
    if cv2.waitKey(1) == ord("a"):
        break
video_cap.release()
