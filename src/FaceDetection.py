import cv2

def face_detection(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('library/haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=5)
    return faces

def draw_rectangle(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness = 2)
    return img