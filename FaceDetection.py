## gregoiremahon - 2022/04/13

## Face detection

## Importing dependencies
import cv2

## face_cascade XML file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')

bw_threshold = 100 ## Entre 80 et 105 en fonction de la luminosité

font = cv2.FONT_HERSHEY_DUPLEX
org = (50, 100)
face_font_color = (255, 0, 0) # BLUE COLOR
thickness = 3
font_scale = 4
cv2.namedWindow("local webcam (ESC to kill)")
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Conversion noir et blanc
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('black_and_white', black_and_white)
    ## Détection visage
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Pas de visage trouvé
    if(len(faces) == 0):
        cv2.putText(img, "No face found", org, font, font_scale, face_font_color, thickness, cv2.LINE_AA)
    else: # visage trouvé
        # Dessin du rectangle autour du visage
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 4)
            break

    ## Affichage image colorée finale
    cv2.imshow('Face Detection',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()
