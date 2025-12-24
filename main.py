import cv2

face_cascade = cv2.CascadeClassifier('xmls/haarcascade_frontalface_default.xml')

cat_face_cascade = cv2.CascadeClassifier('xmls/haarcascade_frontalcatface.xml')

eye_cascade = cv2.CascadeClassifier('xmls/haarcascade_eye.xml') 

right_eye_cascade = cv2.CascadeClassifier('xmls/haarcascade_righteye_2splits.xml')

left_eye_cascade = cv2.CascadeClassifier('xmls/haarcascade_lefteye_2splits.xml')

smile_cascade = cv2.CascadeClassifier('xmls/haarcascade_smile.xml')

cap = cv2.VideoCapture(1)

while 1: 

    # reads frames from a camera
    ret, img = cap.read() 

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = cat_face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # eyes = eye_cascade.detectMultiScale(roi_gray) 

        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

        # left_eye = left_eye_cascade.detectMultiScale(roi_gray)

        # right_eye = right_eye_cascade.detectMultiScale(roi_gray)

        # for (ex,ey,ew,eh) in left_eye:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

        # for (ex,ey,ew,eh) in right_eye:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(127,0,255),2)

        # smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # for (sx,sy,sw,sh) in smiles:
        #     cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)

    # Display an image in a window
    cv2.imshow('img',img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()