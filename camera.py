import os
import time
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, img= cap.read()
    print("Reading camera...")
    img = cv2.flip(img,2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Face_Detect_Test', img)
    cv2.imwrite("face.png",img)
    print("Output PNG file...")
    time.sleep(0.2)
    os.system("rm -rf face.png")
    print("Delete PNG file...")
    
    k = cv2.waitKey(1)
    
    if k == 27:
        cv2.imwrite("face.png",img)
        break
    
cap.release()
cv2.destroyAllWindows()