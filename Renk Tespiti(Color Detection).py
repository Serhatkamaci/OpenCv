import cv2
import numpy as np

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:

    ret,frame=cap.read()
    frame=cv2.flip(frame,1)

    hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red= np.array([170,175,20])

    upper_red=np.array([180,255,255])

    red_mask=cv2.inRange(hsv_frame,lower_red,upper_red)
    red=cv2.bitwise_and(frame,frame,mask=red_mask) # Burada kirmizi rengi gostereyim.


    cv2.imshow("Detecting Color",frame)
    cv2.imshow("Red_mask",red_mask)
    cv2.imshow("Red",red)


    if cv2.waitKey(1) & 0XFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()