import cv2
cap=cv2.VideoCapture(0)
filename="D:\\Serhatin_hayali2.avi"
codec=cv2.VideoWriter_fourcc('W','M','V','2')
framerate=30
resulution=(1920,1080)
VideoFile_Output= cv2.VideoWriter(filename,codec,framerate,resulution)


while True:
    fet,frame=cap.read()
    frame=cv2.flip(frame,1)
    VideoFile_Output.write(frame)
    cv2.imshow("Serhat_Window",frame)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
    if fet==False: # Cunku hepsini okuduktan sonra elimde eger bir resim kalmaz ise otomatik hata verecektir.
        break
cap.release()
VideoFile_Output.realease()
cv2.destroyAllWindows()
