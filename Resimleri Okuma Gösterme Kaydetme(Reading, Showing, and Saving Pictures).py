import cv2
img=cv2.imread("resim_okuma.jpg",0)
cv2.namedWindow("Serhat",cv2.WINDOW_NORMAL)
cv2.imshow("Serhat",img)
cv2.imwrite("D:\\serhat.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()