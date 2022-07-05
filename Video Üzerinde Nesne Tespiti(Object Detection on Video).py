import cv2
import numpy as np

ids_list = []
boxes_list = []
confidences_list = []

cap=cv2.VideoCapture("D:\\video3.mp4")
while True:

    ret,frame=cap.read()

    frame_width=frame.shape[1]
    frame_height=frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

    labels = ["person", "children", "backpack", "smartphone", "car", "motorcycle", "airplane", "bus", "train", "truck",
                      "boat",
                      "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                      "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                      "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                      "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                      "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                      "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                      "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                      "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

    colors = ["0,255,255", "0,0,255", "255,0,0", "255,255,0", "0,255,0"]
    colors = [np.array(i.split(",")).astype("int") for i in colors]
    colors = np.array(colors) 
    colors = np.tile(colors, (18, 1)) 

    model=cv2.dnn.readNetFromDarknet("D:\OpenCV\Yolo Algoritmasi\Pratrained_models\yolov3.cfg","D:\OpenCV\Yolo Algoritmasi\Pratrained_models\yolov3.weights") # Veri yapisini olusturdum.

    layers=model.getLayerNames()

    output_layers=[layers[i-1] for i in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)
    detection_layers=model.forward(output_layers)




    ids_list=[]
    boxes_list=[]
    confidences_list=[]



    for i in detection_layers:
        for j in i:

            score=j[5:]

            predicted_id=np.argmax(score)
            confidence=score[predicted_id]

            if confidence > 0.85:

                label=labels[predicted_id]
                bounding_box=j[0:4]

                bounding_box = j[0:4]*np.array([frame_width,frame_height,frame_width,frame_height])
                (box_centerx,box_centery,box_width,box_height)=bounding_box.astype("int")


                start_x=int(box_centerx-(box_width/2))
                start_y = int(box_centery - (box_width / 2))

         

                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)])



    max_ids=cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4) # Bu bir listedir.
    print(boxes_list)
    print(max_ids)
    print(confidences_list)


    for max_id in max_ids:

        max_class_id=max_id
        box=boxes_list[max_class_id]

        start_x=box[0]
        start_y=box[1]
        box_width=box[2]
        box_height=box[3]



        label=labels[max_class_id]
        confidence=confidences_list[max_class_id]



        end_x = start_x+ box_width
        end_y = start_y + box_height



        box_color=colors[max_class_id]
        box_color = [int(i) for i in box_color]

        label="{}: {:.2f}%".format(label,confidence*100)
        print("Predicted Object {}".format(label))

        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color,1)
        cv2.putText(frame,label,(start_x,start_y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,box_color)

    cv2.imshow("Detection_layers",frame)
    if cv2.waitKey(1) & 0XFF==ord("q"):
        break
    cap.release()
    cv2.destroyAllWindows()
