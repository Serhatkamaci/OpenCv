
import cv2
import numpy as np

cap = cv2.VideoCapture("D:/OpenCV/Yolo Algoritmasi/yolo_pretrainedvideo/Yolo_Videos/people.mp4")

# Simdi tek bir resim yok ben her bir frame'in boy ve enini bulmam gerekiyor.

while True:

    ret, frame = cap.read()
    frame_width = frame.shape[1]  # Genislik
    frame_height = frame.shape[0]  # Yukseklik

    # Simdi her bir frame'in en ve boy'unu buldum . Simdi her bir frame i blob formata cevirmem gerekiyor.
    frame_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    #Goruntuyu modele verebilmek icin cevirmemiz gereken format bu da 4 boyutluk tensör
    
    labels = ["person", "children", "the stroller", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
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
    colors = np.array(colors)  # Tek bir array'e cevirdim
    colors = np.tile(colors, (18, 1))

    model = cv2.dnn.readNetFromDarknet("D:/OpenCV/Yolo Algoritmasi/Pratrained_models/yolov3.cfg",
                                       "D:/OpenCV/Yolo Algoritmasi/Pratrained_models/yolov3.weights")
    layers = model.getLayerNames()
    output_layer = [layers[i - 1] for i in model.getUnconnectedOutLayers()]
    model.setInput(frame_blob)

    detection_layers = model.forward(output_layer)

    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]  # Bunun mantigini kursat abiye sor deki scores liste mi?

            predicted_id = np.argmax(scores)

            confidence = scores[predicted_id]

            if confidence > 0.80:
                label = labels[predicted_id]

                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])

                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width / 2))
                start_y = int(box_center_y + (box_height / 2))
                end_x = int(box_center_x + (box_width / 2))
                end_y = int(box_center_y - (box_height / 2))

                box_color = colors[predicted_id]

                box_color = [int(each) for each in box_color]

                label = "{}: {:.2f}%".format(label, confidence * 100)
                print("Predicted Object {}".format(label))

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)

                cv2.putText(frame, label, (start_x, start_y - 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 1)

    cv2.imshow("Detection Window", frame)
    if (cv2.waitKey(0) & 0XFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()



