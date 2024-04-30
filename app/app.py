from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/detect', methods=['POST'])
def detect():
    # Recibir la imagen
    image_data = request.get_data()
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decodificar la matriz numpy en una imagen OpenCV
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Cargar el modelo pre-entrenado de COCO
    net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
    classes = []
    with open("model/coco.names", "r") as f:
        classes = f.read().splitlines()

    # Preprocesamiento de la imagen
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obtener las capas de salida
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Procesar las detecciones
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar non-max suppression para eliminar detecciones redundantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Construir el objeto JSON con las detecciones
    objects_detected = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            objects_detected.append({
                'object': label,
                'position': {'x': x, 'y': y, 'width': w, 'height': h}
            })

    return jsonify(objects_detected)

if __name__=='__main__':
    app.run(debug=True, port=5000)
