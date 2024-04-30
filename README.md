# Aplicación de Detección de Objetos con Flask

Esta aplicación Flask utiliza un modelo pre-entrenado de YOLOv3 para detectar objetos en imágenes que se envían a través de una solicitud HTTP POST. La aplicación devuelve los resultados de detección en formato JSON.

## Configuración del Proyecto

### Instalación de Dependencias

Asegúrate de tener Python instalado en tu sistema. Luego, puedes instalar las dependencias del proyecto ejecutando:

bash
pip install flask opencv-python
Descarga del Modelo Pre-Entrenado
Descarga los archivos yolov3.weights, yolov3.cfg y coco.names del modelo pre-entrenado de YOLOv3 y colócalos en la carpeta model en la raíz del proyecto.

## Uso
Para ejecutar la aplicación, simplemente ejecuta el script app.py:

bash
Copy code
python app.py
Esto iniciará el servidor Flask en modo de depuración y estará disponible en http://localhost:5000.

Enviar una Solicitud de Detección de Objetos
Puedes enviar una imagen al servidor para su detección de la siguiente manera:

bash
Copy code
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/detect
Reemplaza path/to/image.jpg con la ruta de la imagen que deseas enviar para su detección.
