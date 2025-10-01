import cv2
import numpy as np
import tensorflow as tf
tflite = tf.lite
import time

MODEL_PATH = "best_integer_quant.tflite"
CAMERA_URL = "rtsp://usuario:senha@ip_da_camera:554/stream1" # Se for local CAMERA_URL = 0

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.4 

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo TFLite carregado com sucesso.")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print(f"❌ Erro: Não foi possível conectar à câmera em {CAMERA_URL}")
    exit()

print("✅ Câmera conectada. Iniciando detecção...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do stream ou erro de frame. Tentando reconectar...")
        time.sleep(1)
        cap.release()
        cap = cv2.VideoCapture(CAMERA_URL)
        continue

    input_image = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype(np.uint8) # Para modelos quantizados INT8

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]


    for detection in output_data:
        box = detection[:4]
        confidence = detection[4] 
        class_id = np.argmax(detection[5:]) 

        if confidence > CONFIDENCE_THRESHOLD:
            
            h_orig, w_orig, _ = frame.shape
            x_center, y_center, w, h = box
            x1 = int((x_center - w / 2) * w_orig)
            y1 = int((y_center - h / 2) * h_orig)
            x2 = int((x_center + w / 2) * w_orig)
            y2 = int((y_center + h / 2) * h_orig)

            # Desenha no frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Classe {class_id}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("Detecção de Quedas - Raspberry Pi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()