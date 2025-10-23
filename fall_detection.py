"""
Sistema de Detecção de Quedas para Raspberry Pi
Compatível com YOLOv8 exportado para TFLite
"""

import cv2
import numpy as np
import tensorflow as tf
import time
from threading import Thread
import queue
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FallDetector:
    def __init__(self, model_path="best_integer_quant.tflite", camera_source=0):
        """
        Inicializa o detector de quedas
        Args:
            model_path: Caminho para o modelo TFLite
            camera_source: 0 para webcam USB ou URL RTSP para câmera IP
        """
        # Configurações do modelo
        self.MODEL_PATH = model_path
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.CONFIDENCE_THRESHOLD = 0.4
        self.IOU_THRESHOLD = 0.45
        
        self.CLASS_NAMES = {
            0: "fallen",  # Pessoa caída
            1: "person"         # Pessoa em pé
        }
        
        self.CAMERA_SOURCE = camera_source
        
        self.fall_counter = 0
        self.FALL_CONFIRM_FRAMES = 10
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        self.running = False
        self.fps = 0
        self.last_alert_time = 0
        self.ALERT_COOLDOWN = 30
        
        # Inicializar TFLite
        self.setup_model()
        
    def setup_model(self):
        """Carrega e configura o modelo TFLite"""
        try:
            # Carregar o interpretador TFLite
            self.interpreter = tf.lite.Interpreter(model_path=self.MODEL_PATH)
            self.interpreter.allocate_tensors()
            
            # Obter detalhes de entrada e saída
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"✅ Modelo carregado: {self.MODEL_PATH}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocessa a imagem para o formato esperado pelo modelo
        """
        # Redimensionar para o tamanho de entrada
        resized = cv2.resize(image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))

        input_data = np.expand_dims(resized, axis=0)
        
        # Verificar o tipo de dados esperado
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
        elif input_dtype == np.float32:
            input_data = (input_data / 255.0).astype(np.float32)
            
        return input_data
    
    def process_yolo_output(self, output_data, original_shape):
        """
        Processa a saída do YOLOv8 TFLite e retorna as detecções.
        Formato de saída do TFLite após a transposição: [num_boxes, 4_coords + num_classes]
        Ex: [8400, 6] onde 6 é [x_center, y_center, w, h, class0_prob, class1_prob]
        """
        # Remove a dimensão do lote e transpõe a matriz
        # De [1, 6, 8400] para [8400, 6]
        output_data = output_data[0].T

        boxes = []
        scores = []
        class_ids = []
        h_orig, w_orig = original_shape[:2]

        for row in output_data:
            # Extrair probabilidades de classe e encontrar a de maior valor
            class_probs = row[4:]
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]

            if confidence > self.CONFIDENCE_THRESHOLD:
                # Coordenadas da caixa
                x_center, y_center, w, h = row[:4]

                # Converter coordenadas normalizadas para pixels no tamanho da imagem original
                x1 = int((x_center - w / 2) * w_orig)
                y1 = int((y_center - h / 2) * h_orig)
                x2 = int((x_center + w / 2) * w_orig)
                y2 = int((y_center + h / 2) * h_orig)
                
                boxes.append([x1, y1, x2 - x1, y2 - y1]) # NMS do OpenCV espera [x, y, w, h]
                scores.append(float(confidence))
                class_ids.append(class_id)

        # Aplicar Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.CONFIDENCE_THRESHOLD, self.IOU_THRESHOLD)
        
        detections = []
        if indices is not None and len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'bbox': [x, y, x + w, y + h], # Converter de volta para [x1, y1, x2, y2]
                    'confidence': scores[i],
                    'class_id': class_ids[i],
                    'class_name': self.CLASS_NAMES.get(class_ids[i], f"class_{class_ids[i]}")
                })

        return detections
    
    def check_fall_confirmation(self, detections):
        """
        Verifica se uma queda foi confirmada baseado em múltiplos frames
        """
        fall_detected = any(d['class_id'] == 0 for d in detections)  # 0 = fallen_person
        
        if fall_detected:
            self.fall_counter += 1
            if self.fall_counter >= self.FALL_CONFIRM_FRAMES:
                return True
        else:
            # Decay gradual do contador
            self.fall_counter = max(0, self.fall_counter - 2)
        
        return False
    
    def send_alert(self, frame):
        """Envia alerta de queda detectada"""
        current_time = time.time()
        
        # Verificar cooldown para evitar spam de alertas
        if current_time - self.last_alert_time < self.ALERT_COOLDOWN:
            return
        
        self.last_alert_time = current_time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Salvar imagem da queda
        filename = f"queda_detectada_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.warning(f"🚨 ALERTA: Queda confirmada! Imagem salva: {filename}")
        
        # TODO: Adicionar aqui integração com sistemas de notificação
        # - Enviar SMS via Twilio
        # - Enviar notificação push
        # - Fazer POST para API
        # - Enviar email
        # - Acionar alarme sonoro
    
    def draw_detections(self, frame, detections, fall_confirmed):
        """Desenha as detecções no frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Escolher cor baseado na classe
            if det['class_id'] == 0:  # fallen_person
                color = (0, 0, 255)  # Vermelho
                label = f"QUEDA! {conf:.2f}"
            else:  # person
                color = (0, 255, 0)  # Verde
                label = f"OK {conf:.2f}"
            
            # Desenhar bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Adicionar label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.rectangle(frame, (x1, y_label - label_size[1] - 5),
                         (x1 + label_size[0], y_label + 5), color, -1)
            cv2.putText(frame, label, (x1, y_label),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Adicionar informações do sistema
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Fall Counter: {self.fall_counter}/{self.FALL_CONFIRM_FRAMES}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if fall_confirmed:
            cv2.putText(frame, "!!! ALERTA ENVIADO !!!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame
    
    def inference_thread(self):
        """Thread separada para inferência"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Preprocessar imagem
                input_data = self.preprocess_image(frame)
                
                # Fazer inferência
                start_time = time.time()
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                inference_time = time.time() - start_time
                
                # Calcular FPS
                self.fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Obter saída
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

                # Processar detecções
                detections = self.process_yolo_output(output_data, frame.shape)
                
                # Colocar resultado na fila
                if self.result_queue.full():
                    self.result_queue.get()
                self.result_queue.put((frame, detections))
    
    def run(self):
        """Loop principal do sistema"""
        # Conectar à câmera
        cap = cv2.VideoCapture(self.CAMERA_SOURCE)
        if not cap.isOpened():
            logger.error(f"❌ Erro: Não foi possível conectar à câmera")
            return
        
        # Configurar câmera para melhor performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info("✅ Câmera conectada. Iniciando detecção...")
        
        # Iniciar thread de inferência
        self.running = True
        inference = Thread(target=self.inference_thread, daemon=True)
        inference.start()
        
        # Variáveis para skip de frames
        frame_skip = 2
        frame_count = 0
        last_result = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame perdido, tentando reconectar...")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Skip frames para melhorar performance
                if frame_count % frame_skip == 0:
                    if self.frame_queue.full():
                        self.frame_queue.get()
                    self.frame_queue.put(frame.copy())
                
                # Verificar novos resultados
                if not self.result_queue.empty():
                    last_result = self.result_queue.get()
                
                # Processar e exibir resultado
                display_frame = frame.copy()
                if last_result is not None:
                    _, detections = last_result
                    
                    # Verificar confirmação de queda
                    fall_confirmed = self.check_fall_confirmation(detections)
                    if fall_confirmed:
                        self.send_alert(frame)
                    
                    # Desenhar detecções
                    display_frame = self.draw_detections(display_frame, detections, fall_confirmed)
                
                # Exibir frame
                cv2.imshow("Detecção de Quedas - Raspberry Pi", display_frame)
                
                # Verificar tecla de saída
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuário")
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Sistema finalizado")


def main():
    """Função principal"""
    # Configurações - AJUSTE AQUI CONFORME NECESSÁRIO
    MODEL_PATH = "best_integer_quant.tflite"  # Caminho do seu modelo
    
    # Para câmera USB local use 0, para câmera IP use a URL RTSP
    # CAMERA_SOURCE = 0  # Webcam USB
    # CAMERA_SOURCE = "rtsp://usuario:senha@192.168.1.100:554/stream1"  # Câmera IP
    CAMERA_SOURCE = 0
    
    detector = FallDetector(
        model_path=MODEL_PATH,
        camera_source=CAMERA_SOURCE
    )
    
    try:
        detector.run()
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
