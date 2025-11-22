"""
Sistema de Detecção de Quedas para Raspberry Pi
Usando OpenCV DNN com modelo ONNX (YOLOv8)
Otimizado para Raspberry Pi 3B
"""

import cv2
import numpy as np
import time
from threading import Thread
import queue
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FallDetector:
    def __init__(self, model_path="best.onnx", camera_source=0):
        """
        Inicializa o detector de quedas
        Args:
            model_path: Caminho para o modelo ONNX
            camera_source: 0 para webcam USB ou URL RTSP para câmera IP
        """
        # Configurações do modelo
        self.MODEL_PATH = model_path
        self.INPUT_WIDTH = 320
        self.INPUT_HEIGHT = 320
        self.CONFIDENCE_THRESHOLD = 0.4
        self.IOU_THRESHOLD = 0.45

        self.CLASS_NAMES = {
            0: "fallen",  # Pessoa caída
            1: "person"   # Pessoa em pé
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

        # Inicializar OpenCV DNN
        self.setup_model()

    def setup_model(self):
        """Carrega e configura o modelo ONNX com OpenCV DNN"""
        try:
            # Carregar modelo ONNX
            self.net = cv2.dnn.readNetFromONNX(self.MODEL_PATH)

            # Configurar backend e target para melhor performance no Raspberry Pi
            # Usar CPU otimizada
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # logger.info(f"✅ Modelo ONNX carregado: {self.MODEL_PATH}")
            # logger.info(f"📐 Input size: {self.INPUT_WIDTH}x{self.INPUT_HEIGHT}")

        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise

    def preprocess_image(self, image):
        """
        Preprocessa a imagem para o formato esperado pelo YOLOv8
        """
        # Criar blob (redimensiona, normaliza e reorganiza dimensões)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1/255.0,  # Normalizar para [0, 1]
            size=(self.INPUT_WIDTH, self.INPUT_HEIGHT),
            mean=(0, 0, 0),
            swapRB=True,  # BGR para RGB
            crop=False
        )
        return blob

    def process_yolo_output(self, output_data, original_shape):
        """
        Processa a saída do YOLOv8 ONNX e retorna as detecções.
        Formato de saída: [1, num_classes + 4, num_boxes]
        """
        # Transpor para [num_boxes, num_classes + 4]
        output_data = output_data[0].T

        boxes = []
        scores = []
        class_ids = []
        h_orig, w_orig = original_shape[:2]

        # Fatores de escala
        x_factor = w_orig / self.INPUT_WIDTH
        y_factor = h_orig / self.INPUT_HEIGHT

        for row in output_data:
            # Extrair probabilidades de classe
            class_probs = row[4:]
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]

            if confidence > self.CONFIDENCE_THRESHOLD:
                # Coordenadas da caixa (formato YOLO: x_center, y_center, width, height)
                x_center, y_center, w, h = row[:4]

                # Converter para coordenadas absolutas
                x_center *= x_factor
                y_center *= y_factor
                w *= x_factor
                h *= y_factor

                # Converter para formato [x1, y1, w, h]
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)

                boxes.append([x1, y1, int(w), int(h)])
                scores.append(float(confidence))
                class_ids.append(int(class_id))

        # Aplicar Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.CONFIDENCE_THRESHOLD,
            self.IOU_THRESHOLD
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'bbox': [x, y, x + w, y + h],  # [x1, y1, x2, y2]
                    'confidence': scores[i],
                    'class_id': class_ids[i],
                    'class_name': self.CLASS_NAMES.get(class_ids[i], f"class_{class_ids[i]}")
                })

        return detections

    def check_fall_confirmation(self, detections):
        """
        Verifica se uma queda foi confirmada baseado em múltiplos frames
        """
        fall_detected = any(
            d['class_id'] == 0 for d in detections)  # 0 = fallen

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
        # - Acionar alarme sonoro no Raspberry Pi (GPIO)

    def draw_detections(self, frame, detections, fall_confirmed):
        """Desenha as detecções no frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            # Escolher cor baseado na classe
            if det['class_id'] == 0:  # fallen
                color = (0, 0, 255)  # Vermelho
                label = f"QUEDA! {conf:.2f}"
            else:  # person
                color = (0, 255, 0)  # Verde
                label = f"OK {conf:.2f}"

            # Desenhar bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Adicionar label
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
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
        # logger.info("🔄 Thread de inferência iniciada")
        # frame_count = 0
        while self.running:
            if not self.frame_queue.empty():
                # frame_count += 1
                # logger.info(f"📸 Processando frame #{frame_count}")
                frame = self.frame_queue.get()

                # Preprocessar imagem
                blob = self.preprocess_image(frame)
                # logger.info(f"✅ Blob criado - Shape: {blob.shape}")

                # Fazer inferência
                start_time = time.time()
                self.net.setInput(blob)
                # logger.info("⚙️ Executando inferência...")
                output_data = self.net.forward()
                inference_time = time.time() - start_time

                # Calcular FPS
                self.fps = 1.0 / inference_time if inference_time > 0 else 0

                # logger.info(f"⏱️ Tempo: {inference_time:.3f}s | FPS: {self.fps:.1f}")
                # logger.info(f"📊 Output shape: {output_data.shape}")

                # Processar detecções
                detections = self.process_yolo_output(output_data, frame.shape)
                # logger.info(f"🎯 Detecções encontradas: {len(detections)}")

                # Colocar resultado na fila
                if self.result_queue.full():
                    self.result_queue.get()
                self.result_queue.put((frame, detections))
            else:
                time.sleep(0.001)  # Pequeno sleep para não travar CPU

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
        logger.info("⌨️  Pressione 'q' para sair")

        # Iniciar thread de inferência
        self.running = True
        inference = Thread(target=self.inference_thread, daemon=True)
        inference.start()

        window_name = "Detecção de Quedas - Raspberry Pi"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        # Variáveis para skip de frames
        frame_skip = 15  # Processar 1 a cada 2 frames
        frame_count = 0
        last_result = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️  Frame perdido, tentando reconectar...")
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
                    display_frame = self.draw_detections(
                        display_frame, detections, fall_confirmed)

                # Exibir frame
                cv2.imshow("Detecção de Quedas - Raspberry Pi", display_frame)

                # Verificar tecla de saída
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("🛑 Interrompido pelo usuário")
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("✅ Sistema finalizado")


def main():
    """Função principal"""
    # Configurações - AJUSTE AQUI CONFORME NECESSÁRIO
    MODEL_PATH = "best.onnx"  # Caminho do seu modelo ONNX

    # Para câmera USB local use 0, para câmera IP use a URL RTSP
    # CAMERA_SOURCE = 0  # Webcam USB
    # CAMERA_SOURCE = "rtsp://usuario:senha@192.168.1.100:554/stream1"  # Câmera IP
    CAMERA_SOURCE = 0

    logger.info("=" * 60)
    logger.info("🚨 Sistema de Detecção de Quedas - Raspberry Pi")
    logger.info("=" * 60)

    detector = FallDetector(
        model_path=MODEL_PATH,
        camera_source=CAMERA_SOURCE
    )

    try:
        detector.run()
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
