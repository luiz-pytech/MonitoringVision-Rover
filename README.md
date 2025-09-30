# MonitoringVision-Rover - Projeto de extensão

## Detecção de Quedas em Tempo Real com YOLOv8 e Raspberry Pi

![Badge de Licença](https://img.shields.io/badge/license-MIT-blue.svg)
![Badge de Python](https://img.shields.io/badge/Python-3.9%2B-blue)

## 📖 Descrição

Este projeto implementa um sistema de detecção de quedas humanas em tempo real. Foi treinado um modelo de visão computacional **YOLOv8n** customizado para identificar duas classes: `person` (pessoa) e `fall` (queda).

O desenvolvimento e treinamento foram realizados no Google Colab, aproveitando a aceleração por GPU disponibilizada pelo Google. O modelo final foi otimizado para o formato **TFLite (INT8)**, permitindo uma inferência eficiente e de alta performance em dispositivos embarcados de baixa potência, como o **Raspberry Pi**.

## ✨ Funcionalidades

-   **Detecção em Tempo Real:** Análise de streams de vídeo para identificação imediata de quedas.
-   **Modelo Leve e Rápido:** Utiliza a arquitetura YOLOv8n, ideal para performance em hardware limitado.
-   **Alta Precisão:** O modelo foi treinado em um dataset público (UR Fall Detection) e validado em outro (GMDCSA24)
-   **Otimizado para Embarcados:** Exportado para TFLite com quantização INT8, garantindo baixa latência e uso eficiente de CPU no Raspberry Pi.
-   **Fácil Implantação:** Inclui um script Python pronto para ser executado no Raspberry Pi com uma câmera IP (Wi-Fi) ou USB.

## 🛠️ Tecnologias e Ferramentas

-   **Linguagem:** Python 3
-   **Frameworks de IA:** Ultralytics (YOLOv8), PyTorch
-   **Otimização:** TensorFlow Lite (TFLite Runtime)
-   **Processamento de Imagem:** OpenCV
-   **Ambiente de Treinamento:** Google Colab
-   **Hardware de Implantação:** Raspberry Pi 4 (ou superior)

## 📁 Estrutura do Projeto

```
.
├── YOLO_Monitoring_Rover.ipynb   # Notebook do Colab para treinamento e exportação
├── fall_detection.py             # Script de inferência para o Raspberry Pi
├── best_integer_quant.tflite     # Modelo TFLite otimizado para implantação no sistema embarcado
├── /results                      # Pasta com resultados do treinamento
└── README.md                     # Este arquivo
```

## ⚙️ Instalação e Configuração

O projeto é dividido em duas fases: Treinamento e Implantação.

### Fase 1: Treinamento (Google Colab)

1.  **Abra o Notebook:** Faça o upload e abra o arquivo `Fall_Detection_Training.ipynb` no Google Colab.
2.  **Configure a API Key:** Adicione sua chave da Roboflow nos "Secrets" do Colab para baixar o dataset de forma segura.
3.  **Habilite a GPU:** No menu do Colab, vá em `Ambiente de execução > Alterar tipo de ambiente de execução` e selecione "T4 GPU".
4.  **Execute as Células:** Rode as células em sequência para instalar as dependências, baixar o dataset, treinar o modelo e exportar o arquivo `best_integer_quant.tflite`.
5.  **Faça o Download:** Ao final, baixe o modelo `best_integer_quant.tflite` gerado.

### Fase 2: Implantação (Raspberry Pi)

1.  **Transfira os Arquivos:** Copie o modelo `best_integer_quant.tflite` e o script `detector_de_quedas.py` para o seu Raspberry Pi.
2.  **Instale as Dependências no Pi:** Abra o terminal no Raspberry Pi e execute os comandos:
    ```bash
    pip3 install tflite-runtime
    pip3 install opencv-python
    pip3 install numpy
    ```

## ▶️ Como Usar

1.  **Edite o Script:** Abra o arquivo `detector_de_quedas.py` e altere a variável `CAMERA_URL` para a URL do stream da sua câmera Wi-Fi (ex: `rtsp://...`) ou para `0` se estiver usando uma câmera USB.
2.  **Execute a Detecção:** No terminal do Raspberry Pi, navegue até a pasta onde estão os arquivos e execute:
    ```bash
    python3 detector_de_quedas.py
    ```
3.  Pressione a tecla `q` na janela de visualização para encerrar o programa.

## 📊 Resultados

O modelo alcançou uma performance excelente durante a validação, com um **mAP50 de aproximadamente 97.5%**.

![Gráficos de Treinamento](/results/results.png)

---
*Criado por Luiz Felipe*