"""
Utilitários para restauração de fotos antigas.

Fornece funções auxiliares para:
- Detecção automática de danos (rachaduras, rasgos, manchas)
- Geração de máscaras a partir de imagens danificadas
- Pré-processamento de fotos antigas
- Pipeline completo de restauração
"""

from PIL import Image
import numpy as np
from typing import Optional, Tuple
import cv2


def detect_damage_mask(image: Image.Image, 
                       method: str = 'threshold',
                       sensitivity: float = 0.7) -> Image.Image:
    """
    Detecta automaticamente áreas danificadas em fotos antigas.
    
    Args:
        image: PIL Image da foto antiga
        method: Método de detecção ('threshold', 'edges', 'combined')
        sensitivity: Sensibilidade da detecção (0.0 a 1.0)
                    Menor = detecta menos danos, Maior = detecta mais danos
    
    Returns:
        PIL Image da máscara (branco = área danificada, preto = área intacta)
    
    Métodos:
    - 'threshold': Detecta pixels muito claros ou escuros (rachaduras brancas, manchas escuras)
    - 'edges': Detecta bordas irregulares (rasgos, buracos)
    - 'combined': Combina threshold + edges (recomendado para fotos muito danificadas)
    
    Exemplos de uso:
        # Foto com rachaduras brancas
        mask = detect_damage_mask(image, method='threshold', sensitivity=0.8)
        
        # Foto rasgada com buracos
        mask = detect_damage_mask(image, method='edges', sensitivity=0.6)
        
        # Foto muito danificada (múltiplos tipos de dano)
        mask = detect_damage_mask(image, method='combined', sensitivity=0.7)
    """
    # Converter para numpy array
    img_array = np.array(image)
    
    # Converter para escala de cinza
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if method in ['threshold', 'combined']:
        # Detectar pixels muito claros (rachaduras, rasgos brancos)
        threshold_high = int(255 * (1 - (1 - sensitivity) * 0.3))  # Ex: 0.7 → 235
        bright_damage = gray > threshold_high
        
        # Detectar pixels muito escuros (manchas, áreas ausentes)
        threshold_low = int(255 * ((1 - sensitivity) * 0.3))  # Ex: 0.7 → 76
        dark_damage = gray < threshold_low
        
        mask[bright_damage | dark_damage] = 255
        
        # Dilatar para conectar áreas próximas
        kernel_size = max(3, int(5 * sensitivity))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    if method in ['edges', 'combined']:
        # Detectar bordas irregulares (rasgos, buracos)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilatar bordas detectadas
        kernel_size = max(5, int(7 * sensitivity))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        if method == 'combined':
            mask = cv2.bitwise_or(mask, edges_dilated)
        else:
            mask = edges_dilated
    
    # Aplicar morfologia para limpar ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove ruído
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Preenche buracos pequenos
    
    # Converter de volta para PIL Image
    mask_image = Image.fromarray(mask, mode='L')
    
    return mask_image


def preprocess_old_photo(image: Image.Image, 
                         denoise: bool = True,
                         adjust_contrast: bool = True) -> Image.Image:
    """
    Pré-processa foto antiga antes da restauração.
    
    Args:
        image: PIL Image da foto antiga
        denoise: Se True, aplica denoising suave
        adjust_contrast: Se True, ajusta contraste automaticamente
    
    Returns:
        PIL Image pré-processada
    """
    img_array = np.array(image)
    
    # Denoising suave
    if denoise:
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, h=10, hColor=10, 
                                                      templateWindowSize=7, searchWindowSize=21)
    
    # Ajuste de contraste adaptativo
    if adjust_contrast:
        # Converter para LAB para ajustar apenas luminância
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Recombinar e converter de volta para RGB
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(img_array)


def restore_photo(image: Image.Image,
                  mask: Optional[Image.Image] = None,
                  model = None,
                  auto_detect_damage: bool = True,
                  detection_method: str = 'combined',
                  sensitivity: float = 0.7,
                  preprocess: bool = True,
                  **kwargs) -> Tuple[Image.Image, Image.Image]:
    """
    Pipeline completo de restauração de fotos antigas.
    
    Args:
        image: PIL Image da foto antiga
        mask: PIL Image da máscara (opcional, será gerada automaticamente se None)
        model: Modelo de inpainting a ser usado (Kandinsky, StableDiffusion, etc.)
        auto_detect_damage: Se True, gera máscara automaticamente
        detection_method: Método de detecção ('threshold', 'edges', 'combined')
        sensitivity: Sensibilidade da detecção (0.0 a 1.0)
        preprocess: Se True, aplica pré-processamento
        **kwargs: Argumentos adicionais para o modelo de inpainting
    
    Returns:
        Tupla (imagem_restaurada, máscara_usada)
    
    Exemplo:
        from canon.T4.process import KandinskyInpainting
        
        model = KandinskyInpainting()
        model.load_model()
        
        # Restauração automática (detecta danos automaticamente)
        restored, mask_used = restore_photo(
            old_photo, 
            model=model,
            auto_detect_damage=True,
            sensitivity=0.8
        )
        
        # Restauração com máscara manual
        restored, mask_used = restore_photo(
            old_photo,
            mask=my_mask,
            model=model,
            preprocess=True
        )
    """
    # Pré-processamento
    if preprocess:
        image = preprocess_old_photo(image)
    
    # Gerar máscara automaticamente se necessário
    if mask is None and auto_detect_damage:
        print(f"🔍 Detectando danos automaticamente (método: {detection_method}, sensibilidade: {sensitivity})")
        mask = detect_damage_mask(image, method=detection_method, sensitivity=sensitivity)
        
        # Verificar se encontrou danos
        mask_array = np.array(mask)
        damage_pixels = np.sum(mask_array > 127)
        total_pixels = mask_array.size
        damage_percentage = (damage_pixels / total_pixels) * 100
        
        print(f"   Área danificada detectada: {damage_percentage:.2f}%")
        
        if damage_percentage < 0.5:
            print("   ⚠️  Poucos danos detectados. Considere ajustar a sensibilidade.")
        elif damage_percentage > 50:
            print("   ⚠️  Muitos danos detectados. Considere reduzir a sensibilidade.")
    
    if mask is None:
        raise ValueError("Máscara não fornecida e auto_detect_damage=False")
    
    # Executar inpainting
    if model is None:
        raise ValueError("Modelo não fornecido. Use model=KandinskyInpainting() ou similar.")
    
    print("🎨 Restaurando foto...")
    restored = model.inpaint(image, mask, **kwargs)
    
    print("✅ Restauração concluída!")
    
    return restored, mask


def create_manual_mask_interactive(image: Image.Image) -> Image.Image:
    """
    Cria máscara manualmente usando interface interativa do OpenCV.
    
    Args:
        image: PIL Image da foto
    
    Returns:
        PIL Image da máscara desenhada
    
    Instruções:
    - Clique e arraste para desenhar áreas danificadas
    - Pressione 'r' para resetar
    - Pressione 'q' para finalizar
    
    Nota: Requer ambiente com display (não funciona em servidores sem GUI)
    """
    img_array = np.array(image)
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    drawing = False
    brush_size = 20
    
    def draw_mask(event, x, y, flags, param):
        nonlocal drawing, mask
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(mask, (x, y), brush_size, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    window_name = 'Desenhe as áreas danificadas | R=resetar | Q=finalizar'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_mask)
    
    while True:
        # Sobrepor máscara na imagem
        display = img_array.copy()
        display[mask > 0] = [255, 0, 0]  # Vermelho nas áreas marcadas
        display = cv2.addWeighted(img_array, 0.7, display, 0.3, 0)
        
        cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    cv2.destroyAllWindows()
    
    return Image.fromarray(mask, mode='L')
