# Correção do Bug na Análise de Qualidade Epipolar

## Problema Identificado

A função `calculate_epipolar_quality_metrics` está falhando com o erro:
```
index 1 is out of bounds for axis 1 with size 1
```

E gerando muitos warnings sobre pontos com "formato inesperado" como:
```
Warning: ponto com formato inesperado: [[317. 676.]]
```

## Causa do Problema

Os pontos estão chegando com shape `(N, 1, 2)` ao invés de `(N, 2)`. O código atual tenta acessar `pt[1]` mas o ponto tem formato `[[x, y]]` ao invés de `[x, y]`.

## Solução

Substitua a função `calculate_epipolar_quality_metrics` no notebook S3 pela versão corrigida abaixo:

```python
def calculate_epipolar_quality_metrics(F, pts1, pts2, mask_F):
    """
    Calcula métricas avançadas de qualidade para geometria epipolar
    """
    quality_metrics = {}
    
    # Pontos inliers
    pts1_inliers = pts1[mask_F.ravel() == 1]
    pts2_inliers = pts2[mask_F.ravel() == 1]
    
    # Garante que os pontos estão no formato correto (N, 2)
    # CORREÇÃO: Trata tanto formato (N, 1, 2) quanto (N, 2)
    if pts1_inliers.ndim == 3 and pts1_inliers.shape[1] == 1:
        pts1_inliers = pts1_inliers.squeeze(1)  # Remove dimensão extra
    elif pts1_inliers.ndim == 1:
        pts1_inliers = pts1_inliers.reshape(-1, 2)
    
    if pts2_inliers.ndim == 3 and pts2_inliers.shape[1] == 1:
        pts2_inliers = pts2_inliers.squeeze(1)  # Remove dimensão extra
    elif pts2_inliers.ndim == 1:
        pts2_inliers = pts2_inliers.reshape(-1, 2)
    
    # Verifica se temos pontos válidos
    if len(pts1_inliers) == 0 or len(pts2_inliers) == 0:
        print(f"Warning: Nenhum ponto inlier encontrado")
        return {
            'mean_epipolar_error': float('inf'),
            'std_epipolar_error': 0,
            'median_epipolar_error': float('inf'),
            'max_epipolar_error': float('inf'),
            'condition_number': float('inf'),
            'spatial_coverage': 0,
            'spatial_dispersion': 0,
            'inlier_ratio': 0,
            'num_inliers': 0,
            'num_outliers': len(pts1)
        }
    
    # 1. Erro epipolar médio para inliers
    try:
        lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1,1,2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        
        # Distância ponto-linha epipolar
        epipolar_errors = []
        for i, (pt, line) in enumerate(zip(pts1_inliers, lines1)):
            # Distância do ponto à linha epipolar: |ax + by + c| / sqrt(a² + b²)
            a, b, c = line
            
            # CORREÇÃO: Acesso seguro às coordenadas
            if pt.shape == (2,):
                x, y = pt[0], pt[1]
            else:
                print(f"Warning: ponto com formato inesperado: {pt}, shape: {pt.shape}")
                continue
                
            error = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
            epipolar_errors.append(error)
        
        if epipolar_errors:
            quality_metrics['mean_epipolar_error'] = np.mean(epipolar_errors)
            quality_metrics['std_epipolar_error'] = np.std(epipolar_errors)
            quality_metrics['median_epipolar_error'] = np.median(epipolar_errors)
            quality_metrics['max_epipolar_error'] = np.max(epipolar_errors)
        else:
            # Valores padrão se não conseguir calcular
            quality_metrics['mean_epipolar_error'] = float('inf')
            quality_metrics['std_epipolar_error'] = 0
            quality_metrics['median_epipolar_error'] = float('inf')
            quality_metrics['max_epipolar_error'] = float('inf')
            
    except Exception as e:
        print(f"Erro ao calcular erro epipolar: {e}")
        quality_metrics['mean_epipolar_error'] = float('inf')
        quality_metrics['std_epipolar_error'] = 0
        quality_metrics['median_epipolar_error'] = float('inf')
        quality_metrics['max_epipolar_error'] = float('inf')
    
    # 2. Análise de condicionamento da matriz F
    try:
        U, S, Vt = np.linalg.svd(F)
        condition_number = S[0] / S[2] if S[2] > 1e-10 else float('inf')
        quality_metrics['condition_number'] = condition_number
    except:
        quality_metrics['condition_number'] = float('inf')
    
    # 3. Distribuição espacial dos inliers
    try:
        if len(pts1_inliers) > 0 and pts1_inliers.shape[1] == 2:
            # Extrai coordenadas x e y de forma segura
            x_coords = pts1_inliers[:, 0]
            y_coords = pts1_inliers[:, 1]
            
            x_coverage = (np.max(x_coords) - np.min(x_coords))
            y_coverage = (np.max(y_coords) - np.min(y_coords))
            quality_metrics['spatial_coverage'] = (x_coverage * y_coverage)
            
            # Dispersão espacial
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            quality_metrics['spatial_dispersion'] = np.std(distances)
        else:
            quality_metrics['spatial_coverage'] = 0
            quality_metrics['spatial_dispersion'] = 0
    except Exception as e:
        print(f"Erro ao calcular distribuição espacial: {e}")
        quality_metrics['spatial_coverage'] = 0
        quality_metrics['spatial_dispersion'] = 0
    
    # 4. Robustez do RANSAC
    inlier_ratio = np.sum(mask_F) / len(pts1)
    quality_metrics['inlier_ratio'] = inlier_ratio
    quality_metrics['num_inliers'] = np.sum(mask_F)
    quality_metrics['num_outliers'] = len(pts1) - np.sum(mask_F)
    
    return quality_metrics
```

## Principais Correções

1. **Detecção do formato 3D**: Adicionado check para `pts1_inliers.ndim == 3` e `pts1_inliers.shape[1] == 1`
2. **Uso do `squeeze(1)`**: Remove a dimensão extra de forma segura
3. **Acesso seguro às coordenadas**: Verifica `pt.shape == (2,)` antes de acessar `pt[0]` e `pt[1]`
4. **Tratamento de erros**: Blocos try-except para cada seção crítica
5. **Validação de pontos**: Verifica se existem pontos inliers antes de processar

## Como Aplicar a Correção

1. Localize a função `calculate_epipolar_quality_metrics` no notebook S3 (célula 13)
2. Substitua toda a função pela versão corrigida acima
3. Execute novamente a célula de análise detalhada de qualidade epipolar

## Resultado Esperado

Após a correção, você deve ver:
- Nenhum warning sobre "formato inesperado"
- Nenhum erro de índice
- Métricas de qualidade calculadas corretamente:
  ```
  Erro epipolar médio: X.XXX pixels
  Erro epipolar mediano: X.XXX pixels
  Condicionamento matriz F: XX.XX
  Cobertura espacial: XXXX pixels²
  Dispersão espacial: XX.XX pixels
  Taxa de inliers: XX.XX%
  ```

## Debug Adicional

Se ainda houver problemas, adicione esta linha antes da função para diagnosticar:

```python
# Debug - mostra formato dos pontos
print(f"Debug - pts1 shape: {pts1.shape}, pts2 shape: {pts2.shape}")
print(f"Debug - mask_F shape: {mask_F.shape}")
print(f"Debug - pts1 sample: {pts1[:2]}")
```

Isso ajudará a identificar qualquer formato não previsto dos dados.