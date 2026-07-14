# Correção do Bug na Análise de Qualidade da Pose

## Problema Identificado

A função `analyze_pose_quality` está falhando com o erro:
```
IndexError: index 1 is out of bounds for axis 0 with size 1
```

No código:
```python
pt1_norm = np.linalg.inv(K) @ np.array([pt1[0], pt1[1], 1])
```

## Causa do Problema

Similar ao bug anterior, os pontos estão chegando com formato `(1, 2)` ao invés de `(2,)`. O código tenta acessar `pt1[1]` mas o ponto tem formato `[[x, y]]` ao invés de `[x, y]`.

## Solução

Substitua a função `analyze_pose_quality` na **célula 21** do notebook S3 pela versão corrigida abaixo:

### Localização no Notebook

- **Célula**: 21 (Cell Id: #VSC-ae28fd24)
- **Linhas**: 391-479
- **Posição**: Segunda função na célula, depois dos comentários

### Código Corrigido

```python
def analyze_pose_quality(R, t, pts1, pts2, K, mask_E):
    """
    Analisa qualidade da pose recuperada
    """
    pose_metrics = {}
    
    # 1. Análise da baseline
    baseline_length = np.linalg.norm(t)
    pose_metrics['baseline_length'] = baseline_length
    
    # 2. Ângulo de rotação
    trace_R = np.trace(R)
    rotation_angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
    pose_metrics['rotation_angle_deg'] = np.degrees(rotation_angle)
    
    # 3. Análise da triangulação
    pts1_inliers = pts1[mask_E.ravel() == 1]
    pts2_inliers = pts2[mask_E.ravel() == 1]
    
    # CORREÇÃO: Garante formato correto dos pontos (N, 2)
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
        print(f"Warning: Nenhum ponto inlier encontrado para análise da pose")
        pose_metrics['mean_parallax_angle'] = 0
        pose_metrics['std_parallax_angle'] = 0
        pose_metrics['min_parallax_angle'] = 0
        # Continua com métricas básicas
    else:
        # Ângulos de paralaxe (amostra de 10 pontos para eficiência)
        parallax_angles = []
        sample_size = min(10, len(pts1_inliers))
        
        for i in range(sample_size):
            pt1 = pts1_inliers[i]
            pt2 = pts2_inliers[i]
            
            # CORREÇÃO: Acesso seguro às coordenadas
            try:
                if pt1.shape == (2,):
                    x1, y1 = pt1[0], pt1[1]
                else:
                    print(f"Warning: pt1 com formato inesperado: {pt1}, shape: {pt1.shape}")
                    continue
                    
                if pt2.shape == (2,):
                    x2, y2 = pt2[0], pt2[1]
                else:
                    print(f"Warning: pt2 com formato inesperado: {pt2}, shape: {pt2.shape}")
                    continue
                
                # Converte para coordenadas normalizadas
                pt1_norm = np.linalg.inv(K) @ np.array([x1, y1, 1])
                pt2_norm = np.linalg.inv(K) @ np.array([x2, y2, 1])
                
                # Calcula ângulo de paralaxe
                # Vetor da câmera 1 para o ponto
                v1 = pt1_norm / np.linalg.norm(pt1_norm)
                # Vetor da câmera 2 para o ponto (transformado)
                v2 = (R.T @ pt2_norm) / np.linalg.norm(R.T @ pt2_norm)
                
                # Ângulo entre os vetores
                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)
                parallax_angles.append(np.degrees(angle))
                
            except Exception as e:
                print(f"Warning: Erro no cálculo de paralaxe para ponto {i}: {e}")
                continue
        
        if parallax_angles:
            pose_metrics['mean_parallax_angle'] = np.mean(parallax_angles)
            pose_metrics['std_parallax_angle'] = np.std(parallax_angles)
            pose_metrics['min_parallax_angle'] = np.min(parallax_angles)
        else:
            print(f"Warning: Nenhum ângulo de paralaxe válido calculado")
            pose_metrics['mean_parallax_angle'] = 0
            pose_metrics['std_parallax_angle'] = 0
            pose_metrics['min_parallax_angle'] = 0
    
    # 4. Determinante de R (deve ser ~1 para rotação válida)
    det_R = np.linalg.det(R)
    pose_metrics['det_R'] = det_R
    pose_metrics['R_orthogonality_error'] = np.linalg.norm(R @ R.T - np.eye(3))
    
    return pose_metrics
```

## Principais Correções Aplicadas

1. **Normalização do formato dos pontos**: Adicionado tratamento para formato `(N, 1, 2)` usando `squeeze(1)`
2. **Acesso seguro às coordenadas**: Verifica `pt.shape == (2,)` antes de acessar `pt[0]` e `pt[1]`
3. **Extração explícita das coordenadas**: Usa `x1, y1 = pt1[0], pt1[1]` ao invés de acessar diretamente
4. **Tratamento de erros**: Blocos try-except para cada cálculo crítico
5. **Validação de pontos**: Verifica se existem pontos inliers antes de processar
6. **Valores padrão**: Define valores padrão para métricas quando há problemas

## Como Aplicar a Correção

1. **Abra o notebook S3**: `/home/haylapbv/T2/MC949-Visao-Computacional/notebooks/T2/S3-1.0-hpbv-epipolar-geometry.ipynb`
2. **Localize a célula 21**: Cell Id `#VSC-ae28fd24` (linhas 391-479)
3. **Encontre a função `analyze_pose_quality`**: Segunda função na célula
4. **Substitua toda a função**: Pela versão corrigida acima
5. **Execute a célula**: Para testar a correção

## Resultado Esperado

Após a correção, você deve ver:
```
=== ANÁLISE DE QUALIDADE DA POSE ===
Comprimento da baseline: X.XXX
Ângulo de rotação: XX.XX°
Determinante de R: X.XXXXXX (ideal: 1.0)
Erro de ortogonalidade: X.XXXXXX
Ângulo de paralaxe médio: XX.XX°
Ângulo de paralaxe mínimo: XX.XX°
✓ Ângulos de paralaxe adequados para triangulação
```

## Debug Adicional

Se ainda houver problemas, adicione estas linhas no início da função para diagnosticar:

```python
# Debug - mostra formato dos pontos de entrada
print(f"Debug - pts1 shape: {pts1.shape}, pts2 shape: {pts2.shape}")
print(f"Debug - mask_E shape: {mask_E.shape}, sum: {np.sum(mask_E)}")
print(f"Debug - pts1 sample: {pts1[:2]}")
print(f"Debug - pts2 sample: {pts2[:2]}")
```

## Problema Similar em Outras Funções

Se você encontrar erros similares em outras funções de análise de qualidade, aplique o mesmo padrão de correção:

1. **Normalizem formato dos pontos** com `squeeze(1)` se necessário
2. **Verifique shape** antes de acessar índices
3. **Use try-except** para cálculos críticos
4. **Defina valores padrão** para casos problemáticos

Esta correção torna a função robusta para diferentes formatos de entrada dos pontos, evitando crashes e fornecendo análises confiáveis da qualidade da pose.