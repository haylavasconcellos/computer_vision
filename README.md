# MC949-Visao-Computacional

Repositório com os projetos da disciplina MC949/MO4446 - Visão Computacional.

## Estrutura de Diretórios

Template de projeto inspirado no [Cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/#directory-structure).

- `data/`
    - `raw/`: Dataset original em sua forma inalterada
    - `interim/`: Versões pré-processadas dos dados originais
    - `results/`: Resultados finais do projeto
- `docs/`: Documentação do projeto
- `models/`: Modelos pré-treinados e checkpoints
- `notebooks/`: Notebooks de playground para fins de pesquisa
- `src/`: Código definitivo do projeto

Como o repositório armazena o código referente a 4 projetos distintos, cada um desses diretórios foi dividido em T1, T2, T3 e T4. Com isso, a estrutura do repositório é a seguinte:

```txt
├── data
│   ├── T1
│   |   ├── interim
│   |   ├── raw
│   |   └── results
│   ├── T2
│   |   ├── interim
│   |   ├── raw
│   |   └── results
│   └── T4
|       ├── imagens
│       ├── mascaras
│       └── results
├── docs
├── models
├── notebooks
│   ├── T1
│   ├── T2
│   └── T4
├── requirements.txt
├── run.sh
└── src
    ├── canon
    │   ├── T1
    │   ├── T2
    │   ├── T4
    │   ├── config.py
    │   ├── download_data.py
    │   └── utils
    └── pyproject.toml
```

## Execução dos Projetos

Para executar os projetos, foi disponibilizado um script `run.sh` na raiz do repositório. A execução do script realiza as seguintes etapas:

1. Criação do ambiente virtual e instalação das bibliotecas necessárias
2. Download dos dados do projeto especificado
3. Execução da pipeline (no caso do T2 e T4)

### Como Usar

Execute os seguintes comandos na raiz do repositório, substituindo `TX` pelo projeto desejado (`T1`, `T2` ou `T4`):

```bash
chmod +x run.sh
./run.sh --project TX
```

**Exemplos:**
- Para T1: `./run.sh --project T1` - Baixa os dados e prepara o ambiente
- Para T2: `./run.sh --project T2` - Baixa os dados e executa automaticamente a pipeline de reconstrução 3D  
- Para T4: `./run.sh --project T4` - Baixa imagens e executa automaticamente a pipeline de inpainting

## Projeto T1: Montagem de Panoramas

O projeto T1 monta panoramas a partir de um conjunto de imagens com sobreposição parcial.

### Funcionalidades Principais

- **Extração e correspondência de features**: Detecção de pontos-chave com SIFT e matching entre imagens
- **Ordenação automática**: Determina a sequência ideal das imagens via grafo de correspondências
- **Stitching**: Alinhamento e composição das imagens em um panorama único
- **Blending**: Suavização das costuras entre imagens adjacentes (feathering)

O código canônico está em `src/canon/T1/`. A exploração e experimentos estão nos notebooks em `notebooks/T1/`.

## Projeto T2: Reconstrução 3D a partir de Múltiplas Vistas

O projeto T2 reconstrói a geometria 3D de uma cena a partir de um conjunto de imagens 2D, usando Structure from Motion (SfM).

### Etapas da Pipeline

- **Extração de features**: SIFT, AKAZE ou ORB, com filtro de Lowe
- **Geometria epipolar**: Estimativa da matriz fundamental e calibração da câmera
- **Reconstrução incremental**: Triangulação de pontos 3D e registro progressivo de novas vistas
- **Visualização**: Nuvem de pontos e malha 3D com Open3D

A pipeline é executada automaticamente via `run.sh`. O código canônico está em `src/canon/T2/`.

## Projeto T4: Modelos de Difusão para Restauração de Imagens

O projeto T4 implementa modelos de difusão para tarefas de restauração e expansão de imagens (inpainting).

### Modelos Implementados

O projeto implementa 3 modelos de inpainting:

- **Stable Diffusion Inpainting**: Versátil e rápido, com suporte a prompts de texto opcionais
- **Paint-by-Example**: Restauração guiada por exemplos visuais
- **Kandinsky 2.2 Inpainting**: Otimizado para restauração de fotos vintage, com detecção automática de danos

### Restauração de Fotos Antigas

O projeto inclui utilitários especializados para restauração de fotos antigas:

- **Detecção Automática de Danos**: Identifica rachaduras, manchas e áreas deterioradas
- **Pré-processamento**: Redução de ruído e ajuste de contraste

Para mais detalhes, consulte `src/canon/T4/utils.py` (função `photo_restoration_utils`).