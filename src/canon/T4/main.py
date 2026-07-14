import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from metrics.metrics import LPIPS, NIQE, SSIM
from process.batch_utils import get_model, list_available_models
from tqdm import tqdm

from canon.config import BASE_DATA_PATH
from canon.T4.utils import load_image_and_masks

# Configurações de Caminhos
DATA_DIR = BASE_DATA_PATH / "T4"
IMAGES_DIR = DATA_DIR / "imagens"
MASKS_DIR = DATA_DIR / "mascaras"
OUTPUT_DIR = DATA_DIR / "results"


def run_inpainting_pipeline():
    # 1. Preparação
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Iniciando Pipeline no dispositivo: {device}")

    # Listar modelos disponíveis
    model_names = list_available_models()
    print(f"[INFO] Modelos encontrados: {model_names}")

    # Encontrar todas as imagens originais
    image_files = sorted(
        list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    )

    if not image_files:
        print(f"[ERROR] Nenhuma imagem encontrada em {IMAGES_DIR}")
        return

    # 2. Execução (Iterar por Modelo -> Imagem -> Máscara)
    # Iteramos por modelo primeiro para evitar carregar/descarregar VRAM repetidamente
    for model_name in model_names:
        print(f"\n{'='*40}")
        print(f"[INFO] Carregando Modelo: {model_name}")
        print(f"{'='*40}")

        # Lista para armazenar resultados das métricas
        model_metrics = []

        try:
            # Carrega o modelo
            model = get_model(model_name, device=device)
            model.load_model()

            # Itera sobre as imagens
            for image_path in tqdm(image_files, desc=f"Processando com {model_name}"):
                image_stem = image_path.stem
                print(f"\n{'='*40}")
                print(f"[INFO] Processando Imagem: {image_stem}")
                print(f"{'='*40}")

                # Carregar imagem e máscaras
                original_image, masks = load_image_and_masks(image_stem)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # --- INFERÊNCIA ---
                for i, mask in enumerate(masks):
                    try:
                        mask_name = f"{image_stem}_mask_{i+1}"
                        result = model.inpaint(
                            original_image, mask, image_path=image_path
                        )

                        generated_image = result["image"]
                        inference_time = result["inference_time"]

                        # --- SALVAR RESULTADO ---
                        # Cria estrutura de pasta: results/modelo/nome_imagem/
                        save_folder = OUTPUT_DIR / model_name
                        save_folder.mkdir(parents=True, exist_ok=True)

                        save_filename = f"{mask_name}_inpainted.png"
                        generated_image.save(save_folder / save_filename)

                        # --- CÁLCULO DE MÉTRICAS ---
                        generated_image = np.array(generated_image)

                        # 1. SSIM (Maior é melhor, 1.0 é idêntico)
                        ssim_val, ssim_diff = SSIM(original_image, generated_image)
                        cv2.imwrite(
                            save_folder / f"{mask_name}_ssim_diff.png", ssim_diff
                        )

                        # 2. LPIPS (Menor é melhor, 0.0 é idêntico)
                        lpips_val = LPIPS(original_image, generated_image)

                        # 3. NIQE (Menor é melhor, qualidade perceptual sem referência)
                        niqe_val = NIQE(generated_image, crop_border=False)

                        # --- REGISTRAR DADOS ---
                        model_metrics.append(
                            {
                                "model": model_name,
                                "image": image_stem,
                                "mask": mask_name,
                                "config": result["config"],
                                "ssim": ssim_val,
                                "lpips": lpips_val,
                                "niqe": niqe_val,
                                "inference_time": inference_time,
                                "output_path": str(save_folder / save_filename),
                            }
                        )

                    except Exception as e:
                        print(
                            f"[ERRO] Falha ao processar {image_stem} com máscara {i}: {e}"
                        )

            # Limpar VRAM após terminar com o modelo
            model.unload_model()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Falha ao carregar modelo {model_name}: {e}")

        finally:
            # --- SALVAR MÉTRICAS DO MODELO ---
            if model_metrics:
                csv_path = OUTPUT_DIR / model_name / "metrics_summary.csv"
                df = pd.DataFrame(model_metrics)
                df.to_csv(csv_path, index=False)
                print(f"\n[INFO] Métricas de {model_name} salvas em: {csv_path}")

            else:
                print("Nenhum resultado foi gerado.")

    print(f"\n{'='*40}")
    print("Pipeline Concluída!")
    print(f"{'='*40}")


def generate_summary():
    # Encontrar todos os arquivos de métricas por modelo
    # Padrão esperado: metrics_NomeDoModelo.csv
    csv_files = []
    model_names = list_available_models()
    for model in model_names:
        csv_files.append(OUTPUT_DIR / model / "metrics_summary.csv")

    if not csv_files:
        print(f"Nenhum arquivo de métricas encontrado em {OUTPUT_DIR}")
        return

    print(f"Encontrados {len(csv_files)} arquivos de métricas.")

    all_metrics = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_metrics.append(df)

        except Exception as e:
            print(f"Erro ao ler {file_path}: {e}")

    if not all_metrics:
        print("Não foi possível carregar dados.")
        return

    # Concatena todos os dados em um único DataFrame
    full_df = pd.concat(all_metrics, ignore_index=True)

    # --- 1. Resumo Geral (Médias) ---
    print("\n" + "=" * 60)
    print("RESUMO DAS MÉTRICAS (Média por Modelo)")
    print("=" * 60)

    # Agrupa por modelo e calcula a média das métricas numéricas
    summary = full_df.groupby("model")[["ssim", "lpips", "niqe", "inference_time"]].agg(
        ["mean", "std"]
    )
    print(summary.round(4))

    # Salva o resumo em CSV
    summary_path = OUTPUT_DIR / "final_summary_averages.csv"
    summary.to_csv(summary_path)
    print(f"\n[Salvo] Resumo das médias salvo em: {summary_path}")

    # --- 2. Análise de Melhores/Piores Casos ---
    print("\n" + "=" * 60)
    print("ANÁLISE DE EXTREMOS (Melhor e Pior caso por Modelo)")
    print("=" * 60)

    # Métricas para analisar e sua direção (True = Maior é melhor, False = Menor é melhor)
    metrics_config = {
        "ssim": {"higher_is_better": True},
        "lpips": {"higher_is_better": False},
        "niqe": {"higher_is_better": False},
    }

    extremes_list = []

    for model_name in full_df["model"].unique():
        print(f"\n--- Modelo: {model_name} ---")

        model_df = full_df[full_df["model"] == model_name]

        for metric, config in metrics_config.items():
            if metric not in model_df.columns:
                continue

            # Remove NaNs para não quebrar a busca
            valid_df = model_df.dropna(subset=[metric])

            if valid_df.empty:
                print(f"  {metric.upper()}: Sem dados válidos.")
                continue

            # Encontra min e max
            min_row = valid_df.loc[valid_df[metric].idxmin()]
            max_row = valid_df.loc[valid_df[metric].idxmax()]

            # Define qual é o "Melhor" e o "Pior" baseado na métrica
            if config["higher_is_better"]:
                best_row = max_row
                worst_row = min_row
            else:
                best_row = min_row
                worst_row = max_row

            print(f"\n{metric.upper()}:")
            print(
                f"MELHOR: {best_row[metric]:.4f} | Img: {best_row['image']} | Mascara: {best_row['mask']}"
            )
            print(
                f"PIOR:   {worst_row[metric]:.4f} | Img: {worst_row['image']} | Mascara: {worst_row['mask']}"
            )

            extremes_list.append(
                {
                    "Model": model_name,
                    "Metric": metric.upper(),
                    "Type": "BEST",
                    "Value": best_row[metric],
                    "Image": best_row["image"],
                    "Mask": best_row["mask"],
                    "Path": (
                        best_row["output_path"] if "output_path" in best_row else "N/A"
                    ),
                }
            )

            extremes_list.append(
                {
                    "Model": model_name,
                    "Metric": metric.upper(),
                    "Type": "WORST",
                    "Value": worst_row[metric],
                    "Image": worst_row["image"],
                    "Mask": worst_row["mask"],
                    "Path": (
                        worst_row["output_path"]
                        if "output_path" in worst_row
                        else "N/A"
                    ),
                }
            )

    # --- Salva os extremos em CSV ---
    if extremes_list:
        extremes_df = pd.DataFrame(extremes_list)
        extremes_df = extremes_df.sort_values(by=["Model", "Metric", "Type"])

        extremes_path = OUTPUT_DIR / "final_summary_extremes.csv"
        extremes_df.to_csv(extremes_path, index=False)
        print(f"\n[Salvo] Relatório de extremos salvo em: {extremes_path}")

    print("\n" + "=" * 60)
    print("Análise Concluída.")


# --- Parser e Entry Point ---
if __name__ == "__main__":
    # Configurar argparse para receber argumentos
    parser = argparse.ArgumentParser(description="Pipeline de Modelos de Difusão")
    parser.add_argument(
        "--task",
        type=str,
        choices=["inpainting", "outscaling", "summary"],
        required=True,
        help="Tarefa a se realizar (inpainting, outscaling ou summary)",
    )

    args = parser.parse_args()
    task = args.task
    if task == "inpainting":
        run_inpainting_pipeline()
    elif task == "upscaling":
        raise NotImplementedError(
            "Pipeline não implementada no script, execute os notebooks `notebboks/T4/S3-1.0-victor-upscaling-bsrgan.ipynb` e `notebooks/T4/S3-1.0-victor-upscaling-stable-diffusion.ipynb` para testar os modelos"
        )
    else:
        generate_summary()
