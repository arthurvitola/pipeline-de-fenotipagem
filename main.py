from processor import process_dataset_folder

dataset_directory = "./reduced-dataset"
output_plots_dir = "./output"

# Fator de escala
scale_ratio = 0.02

# Executa o processamento em lote, gera os plots de diagnóstico e exporta para csv
results_df = process_dataset_folder(
    folder_path=dataset_directory,
    px_to_cm_ratio=scale_ratio,
    output_dir=output_plots_dir
)

if not results_df.empty:
    output_csv = "maize_phenotyping_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nBatch processing complete! Results saved to {output_csv}")
    print(f"Pipeline diagnostic plots saved to folder: {output_plots_dir}/")
    print(results_df.head())