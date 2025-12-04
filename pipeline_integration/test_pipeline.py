import yaml
from pipeline_integration.orchestrator import PipelineOrchestrator

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_pipeline():
    config = load_config()

    model_path = config["paths"]["model_save_path"]
    video_path = config["paths"]["test_video_path"]
    image_folder_path = config["paths"]["test_image_folder_path"]

    # Test con un video
    print("Esecuzione della pipeline con un video...")
    orchestrator = PipelineOrchestrator(source_type="video", source_path=video_path, model_path=model_path)
    orchestrator.run()

    # Test con una cartella di immagini
    print("Esecuzione della pipeline con una cartella di immagini...")
    orchestrator = PipelineOrchestrator(source_type="image_folder", source_path=image_folder_path, model_path=model_path)
    orchestrator.run()

if __name__ == "__main__":
    test_pipeline()