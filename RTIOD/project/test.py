from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import hydra

@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(args):
    model = YOLO(args.modelCheckpoint)
    res = model.val(data="data/data.yaml", save_json=True, plots=True)

if __name__ == "__main__":
    main()