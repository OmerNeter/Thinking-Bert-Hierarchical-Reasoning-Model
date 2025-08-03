from config import config, final_model_path
from train import train
from inference import run_inference
import logging

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    train(config)
    logger.info('--- Finished Training ---')

    run_inference(config, model_path=final_model_path)
    run_inference(config, model_path=final_model_path, n_hier_override=4, t_hier_override=3)
    run_inference(config, model_path=final_model_path, n_hier_override=1, t_hier_override=1)