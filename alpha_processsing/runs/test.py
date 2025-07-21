# main.py
from runs.pipeline import loop_experiments
import logging

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 定义配置文件路径
    CONFIG_PATH = "configs/main_config.yaml"

    logging.info("===== Experiments started !=====")
    loop_experiments(CONFIG_PATH) # 调用总调度器
    logging.info("===== All the experiments are done ! =====")