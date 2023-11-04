import pandas as pd
import numpy as np
from utils.all_utils import prepare_data,save_plot
from utils.model import Perceptron
import logging
import os

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir,"running_logs.logs"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s:",
    filemode="a"
    )

def main(data, modelName,plotName,eta,epochs):
    df_OR = pd.DataFrame(data)
    logging.info(f"this is the raw dataset: {df_OR}")
    X,y = prepare_data(df_OR)
    model_or = Perceptron(eta=eta,epochs=epochs)
    model_or.fit(X,y)

    _ = model_or.total_loss()

    model_or.save(filename=modelName,model_dir="model")

    save_plot(df_OR,model_or,filename=plotName)


if __name__ == "__main__": # it creates an entry point
    OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]
    }
    ETA = 0.1
    EPOCHS = 10
    try:
        logging.info("starting training")
        main(data=OR,modelName="or.model",plotName= "or.png",eta=ETA,epochs=EPOCHS)
        logging.info("done training")
    except Exception as e:
        logging.exception(e)
        raise e











