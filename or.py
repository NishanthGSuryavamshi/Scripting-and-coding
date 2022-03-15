from pip import main
from utils.all_utils  import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron

def main(data,modelName,plotName,eta,epochs):
    df_OR = pd.DataFrame(data)
    X, y = prepare_data(df_OR)

 
    model_or = Perceptron(eta=eta, epochs=epochs)
    model_or.fit(X, y)

    _ = model_or.total_loss()
    model_or.save(filename=modelName, model_dir='model')
    save_plot(df_OR,model_or,filename=plotName)

if __name__=='__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,1]
}
ETA=0.1
EPOCHS=10
main(data=OR, modelName='or.model',plotName='or.png',eta=ETA,epochs=EPOCHS)
df_OR = pd.DataFrame(OR)



X, y = prepare_data(df_OR)



model_or = Perceptron(eta=ETA, epochs=EPOCHS)
model_or.fit(X, y)

_ = model_or.total_loss()
model_or.save(filename="or.model", model_dir="model_or")
save_plot(df_OR,model_or,filename='or.png')
