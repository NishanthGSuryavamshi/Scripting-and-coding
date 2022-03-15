from pyexpat import model
from pip import main
from utils.all_utils  import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron

def main(data,modelName,plotName,eta,epochs):
    df_and = pd.DataFrame(data)
    X, y = prepare_data(df_and)

 
    model_and = Perceptron(eta=eta, epochs=epochs)
    model_and.fit(X, y)

    _ = model_and.total_loss()
    model_and.save(filename=modelName, model_dir='model')
    save_plot(df_and,model_and,filename=plotName)

if __name__=='__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,1,1,1]
}
ETA=0.1
EPOCHS=10
main(data=AND, modelName='and.model',plotName='and.png',eta=ETA,epochs=EPOCHS)
df_and = pd.DataFrame(AND)



X, y = prepare_data(df_and)



model_or = Perceptron(eta=ETA, epochs=EPOCHS)
model_or.fit(X, y)

_ = model_or.total_loss()
model_or.save(filename="and.model", model_dir="model")
save_plot(df_and,model,filename='and.png')
