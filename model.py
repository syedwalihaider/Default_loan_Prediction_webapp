import pandas as pd
import pickle

with open("C:\\Users\\Dell\\tutorial\\columns.pkl", "rb") as f:
    columns = pickle.load(f)

pipe = pickle.load(open("C:\\Users\\Dell\\tutorial\\model.pkl","rb"))

class pred:
    def find(self,a):
        a = pd.DataFrame(a,columns=columns)
        response = pipe.predict(a)
        return response



