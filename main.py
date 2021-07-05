from flask import Flask , render_template , request
app = Flask(__name__)
import pickle

#opening file where pichle data is stored
file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=['GET','POST'])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        Age = int(myDict['Age'])
        runnyNose = int(myDict['runnyNose'])
        bodyPain = int(myDict['bodyPain'])
        cough = int(myDict['cough'])
        diffBreath = int(myDict['diffBreath'])
        # code for inference
        input=[fever,bodyPain,Age,runnyNose,diffBreath,cough]
        infProb = clf.predict_proba([input])[0][1]
        print(infProb)
        return render_template('show.html', inf = (infProb*100))
    return render_template('index.html' )
    
@app.route('/')
def home():
    return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)
    