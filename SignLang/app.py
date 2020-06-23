import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import backend as k
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
global graph
from tensorflow.python.keras.backend import set_session
sess = tf.Session()
graph = tf.get_default_graph()




from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
set_session(sess)
model = load_model("SignLangModel.h5")
@app.route('/')
def index():
    return render_template('base.html')


@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            set_session(sess)
            preds = model.predict_classes(x)
            
            
            print("prediction",preds)
            
        index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        
        text = "the predicted alphabet is : " + str(index[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = True, port=8000 ,threaded = False)

        
    
    
    