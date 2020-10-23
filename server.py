import flask
from leaning_angle import initialize, leaning_angle
import numpy as np
import cv2
app = flask.Flask(__name__)
app.config["DEBUG"] = False
segmentation_module = initialize()
# initialize()
print("Initialization complete!!")
@app.route('/', methods=['POST'])
def home():
    r = flask.request
    image = r.files['data']
#     leaning_angle(segmentation_module, image)
    print("Got image!!")
    angle = leaning_angle(segmentation_module, image)
    return flask.Response(response=str(angle), status=200, mimetype="application/json")

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8081)
