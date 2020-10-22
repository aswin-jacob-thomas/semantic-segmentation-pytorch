import flask
from leaning_angle import initialize, leaning_angle
import numpy as np
import cv2
app = flask.Flask(__name__)
app.config["DEBUG"] = True
# segmentation_module = initialize()
initialize()
@app.route('/', methods=['POST'])
def home():
    r = flask.request
    image = r.files['data']
#     leaning_angle(segmentation_module, image)
    print("Got image!!")
    angle = leaning_angle(image)
    return flask.Response(response=str(angle), status=200, mimetype="application/json")

app.run(host='0.0.0.0', port=8081)
