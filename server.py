import flask
from leaning_angle import initialize, leaning_angle
import numpy as np
import cv2
app = flask.Flask(__name__)
app.config["DEBUG"] = True
segmentation_module = initialize()
@app.route('/', methods=['POST'])
def home():
    r = flask.request
    # nparr = np.fromstring(r.files['data'], np.uint8)
    # # decode image
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = r.files['data']
    leaning_angle(segmentation_module, image)
    return flask.Response(response='Leaning angle found', status=200, mimetype="application/json")

app.run()
