import json
import os
from dotenv import load_dotenv
import base64
import cv2
import numpy as np
from flask import Flask, request, Response, render_template_string
from inference import get_model
import supervision as sv

app = Flask(__name__)

# Load your backgammon model using the provided model id.
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
model = get_model(model_id="backgammon-eofws/5", api_key=api_key)


@app.route('/')
def index():
    # HTML page with an image upload form and areas to display the JSON detections and the annotated image.
    html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backgammon Board Detector</title>
        </head>
        <body>
            <h1>Upload Backgammon Board Image</h1>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br><br>
                <input type="submit" value="Analyze">
            </form>
        </body>
        </html>
        '''
    return render_template_string(html)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return "No image provided", 400

    # Read the image file from the request.
    print("I am analysing!")
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print("Image shape:", image.shape)
    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(image)[0]

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference result
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Encode the annotated image to JPEG, then to base64 so it can be embedded in HTML.
    retval, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
    sv.plot_image(annotated_image)
    print(annotated_image_b64)
    cv2.imwrite("debug_uploaded_image.jpg", image)

    # Create an HTML page that shows the annotated image and the detection information.
    result_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Results</title>
    </head>
    <body>
        <h1>Analysis Results</h1>
        <h2>Annotated Image</h2>
        <img src="data:image/jpeg;base64,{annotated_image_b64}" alt="Annotated Image" style="max-width: 100%; height: auto;"/>
        <h2>Detections</h2>
        <pre>{detections}</pre>
        <br>
        <a href="/">Upload another image</a>
    </body>
    </html>
    '''
    return result_html


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
