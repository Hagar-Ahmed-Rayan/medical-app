from flask import Flask, jsonify,  redirect, url_for, request, render_template
from PIL import Image
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from main_test import *
app = Flask(__name__)

# # preprocess Image #

# # get predictions #
def preprocess_Image(path1, path2):
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485),
                                     (0.229))])
    image_1 = Image.open(path1).convert("RGB") # (512, 420)
    image_2 = Image.open(path2).convert("RGB") # (512, 420)
    image_1 = transform(image_1) # [3, 224, 224]
    image_2 = transform(image_2) # [3, 224, 224]
    image = torch.stack((image_1, image_2), 0) # [2, 3, 224, 224]
    return image

def make_prediction(path1, path2, model):
    image = preprocess_Image(path1, path2).unsqueeze(0) # [2,3,224,224]
    # weight of size [64, 3, 7, 7]
    output = model(image.to('cpu'), mode='sample')
    report = model.tokenizer.decode_batch(output.cpu().numpy())
    return report
# Load your trained model

# parse arguments
args = parse_agrs()

# fix random seeds
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# create tokenizer
tokenizer = Tokenizer(args)
model = R2GenModel(args, tokenizer)
load_path = "results/iu_xray/model_best.pth"
# model = model.load("results/iu_xray/model_best.pth",map_location=torch.device('cpu'))
checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
# @app.route('/')
# def hello():
#     return 'Hello World!'
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

# function  accepts only POST requests:
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f=[]
        for item in request.files.getlist('file1'):
            f.append(item)
        f1 = request.files['file1']
        f2 = request.files['file1']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path1 = os.path.join(basepath, 'uploads', secure_filename(f[0].filename))
        file_path2 = os.path.join(basepath, 'uploads', secure_filename(f[1].filename))
        f[0].save(file_path1)
        f[1].save(file_path2)

        # Make prediction
        pred_report = make_prediction(file_path1, file_path2, model)

        # result = str(pred_class[0][0][1])               # Convert to string
        return pred_report[0]
    return None
if __name__ == '__main__':
    app.run()


# #  run a Flask development server by typing
# # $ FLASK_ENV=development FLASK_APP=app.py flask run

# # When you visit http://localhost:5000/ in your web browser, you will be greeted with Hello World! text

