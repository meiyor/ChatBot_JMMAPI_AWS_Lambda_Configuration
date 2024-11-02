"""
This app.py code contains the functions for runing and setting app
the chatbot API and realease it as a Dockerized image in ECR.
Please follow the serverless deployment to set this up in AWS
Add the permissions defined in the READme if you want to configure it
"""
# import Flask dependencies

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import serverless_wsgi

# import the SQLAlchemy and database support modules
from models_database import db, APIData

# import pytorch objects
import torch
import torchvision.transforms as transforms

# import this for the bedrock endpoint
import boto3
import json
import os
import random
import base64
import datetime
import string
import boto3
from PIL import Image
from io import BytesIO


# remove warning here
import warnings
warnings.filterwarnings(action='ignore', message='Could not obtain multiprocessing lock')

# define Pytorch validation image transformations
img_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406],
            std=[
                0.229,
                0.224,
                0.225]),
    ])

# define the labels
labels_string = 'Faces,Faces_easy,Leopards,Motorbikes,accordion,airplanes,anchor,ant,barrel,bass,beaver,binocular,bonsai,brain,brontosaurus,buddha,butterfly,camera,cannon,car_side,ceiling_fan,cellphone,chair,chandelier,cougar_body,cougar_face,crab,crayfish,crocodile,crocodile_head,cup,dalmatian,dollar_bill,dolphin,dragonfly,electric_guitar,elephant,emu,euphonium,ewer,ferry,flamingo,flamingo_head,garfield,gerenuk,gramophone,grand_piano,hawksbill,headphone,hedgehog,helicopter,ibis,inline_skate,joshua_tree,kangaroo,ketch,lamp,laptop,llama,lobster,lotus,mandolin,mayfly,menorah,metronome,minaret,nautilus,octopus,okapi,pagoda,panda,pigeon,pizza,platypus,pyramid,revolver,rhino,rooster,saxophone,schooner,scissors,scorpion,sea_horse,snoopy,soccer_ball,stapler,starfish,stegosaurus,stop_sign,strawberry,sunflower,tick,trilobite,umbrella,watch,water_lilly,wheelchair,wild_cat,windsor_chair,wrench,yin_yang'
labels_vals = labels_string.split(',')

# set the environment values for the database
user_name = os.environ['USER_NAME']
password = os.environ['PASSWORD']
rds_host = os.environ['RDS_HOST']
rds_port = os.environ['RDS_PORT']
db_name = os.environ['DB_NAME']

# read the models obtained by the ResNet18 on AWS SageMaker
checkpoint_base = torch.load(
    "./models/model_checkpoint_Trial-2024-10-06-043100936995-gxzk0")
model_base = checkpoint_base["model"]
model_base.load_state_dict(checkpoint_base["state_dict"])

checkpoint_pruning = torch.load(
    "./models/model_checkpoint_Trial-2024-10-06-053640801541-plru1")
model_pruning = checkpoint_pruning["model"]
model_pruning.load_state_dict(checkpoint_pruning["state_dict"])

# use this provisionally if you really need it or not!! please let them
# run when you consider
model_base.eval()
model_pruning.eval()

# set the boto3 values for querying
bedrock_runtime = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.environ.get("MY_AWS_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("MY_AWS_SECRET_KEY"))

# define the boto3 session
session = boto3.Session(
    aws_access_key_id=os.environ.get("MY_AWS_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("MY_AWS_SECRET_KEY"),
    region_name='us-east-1')
s3 = session.resource('s3')

# define the app object
app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql+psycopg2://postgres:DataBase@localhost:5432/apidb"
# use this configuration for docker
# app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql+psycopg2://postgres:DataBase@backend_chatbot:5432/apidb"
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://' + \
    user_name + ':' + password + '@' + rds_host + ':' + rds_port + '/' + db_name
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialize database initially - don't call it from main
db.init_app(app)
with app.app_context():
   db.create_all()

CORS(app)

@app.get("/")
def index_get():
    return render_template("index.html")


@app.post("/adduser")
def adduser():
    """
     This is the preliminary function for adding and validating
     the user form. This will execute first than anything.

     :return jsonify(ack): the jsonified object of the ack value to
     validate the complete input.
     :rtype jsonify(ack): json dict/map
    """
    # validate the input
    ack = 'none'
    global username
    global password
    userpass = request.get_json()
    print(userpass)
    username = userpass.get("user")
    password = userpass.get("pass")
    if len(username) == 0 or len(password) == 0:
        return (jsonify('incomplete'))
    else:
        return jsonify(ack)


@app.post("/ini")
def ini():

    """
     This is the initialization function that starts the API.
     it returns a welcome message to the client from the chatbot
     widget.

     :return jsonify(message): the jsonified object of the  initial
     message composed with the welcome string in "answer" and the returned
     image status in "file_name"
     to the API
     :rtype jsonify(message): json dict/map
    """

    global img
    global img_base64
    global interactions
    global data_cumm
    global img_cumm

    img = []
    img_base64 = ""
    interactions = 1
    data_cumm = []
    img_cumm = []
    message = {
        "answer": "Let's start having an interaction with the chatbot.. <br>",
        "file_name": 'https://bedrockapijmm.s3.us-east-1.amazonaws.com/gray.jpg'}
    return jsonify(message)


@app.post("/predict")
def predict(Data=APIData, db=db):

    """
     This is the predict post function receiving the APIData models object
     and the database object generated for this Flask API. The database must
     be initialized outside the main or any POST functions defined in this
     app.py file. This function contains the Bedrock enpoint invoking, the
     data deploying, and the database updating after two interactions.

     :param Data: An APIData object defined in the models_database module
     This defines the parameters of the database but to add a new item on it.
     :param db: This is a SQLAlchemy database session as Postgresql. This initialized
     before any app function will run.
     :return jsonify(message): the jsonified object of the message
     coming from the bedrock endpoint response in "answer" and the returned
     image status in "file_name" with the updated image input coming from the front
     to the API, in case the image is presented.
     :rtype jsonify(message): json dict/map
    """


    global img
    global img_base64
    global interactions
    global data_cumm
    global username
    global img_cumm
    global file_name

    # This is the input of the chatbox
    text = request.get_json().get("message")

    random.random()

    if img_base64:

        processing_img = True

        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5000,
                "messages": [
                      {
                          "role": "user",
                          "content": [
                              {
                                  "type": "image",
                                  "source": {
                                      "type": "base64",
                                      "media_type": "image/jpeg",
                                      "data": img_base64
                                  }
                              },
                              {
                                  "type": "text",
                                  "text": " " + text + " \n"
                              }
                          ]
                      }
                ]
            })
        }

        # capture ResNet model estimation
        # first do image transformation
        img_test = Image.open(
            BytesIO(
                base64.b64decode(img_base64))).convert('RGB')

        img_tensor = img_transforms(img_test)
        img_tensor -= torch.min(img_tensor)
        img_tensor /= torch.max(img_tensor)

        # normalize image before put in the model
        img_tensor = 2 * img_tensor - 1
        img_tensor = img_tensor.unsqueeze(0)

        print(img_tensor, img_tensor.shape, 'data')

        predict_base = model_base(img_tensor)
        predict_pruning = model_pruning(img_tensor)

        class_prob_base = torch.softmax(predict_base, dim=1)

        # get most probable class and its probability:
        class_prob_base, topclass_base = torch.max(class_prob_base, dim=1)

        class_prob_base = class_prob_base * 100

        class_prob_pruning = torch.softmax(predict_pruning, dim=1)

        # get most probable class and its probability:
        class_prob_pruning, topclass_pruning = torch.max(
            class_prob_pruning, dim=1)

        class_prob_pruning = class_prob_pruning * 100

        img = []
        img_base64 = ""

    else:

        processing_img = False

        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5000,
                "messages": [
                      {
                          "role": "user",
                          "content": [
                              {
                                  "type": "text",
                                  "text": " " + text + " \n"
                              }
                          ]
                      }
                ]
            })
        }

    # get the endpoint response
    response = bedrock_runtime.invoke_model(**kwargs)

    response_text = json.loads(response['body'].read())

    # takes the input text

    print(response_text['content'][0], 'dataresponse')

    # get the time just after the query is done
    time_now = datetime.datetime.now().strftime("%I:%M:%S%p-%B-%d-%Y")

    if processing_img:

        # create temporal file for uploading the image
        if not os.path.exists('/tmp/results/'):
            os.makedirs('/tmp/results/', exist_ok=True)

        code_str = ''.join(random.choice(string.ascii_letters)
                           for i in range(16))
        file_name = file_name.split('.')[0] + '_' + code_str + '.jpg'
        img_test.save('/tmp/results/' + file_name)

        data_cumm.append(username + ': ' + text + '\n ' + ' ChatBot: ' + response_text['content'][0]['text'] + '\n ' + 'ResNet18: ' + labels_vals[topclass_base.cpu().numpy()[0]] + ' prob: ' + str(class_prob_base.cpu(
        ).detach().numpy()[0]) + '%\n ' + 'ResNet18-pruning: ' + labels_vals[topclass_pruning.cpu().numpy()[0]] + ' prob: ' + str(class_prob_pruning.cpu().detach().numpy()[0]) + '%\n ' + time_now + '\n ')

        # send the files to S3
        s3.meta.client.upload_file(
            Filename='/tmp/results/' +
            file_name,
            Bucket='bedrockapijmm',
            Key=file_name)

        if class_prob_base.cpu().detach().numpy()[0] >= 45:
            message = {"answer": response_text['content'][0]['text'] +
                       '<br> The output of the offline ResNet models is: <br> <b>ResNet18:</b>  The image is a <b>' +
                       labels_vals[topclass_base.cpu().numpy()[0]] +
                       '</b> with a probability of ' +
                       str(class_prob_base.cpu().detach().numpy()[0]) +
                       '%<br> <b>ResNet18 pruning:</b> The image is a <b>' +
                       labels_vals[topclass_pruning.cpu().numpy()[0]] +
                       '</b> with a probability of ' +
                       str(class_prob_pruning.cpu().detach().numpy()[0]) +
                       '%<br>', "file_name": 'https://bedrockapijmm.s3.us-east-1.amazonaws.com/' +
                       file_name}
        else:
            message = {
                "answer": response_text['content'][0]['text'],
                "file_name": 'https://bedrockapijmm.s3.us-east-1.amazonaws.com/' +
                file_name}
    else:
        message = {
            "answer": response_text['content'][0]['text'],
            "file_name": 'https://bedrockapijmm.s3.us-east-1.amazonaws.com/gray.jpg'}
        data_cumm.append(username + ': ' + text + '\n ' + ' ChatBot: ' +
                         response_text['content'][0]['text'] + '\n ' + time_now + '\n ')

    # save in the database after 2 interactions
    if interactions % 2 == 0:

        # random 16-char  code generator for each successful prediction and
        # ID generator
        code_str = ''.join(random.choice(string.ascii_letters)
                           for i in range(16))
        id = random.randint(0, 5000)

        # create the empty ids.txt file
        if not os.path.exists('/tmp/ids.txt'):
           #with open("/tmp/ids.txt", 'w') as file:
           #     pass
           # download the ids.txt file
           # send the files to S3
           s3.meta.client.download_file(
                Filename='/tmp/ids.txt',
                Bucket='bedrockapijmm',
                Key='ids.txt')

        file_ids_r = open("/tmp/ids.txt", "r")
        lst = []
        for line in file_ids_r:
            lst.append(int(line.strip()))

        # check if the ids are repeated
        while id in lst:
            id = random.randint(0, 5000)

        # create file with new ids to save in the database
        with open("/tmp/ids.txt", "a+") as file_ids:
            file_ids.write(str(id) + '\n')

        # after you write the file load it again on the bucket
        s3.meta.client.upload_file(
                Filename='/tmp/ids.txt',
                Bucket='bedrockapijmm',
                Key='ids.txt')

        time_after = datetime.datetime.now().strftime("%I:%M:%S%p-%B-%d-%Y")

        # fill the database item values
        register = Data(
            id=id,
            code=code_str,
            date=time_after,
            username=username,
            Interaction_Register=data_cumm,
            Images_Files=img_cumm)
        data_cumm = []
        img_cumm = []

        # add a new register to the database
        db.session.add(register)
        db.session.commit()

        # generate the database_copy folder
        # if not os.path.exists('./database_copy/'):
        #    os.makedirs('./database_copy/', exist_ok=True)

        # generate the database_copy folder
        # if not os.path.exists('./database_copy/'):
        #    os.makedirs('./database_copy/', exist_ok=True)

        # copy the files from a database to the other
        # if os.path.exists('/var/lib/postgresql/15/main/'):
        #    shutil.copytree(
        #        '/var/lib/postgresql/15/main/',
        #        './database_copy/',
        #        dirs_exist_ok=True)

    interactions = interactions + 1

    return jsonify(message)


@app.post("/img_capture")
def img_capture():

    """
     This function captures the image information after user click on
     "attach image". Thus function also accumulate the image inputs
     for database saving and update the image status as a new base64
     image input for the rest of the Flask app execution.

     :return jsonify(message): the jsonified object of the message
     coming from the bedrock endpoint response in "answer" and the returned
     image status in "file_name" with the updated image input coming from the front
     to the APIm, in case the image is presented.
     :rtype jsonify(message): json dict/map
    """

    global img
    global img_base64
    global img_cumm
    global file_name
    # This is the img input of the chatbox
    results_img = request.get_json().get("results_img")

    file_name = request.get_json().get("file_name")

    img = base64.b64decode(results_img.split(',')[1])
    img_base64 = results_img.split(',')[1]

    # generate image list when the image query is requested
    img_cumm.append(file_name + ': ' + img_base64 + '; ')

    print(file_name + ': ' + img_base64 + '; ')
    message = {
        "answer": "loading img!",
        "file_name": 'https://bedrockapijmm.s3.us-east-1.amazonaws.com/gray.jpg'}
    return jsonify(message)

# set handler
def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)

if __name__ == "__main__":
    # create the table in the database
    # run the main in localhost
    # app.run(port=5000, debug=True)
    # run this for Docker
    app.run(host='0.0.0.0', port=8000, debug=True)
