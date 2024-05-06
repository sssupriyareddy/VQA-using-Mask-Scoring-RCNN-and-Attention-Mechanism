import os
import requests
from flask import Flask, flash, request,redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.io import imread

import pandas as pd
mapping = {}
global flag
flag=False
def clean(mapping):
    for key, questions in mapping.items():
        for i in range(len(question)):
            # take one question at a time
            question = questions[i]
            # preprocessing steps
            # convert to lowercase
            question = question.lower()
            # delete digits, special chars, etc., 
            question = question.replace('[^A-Za-z]', '')
            # delete additional spaces
            question = question.replace('\s+', ' ')
            # add start and end tags to the question
            question =  " ".join([word for word in question.split() if len(word)>1])
            questions[i] = question
def adaptive():
    # create mapping of image to questions
    
    # process lines
    for line in tqdm(questions_doc.split('\n')):
        # split the line by comma(,)
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, question = tokens[0], tokens[1:]
        # remove extension from image ID
        image_id = image_id.split('.')[0]
        # convert question list to string
        question = " ".join(question)
        # create list if needed
        if image_id not in mapping:
            mapping[image_id] = []
        # store the question
        mapping[image_id].append(question)
        clean(mappling)
def imageClassification(img_path):
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np

    # Load pre-trained ResNet-50 model
    model = ResNet50(weights='imagenet')

    # Load and preprocess an image
    #img_path = 'image1.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model
    # Make predictions
    
import tensorflow as tf
import numpy as np
import cv2

# Define the backbone CNN (e.g., ResNet50)
def build_backbone(input_shape):
    backbone = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False)
    return backbone

# Region Proposal Network (RPN)
def build_rpn(backbone):
    rpn_layer = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(backbone.output)
    rpn_class = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax', name='rpn_class')(rpn_layer)
    rpn_bbox = tf.keras.layers.Conv2D(4, (1, 1), activation='linear', name='rpn_bbox')(rpn_layer)
    return [rpn_class, rpn_bbox]

# Region of Interest (RoI) Pooling
def apply_roi_pooling(feature_maps, rois):
    # Implement RoI pooling here (e.g., using tf.image.crop_and_resize)
    return pooled_features

# Fully Connected Layers for classification and bounding box regression
def build_fully_connected_layers(pooled_features):
    fc_layers = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu'))(pooled_features)
    class_prediction = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(fc_layers)
    bbox_prediction = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='linear'))(fc_layers)
    return [class_prediction, bbox_prediction]




def prediction(img):
        
    # Define input shape and number of classes
    input_shape = (224, 224, 3)
    num_classes = 81  # Example number of classes (COCO dataset)

    # Build the backbone CNN
    backbone = build_backbone(input_shape)

    # Build the Region Proposal Network (RPN)
    rpn_outputs = build_rpn(backbone)

    # Build the Mask R-CNN model
    model = tf.keras.Model(inputs=backbone.input, outputs=rpn_outputs)

    # Compile the model (you may need to define losses and metrics)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load and preprocess an example image
    image = cv2.imread(img)
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32)


    predictions = model.predict(image)
    return predictions,False

data=pd.read_csv("input/dataset/data.csv")


import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# Function to remove stop words from a sentence
def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def sentence_Encoding(sentence1, sentence2):
    sentence1 = remove_stopwords(sentence1)
    sentence2 = remove_stopwords(sentence2)
    
    # Tokenize the sentences and convert them to lowercase
    vectorizer = CountVectorizer().fit_transform([sentence1.lower(), sentence2.lower()])
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(vectorizer)
    
    return similarity_score[0][1]
def train():
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical

    # Load and preprocess CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)

    # Create ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(test_images, test_labels)
    
UPLOAD_FOLDER = './static/questionimage'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "m4xpl0it"
dname="input/dataset/images/"
@app.route('/empty_page')
def empty_page():
    filename = session.get('filename', None)
    os.remove(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for('index'))

@app.route('/image')
def image():
    return render_template("image.html")

@app.route('/text')
def text():
    return render_template("text.html",msg="")



@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')
@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    try:
        flag=False
        features={}
        if request.method == 'POST':
            f = request.files['bt_image']
            q = request.form['ques']
            print(q)
            
            filename = str(f.filename)
    
            if filename!='':
                ext = filename.split(".")
                
                if ext[1] in ALLOWED_EXTENSIONS:
                    filename = secure_filename(f.filename)
                    f.save("static/test.png")
                    fnamesd=[f for f in os.listdir("input/dataset/images") ]
                    
                    file1="static/test.png"
                    finalname=""
                    for file in fnamesd:
                        try:
                            
                            file=dname+file
                            image1 = imread(file1, as_gray=True)
                            image2 = imread(file, as_gray=True)
                            
                            '''
                            if flag:
                                predictions,flag = prediction(file)
                                features[file]=decode_predictions(predictions, top=3)[0]
                            '''
                            ssim_score = ssim(image1, image2, data_range=image1.max() - image1.min())
                            
                            mse_score = mse(image1, image2)
                            if ssim_score > 0.9 and mse_score < 0.1:  # Adjust thresholds as needed
                                finalname=file
                                break
                        except:
                            pass
                            
                    
                    imageid=(finalname[21:]).split(".")[0]
                    df = data[data['image_id']==imageid]
                    print("Image ID ",imageid)
                    
                    m=0
                    qes=""
                    a=""
                    
                    for i, row in df.iterrows():
                        similarity = sentence_Encoding(row['question'],q)
                        if similarity>m:
                            m=similarity
                            #qes=row['question']
                            a=row['answer']
                
                
                    return render_template("pred.html",pred=a,q=q)    
                    
    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return render_template('index.html')

if __name__=="__main__":
    app.run(port=3000,debug=True)
    
