import streamlit as st
import os
import open_clip
import torch
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from utils import icon
import tempfile
import hashlib
import io
# from streamlit_option_menu import option_menu

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

@st.cache_resource
def load_model(selected_option):
    if selected_option == 'ViT-B':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif selected_option == 'ViT-L':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', pretrained='laion2b_s32b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model.eval()
    return model, preprocess, tokenizer


def hash_bytes(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

@st.cache_data
def process_uploaded_image(file_bytes, _model, _preprocess, _text_features, device):
    """Classify an uploaded image against CIFAR-100 labels."""
    key = hash_bytes(file_bytes)
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_input = _preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = _model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ _text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(5)
    class_names = [cifar100_classes[i] for i in indices]
    percentages = [100 * v.item() for v in values]

    return pd.DataFrame({"Class Name": class_names, "Percentage": percentages})


@st.cache_data
def process_image(image_path):
    # Load the image
    image = Image.open(image_path)
    image_width = 450

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = tokenizer([f'a photo of a {c}' for c in cifar100_classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(5)
    top_probs, top_labels = similarity.topk(5, dim=-1)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100_classes[index]:>16s}: {100 * value.item():.2f}%")

    # Empty lists to store the results
    class_names = []
    percentages = []

    # Iterate over values and indices
    for value, index in zip(values, indices):
        # Get the class name and percentage
        class_name = cifar100_classes[index]
        percentage = 100 * value.item()

        # Append to the lists
        class_names.append(class_name)
        percentages.append(percentage)

    # Create the DataFrame
    df = pd.DataFrame({
        'Class Name': class_names,
        'Percentage': percentages
    })

    # Sort the DataFrame by Percentage
    df = df.sort_values(by='Percentage', ascending=False)

    return df

def process_image_labels(image_path, labels):
    # Load the image
    image = Image.open(image_path)
    image_width = 450

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = tokenizer([f'a photo of a {c}' for c in labels]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(len(labels))
    top_probs, top_labels = similarity.topk(len(labels), dim=-1)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")

    # Empty lists to store the results
    class_names = []
    percentages = []

    # Iterate over values and indices
    for value, index in zip(values, indices):
        # Get the class name and percentage
        class_name = labels[index]
        percentage = 100 * value.item()

        # Append to the lists
        class_names.append(class_name)
        percentages.append(percentage)

    # Create the DataFrame
    df = pd.DataFrame({
        'Class Name': class_names,
        'Percentage': percentages
    })

    # Sort the DataFrame by Percentage
    df = df.sort_values(by='Percentage', ascending=False)

    return df

def process_image_labels_binary(image_path, label):
    # Load the image
    image = Image.open(image_path)
    image_width = 450

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)
    labels = ['Yes.', 'No.']
    # Use a general text prompt as a baseline
    text_prompts = [f'Does this image contain {label}?', 'Does this image have something?']
    text_inputs = tokenizer(text_prompts).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(len(labels))
    top_probs, top_labels = similarity.topk(len(labels), dim=-1)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")

    # Empty lists to store the results
    class_names = []
    percentages = []

    # Iterate over values and indices
    for value, index in zip(values, indices):
        # Get the class name and percentage
        class_name = labels[index]
        percentage = 100 * value.item()

        # Append to the lists
        class_names.append(class_name)
        percentages.append(percentage)

    # Create the DataFrame
    df = pd.DataFrame({
        'Class Name': class_names,
        'Percentage': percentages
    })

    # Sort the DataFrame by Percentage
    df = df.sort_values(by='Percentage', ascending=False)

    return df



@st.cache_resource
def get_cifar100_text_features(_model, _tokenizer, device, class_list):
    text_inputs = _tokenizer([f'a photo of a {c}' for c in class_list]).to(device)
    with torch.no_grad():
        text_features = _model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

############################################################################################################
# Load the CIFAR-100 classes
############################################################################################################


cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
                    'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 
                    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
                     'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 
                     'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 
                     'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 
                     'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

############################################################################################################
# UI configurations
############################################################################################################

st.set_page_config(page_title="Home - Clip Model Prototype",
                   #page_icon=":desktop_computer:",
                   initial_sidebar_state="auto",
                   layout="wide"             
                   )

############################################################################################################
# UI sidebar - Menu
############################################################################################################

with st.sidebar:
     st.title(":blue[Image Classification Prototype: Clip Model]")
     st.divider()
     st.markdown('<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.2;"><strong>Contact us: <br></br>Isaac Bravo:</strong> <a href="https://www.linkedin.com/in/isaac-bravo/"><img src="https://openvisualfx.com/wp-content/uploads/2019/10/linkedin-icon-logo-png-transparent.png" width="20" height="20"></a><a href="https://github.com/IsaacBravo"><img src="https://www.pngarts.com/files/8/Github-Logo-Transparent-Background-PNG-420x236.png" width="35" height="20"></a><br></br><strong>Katharina Prasse:</strong> <a href="https://www.linkedin.com/in/katharina-prasse/"><img src="https://openvisualfx.com/wp-content/uploads/2019/10/linkedin-icon-logo-png-transparent.png" width="20" height="20"></a><a href="https://github.com/KathPra"><img src="https://www.pngarts.com/files/8/Github-Logo-Transparent-Background-PNG-420x236.png" width="35" height="20"></a></p>', 
                 unsafe_allow_html=True)
     st.divider()
     sidebar_title = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 15px; letter-spacing: -0.005em; line-height: 1.5;">This page is part of the ClimateVision project, academic initiative between the Technical University of Munich and the University of Mannheim. This project is founding by the Bundesministerium für Forschung, Technologie und Raumfahrt (BMFTR) and the European Union (EU). If you want to know more about the project, please check our website <a href="https://web.informatik.uni-mannheim.de/climatevisions/">here.</a></p>'
     st.markdown(sidebar_title, unsafe_allow_html=True)

############################################################################################################
# UI - Header
############################################################################################################

icon.show_icon(":desktop_computer:")

st.header(":blue[Welcome to the ClimateVision Project! 👋]")
original_header = '<p style="font-family:Source Sans Pro; text-align:justify; color:#1F66CB; font-size: 18px; letter-spacing: -0.005em; line-height: 1.5; background-color:#EBF2FC; padding:25px; border-radius:10px; border:1px solid graylight;">This page provides users with the ability to upload an image and receive predictions from OpenCLIP model, an open-source implementation of OpenAI\'s CLIP model. The CLIP model, short for "Contrastive Language-Image Pre-training," is a powerful artificial intelligence model capable of understanding both images and text. Using this model, the application predicts the class or content depicted in the uploaded image based on its visual features and any accompanying text description. By leveraging the CLIP model`s unique ability to analyze images in conjunction with text, users can gain insights into what the model perceives from both modalities, offering a richer understanding of the image content.</p>'
st.markdown(original_header, unsafe_allow_html=True)

st.markdown('<br></br>', unsafe_allow_html=True)

############################################################################################################
# UI - Pre-trained model selection
############################################################################################################

model_selection_title = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 25px; font-weight: 600; letter-spacing: -0.005em; line-height: 1.2;">Pre-trained model selection</p>'
st.markdown(model_selection_title, unsafe_allow_html=True)
model_selection_header = '<p style="font-family:Source Sans Pro; text-align:justify; color:#1F66CB; font-size: 18px; letter-spacing: -0.005em; line-height: 1.5; background-color:#EBF2FC; padding:25px; border-radius:10px; border:1px solid graylight;">OpenCLIP is pre-trained on the LAION-2B dataset, a massive dataset of image-text pairs, specifically designed to help train models that understand both visuals and text. You\'ll have three options for the visual backbone: ViT-B, ViT-L, and ViT-H. If you\'re looking for better performance, especially with more complex images, the larger backbones (ViT-L and ViT-H) can capture more detail and improve accuracy.</p>'
st.markdown(model_selection_header, unsafe_allow_html=True)

# Pre-trained models that are supported
model_options = ['ViT-B', 'ViT-L', 'ViT-H']
selected_option = st.selectbox('', model_options)

############################################################################################################
# Model selection
############################################################################################################

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# open_clip.list_pretrained()
model, preprocess, tokenizer = load_model(selected_option)

cifar100_text_features = get_cifar100_text_features(model, tokenizer, device, cifar100_classes)

############################################################################################################
# UI - User input and predictions # USER EXAMPLE 1
############################################################################################################

#st.info("""
#This page allows the user to upload an image and get predictions from the model. 
#The model is based on the CLIP model from OpenAI, which predicts the class of the image based on its text description. 
#""")
st.markdown('<br></br>', unsafe_allow_html=True)

original_title = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 25px; font-weight: 600; letter-spacing: -0.005em; line-height: 1.2;">Model example 1: A photo of a climate change</p>'
st.markdown(original_title, unsafe_allow_html=True)

original_title_text_1 = '<p style="font-family:Source Sans Pro; text-align:justify; color:#1F66CB; font-size: 18px; letter-spacing: -0.005em; line-height: 1.5; background-color:#EBF2FC; padding:25px; border-radius:10px; border:1px solid graylight;">The following example provide a glance on how CLIP Model works to label a climate change image.</p>'
st.markdown(original_title_text_1, unsafe_allow_html=True)
st.divider()

# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()

############################################################################################################
# style
############################################################################################################
th_props = [
  ('font-size', '20px'),
  ('font-weight', 'bold'),
  ('color', '#fff'),
  ('text-align', 'center'),
  ('text-shadow', '0 1px 0 #000'),
  ('background-color', 'blue'),
  ('padding', '5px 10px'),
  ('box-shadow', '0 0 20px rgba(0, 0, 0, 0.15)')
]
                               
td_props = [
  ('font-size', '17px'),
  ('text-align', 'center')
]

table_props = [
  ('border', '1px solid #6d6d6d'),
  ('border-radius', '30px'),
  ('overflow', 'hidden')
]
                                 
styles_dict = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props),
  dict(selector="table", props=table_props)
]

############################################################################################################
# Prepare the inputs -- EXAMPLE 1
############################################################################################################

image_path = 'climate_image.jpeg'
image = Image.open(image_path)
image_width = 450

result_df = process_uploaded_image(
        open(image_path, "rb").read(),
        model, preprocess, cifar100_text_features, device
    )

# UI - Original image and predictions (pre-loaded image)
grid_image, grid_predictions = st.columns([3,3])

with grid_image:
    example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
    st.markdown(example_text_1, unsafe_allow_html=True)
    # st.image(image, caption='Pre-loaded Image', width=image_width)
    st.image(image, caption='Pre-loaded Image',  use_container_width ='always')

with grid_predictions:
    example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions &#127919;</p>'
    st.markdown(example_text_2, unsafe_allow_html=True)
    # st.table(result_df.style.set_properties(**{'border-radius': '30px'}).set_table_styles(styles_dict))
    st.markdown("""
        <style>
        table {border-radius: 60px;}
        </style>
        """, unsafe_allow_html=True)
    st.dataframe(result_df.style.background_gradient(cmap='Blues'))

st.divider()

############################################################################################################
# UI - User input and predictions # USER EXAMPLE 1
############################################################################################################

user_example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Upload your own Image! &#128247;</p>'
st.markdown(user_example_text_1, unsafe_allow_html=True)
user_example_text_2 = '<p style="font-family:Source Sans Pro; text-align:justify; color:#1F66CB; font-size: 18px; letter-spacing: -0.005em; line-height: 1.5; background-color:#EBF2FC; padding:25px; border-radius:10px; border:1px solid graylight;">Now you can test the model with your image and see how it detects the elements.</p>'
st.markdown(user_example_text_2, unsafe_allow_html=True)

uploaded_file_example_1  = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="1")
st.warning("""
Please consider that analizing an image may take a few seconds.
""")

grid_image, grid_predictions = st.columns([3,3])

if uploaded_file_example_1 is not None:
    image_user = Image.open(uploaded_file_example_1 )
    # Create a temporary directory to save the uploaded file
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file_example_1 .name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_example_1 .getvalue())

    with grid_image:
        example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
        st.markdown(example_text_1, unsafe_allow_html=True)
        # st.image(image_user, caption='Uploaded Image', width=image_width)
        st.image(image_user, caption='Uploaded Image',  use_container_width ='always')
    with grid_predictions:
        result = process_image(file_path)
        example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions &#127919;</p>'
        st.markdown(example_text_2, unsafe_allow_html=True)
        st.dataframe(result.style.background_gradient(cmap='Blues'))
    st.divider()
else:
    st.write("Please upload an image. :point_up:")

st.divider()

############################################################################################################
# Prepare the inputs -- EXAMPLE 2
############################################################################################################

image_2 = Image.open('climate_image_2.jpeg')
image_path_2 = 'climate_image_2.jpeg'

original_title_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 25px; font-weight: 600; letter-spacing: -0.005em; line-height: 1.2;">Model example 2: A photo of a climate change (Defining input labels)</p>'
st.markdown(original_title_2, unsafe_allow_html=True)

original_title_text_2 = '<p style="font-family:Source Sans Pro; text-align:justify; color:#1F66CB; font-size: 18px; letter-spacing: -0.005em; line-height: 1.5; background-color:#EBF2FC; padding:25px; border-radius:10px; border:1px solid graylight;">The following example provide a glance on how CLIP Model works to label a climate change image, based on labels defined by the researcher.</p>'
st.markdown(original_title_text_2, unsafe_allow_html=True)
st.divider()

# UI - Original image and predictions (pre-loaded image)
# grid_image, grid_space, grid_predictions_1, grid_predictions_2 = st.columns([3,1,3,3])
grid_image, grid_predictions_1, grid_predictions_2 = st.columns([3,3,3])

result_df_labels_1 = process_image_labels_binary(image_path_2, 'flood')
result_df_labels_2 = process_image_labels(image_path_2, labels=['wildfires', 'drought', 'pollution', 'deforestation', 'flood'])

with grid_image:
    example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
    st.markdown(example_text_1, unsafe_allow_html=True)
    # st.image(image_2, caption='Pre-loaded Image', width=image_width)
    st.image(image_2, caption='Pre-loaded Image', use_container_width ='always')

with grid_predictions_1:
    example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Single-Label Predictions &#127919;</p>'
    st.markdown(example_text_2, unsafe_allow_html=True)
    st.dataframe(result_df_labels_1.style.background_gradient(cmap='Blues'))
    st.info("""In this example, we input the single label 'flood.' The model responds with either 'yes' or 'no' based on whether the image contains a flood.""")

with grid_predictions_2:
    example_text_2 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Multi-Label Predictions &#127919;</p>'
    st.markdown(example_text_2, unsafe_allow_html=True)
    st.dataframe(result_df_labels_2.style.background_gradient(cmap='Blues'))
    st.info("""In this example, we define the labels 'wildfires', 'drought', 'pollution', 'deforestation', and 'flood'. The model evaluates the image and returns the likelihood that it matches each label.""")

st.divider()


############################################################################################################
# UI - User input and predictions # USER EXAMPLE 2
############################################################################################################

# UI - User input and predictions
user_example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Upload your own Image! &#128247;</p>'
st.markdown(user_example_text_1, unsafe_allow_html=True)
user_example_text_2 = '<p style="font-family:Source Sans Pro; text-align:justify; color:#1F66CB; font-size: 18px; letter-spacing: -0.005em; line-height: 1.5; background-color:#EBF2FC; padding:25px; border-radius:10px; border:1px solid graylight;">Now you can test the model with your image and see how it detects the elements.</p>'
st.markdown(user_example_text_2, unsafe_allow_html=True)

uploaded_file_example_2 = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="2")
st.warning("""
Please consider that analizing an image may take a few seconds.
""")
st.write("Please upload an image. :point_up:")

grid_text_1, grid_text_2 = st.columns([3,3])
with grid_text_1:
    # user_example_text_3 = '<p style="font-family:Source Sans Pro; color:black; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Now you can define either one or a set of labels to classify your image: &#128073;</p>'
    # st.markdown(user_example_text_3, unsafe_allow_html=True)
    st.write("Now you can define either one or a set of labels to classify your image. :point_right:")
with grid_text_2:
    labels_user = st.text_input('Please enter one or more labels. If entering multiple labels, separate them with commas.')

grid_image, grid_predictions = st.columns([3,3])

if uploaded_file_example_2 is not None:
    image_user_example_2 = Image.open(uploaded_file_example_2)
    # Create a temporary directory to save the uploaded file
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file_example_2.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_example_2.getvalue())

    with grid_image:
        example_text_1 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Original Image &#128247;</p>'
        st.markdown(example_text_1, unsafe_allow_html=True)
        # st.image(image_user_example_2, caption='Uploaded Image', width=image_width)
        st.image(image_user_example_2, caption='Uploaded Image',  use_container_width ='always')
    with grid_predictions:
        if labels_user:
            example_text_3 = '<p style="font-family:Source Sans Pro; color:#2368CC; font-size: 20px; letter-spacing: -0.005em; line-height: 1.5;">Model Predictions &#127919;</p>'
            st.markdown(example_text_3, unsafe_allow_html=True)
            labels_user_list = [label.strip() for label in labels_user.split(',')]
            if len(labels_user_list) == 1:
                result_df_labels_3 = process_image_labels_binary(file_path, labels_user_list[0])
            else:
                result_df_labels_3 = process_image_labels(file_path, labels=labels_user_list)
            st.dataframe(result_df_labels_3.style.background_gradient(cmap='Blues'))
        else:
            st.info("Please enter one or more labels. If entering multiple labels, separate them with commas.")
else:
    st.write("")

st.divider()
