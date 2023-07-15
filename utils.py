from datetime import datetime
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import warnings
from newspaper import Article
import nltk
import sys
sys.path.append('/apps/my-website')
# Suppress the RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VISITS_DB_FILE = "visits_db.txt"

def get_counts():
    with open(VISITS_DB_FILE, "r") as file:
        lines = file.readlines()

    # Count the number of existing visits
    visit_count = len(lines)
    return visit_count

def increment_visit_count():
    # Read the existing visit records from the file
    with open(VISITS_DB_FILE, "r") as file:
        lines = file.readlines()

    # Count the number of existing visits
    visit_count = len(lines)

    # Increment the visit count by 1
    visit_count += 1

    # Append the new visit record to the file
    with open(VISITS_DB_FILE, "a") as file:
        file.write(f"{visit_count},{datetime.now()},\n")

    return visit_count

def delete_files_in_directory(directory_path):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)

# with open("imagenet_classes.txt", "r") as f:
# Get the directory path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path relative to the current file's directory
file_path = os.path.join(current_dir, "imagenet_classes.txt")

# Open the file
with open(file_path, "r") as f:
    categories = [s.strip() for s in f.readlines()]


def preprocess_image(input_image):
    # Convert the image to RGB mode if it has an alpha channel
    if input_image.mode == 'RGBA':
        input_image = input_image.convert('RGB')
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)  
    return input_batch

def load_our_model():
    model_path = 'model/squeezenet_model.pth'
    # model = models.squeezenet1_1(pretrained=False)
    model = models.squeezenet1_1(weights=None)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_our_model()

def get_class_from_probability(probabilities):
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = []
    for i in range(top5_prob.size(0)):
        result.append((i+1, categories[top5_catid[i]], round(top5_prob[i].item() * 100, 2)))
    return result

def get_top5_class_probabilities(adversarial_probabilities, original_result):
    # Get the class labels from the original result
    original_classes = [item[1] for item in original_result]

    # Find the indices of original classes in the adversarial probabilities
    original_indices = [categories.index(cls) for cls in original_classes]

    # Extract the probabilities of original classes from the adversarial probabilities
    top5_class_probabilities = []
    for i, index in enumerate(original_indices):
        probability = adversarial_probabilities[index]
        class_name = original_result[i][1]
        top5_class_probabilities.append((i + 1, class_name, round(probability.item() * 100, 1)))

    return top5_class_probabilities

def get_perturbation_adv_img(input_image, perturbation):
    '''
    input_image: A PIL image of size 512, 512
    perturbation: perturbation = epsilon * torch.sign(data_grad)
    perturbation is of type torch.Tensor of size 224, 224
    with values from -epsilon to +epsilon
    '''
    resized_input = input_image.resize((224, 224))
    perturbation_arr = perturbation.squeeze().permute(1, 2, 0).detach().numpy()
    resized_input_arr = np.array(resized_input)
    adversarial_image_array = resized_input_arr.astype(np.float16) + perturbation_arr.astype(np.float16)
    adversarial_image_array_clip = np.clip(adversarial_image_array, 0, 255)
    # adversarial_image = Image.fromarray(adversarial_image_array_clip, mode='RGB')
    adversarial_image = Image.fromarray(adversarial_image_array_clip.astype(np.uint8), mode='RGB')
    adversarial_image = adversarial_image.resize((512, 512))
    # display(input_image)
    # display(adversarial_image)
    multipled_perturbation_arr = (perturbation_arr * 255 * 50).astype(np.uint8)
    multipled_perturbation = Image.fromarray(np.abs(multipled_perturbation_arr), mode='RGB')
    multipled_perturbation = multipled_perturbation.resize((512, 512))
    return adversarial_image, multipled_perturbation


def get_inference_result(input_image: Image.Image) -> list:

    input_batch = preprocess_image(input_image)
    with torch.no_grad():
        output = model(input_batch)

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    probabilities = torch.nn.functional.softmax(output[0], dim=0) 

    result = get_class_from_probability(probabilities)

    return result

def get_adversarial_result(input_image, image_filename):
    input_batch = preprocess_image(input_image)

    # Perform FGSM attack on the input image
    input_batch.requires_grad = True
    with torch.enable_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    original_result = get_class_from_probability(probabilities)
    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    loss = torch.nn.functional.nll_loss(output, predicted_class)
    model.zero_grad()
    loss.backward()

    # Collect the gradients of the input image
    data_grad = input_batch.grad.data

    # Using a very high epsilon value so change is evident
    epsilon=24/255
    # Create the adversarial image by perturbing the input image
    perturbation = epsilon * torch.sign(data_grad)
    perturbed_image = input_batch + perturbation

    # Clamp the pixel values to the valid range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    with torch.no_grad():
        adversarial_output = model(perturbed_image)

    adversarial_probabilities = torch.nn.functional.softmax(adversarial_output[0], dim=0)

    adversarial_result = get_class_from_probability(adversarial_probabilities)

    adversarial_top5_class_probabilities = get_top5_class_probabilities(adversarial_probabilities, original_result)

    adversarial_image, multipled_perturbation = get_perturbation_adv_img(input_image, perturbation)
    '''    
    perturbation_arr = perturbation.squeeze().permute(1, 2, 0).detach().numpy()
    multipled_perturbation_arr = (perturbation_arr * 255 * 50).astype(np.uint8)
    multipled_perturbation = Image.fromarray(multipled_perturbation_arr, mode='RGB')
    multipled_perturbation = multipled_perturbation.resize((512, 512))
    normal_perturbation = Image.fromarray((perturbation * 255).astype(np.uint8), mode='RGB')

    resized_perturbation = normal_perturbation.resize(input_image.size)

    adversarial_image_array = np.array(resized_perturbation).astype(np.uint16) + np.array(input_image).astype(np.uint16)
    # Ensure the resulting array is within the valid range [0, 255]
    adversarial_image_array = np.clip(adversarial_image_array, 0, 255)
    # Convert the adversarial_image_array  back to PIL.Image.Image
    adversarial_image = Image.fromarray(adversarial_image_array.astype(np.uint8))
    '''
    # Create a new blank image with the stacked dimensions
    stacked_width = input_image.width + adversarial_image.width + 1  # Add 1 pixel for the boundary
    stacked_height = max(input_image.height, adversarial_image.height)
    stacked_image = Image.new('RGB', (stacked_width, stacked_height))

    # Paste the input_image on the left and adversarial_image on the right
    stacked_image.paste(input_image, (0, 0))
    stacked_image.paste(adversarial_image, (input_image.width + 1, 0))  # Add 1 pixel offset for the boundary

    # Add a boundary line
    boundary_color = (255, 0, 0)  # Red color for the boundary line
    boundary_width = 2  # Width of the boundary line
    line_coordinates = [(input_image.width, 0), (input_image.width, stacked_height - 1)]  # Coordinates for the line
    line_draw = ImageDraw.Draw(stacked_image)
    line_draw.line(line_coordinates, fill=boundary_color, width=boundary_width)

    # Add labels to the images
    label_font = ImageFont.load_default()
    label_draw = ImageDraw.Draw(stacked_image)

    # Add label for the original image
    label_draw.text((10, 10), "Original", fill=(112, 238, 27, 1), font=label_font)

    # Add label for the adversarial image
    label_draw.text((input_image.width + 10, 10), "Adversarial", fill="red", font=label_font)

    # Save the stacked image
    stacked_save_path = f'static/stacked/{image_filename}'
    os.makedirs(os.path.dirname(stacked_save_path), exist_ok=True)
    stacked_image.save(stacked_save_path)

    # Save the perturbation scaled by 50
    scaled_save_path = f'static/perturbations/{image_filename}'
    os.makedirs(os.path.dirname(scaled_save_path), exist_ok=True)
    multipled_perturbation.save(scaled_save_path)

    # Save the perturbed input image
    adversarial_image_path = f'static/adversarial/{image_filename}'
    os.makedirs(os.path.dirname(adversarial_image_path), exist_ok=True)
    adversarial_image.save(adversarial_image_path)



    return original_result, adversarial_result, adversarial_top5_class_probabilities



def generate_summary(url: str) -> str:
    # print("INSIDE SUMMARY")
    article = Article(url)
    article.download()
    article.parse()
    # print(article.authors, type(article.authors))
    if not article.authors:
        article.authors = ''
    if not article.publish_date:
        article.publish_date = ''
    # print(article.publish_date, type(article.publish_date))
    # print(article.top_image, type(article.top_image))
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    finally:
        article.nlp()
    # return article.summary
    return {"Authors": article.authors, 
            "Date": article.publish_date, 
            "Image": article.top_image, 
            "Summary": article.summary
    }


