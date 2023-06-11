from memory_profiler import profile

@profile
def my_fun():
    # Convert the uploaded image to an PIL.Image.Image
    filename = '/Users/hrushi/Desktop/dove.jpg'
    with Image.open(filename) as input_image:
       input_image.load()
    input_image = input_image.convert('RGB')
    # Let's have a standard image size to avoid resizing with html
    input_image = input_image.resize((512, 512))
    input_image.save('delete_later.png')

    # Perform inference using the get_inference_result function
    if input_image.mode == 'RGBA':
        input_image = input_image.convert('RGB')

    model_path = 'model/squeezenet_model.pth'
    # model = models.squeezenet1_1(pretrained=False)
    model = models.squeezenet1_1(weights=None)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)  
    with torch.no_grad():
         output = model(input_batch)

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    probabilities = torch.nn.functional.softmax(output[0], dim=0) 
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = []
    for i in range(top5_prob.size(0)):
        result.append((i+1, categories[top5_catid[i]], round(top5_prob[i].item() * 100, 2)))
    return result

@profile
def adv_fun():
    from PIL import Image
    import torch
    import torchvision.models as models
    from PIL import Image
    from PIL import ImageDraw, ImageFont
    import torch.nn.functional as F
    import numpy as np
    import warnings
    # Convert the uploaded image to an PIL.Image.Image
    filename = '/Users/hrushi/Desktop/dove.jpg'
    with Image.open(filename) as input_image:
       input_image.load()
    input_image = input_image.convert('RGB')
    # Let's have a standard image size to avoid resizing with html
    input_image = input_image.resize((512, 512))
    input_image.save('delete_later.png')

    # Perform inference using the get_inference_result function
    if input_image.mode == 'RGBA':
        input_image = input_image.convert('RGB')

    model_path = 'model/squeezenet_model.pth'
    # model = models.squeezenet1_1(pretrained=False)
    model = models.squeezenet1_1(weights=None)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    def preprocess(image):
        # Resize the image to 256x256
        resized_image = F.interpolate(image, size=256, mode='bilinear', align_corners=False)

        # Center crop the image to 224x224
        center_cropped_image = resized_image[:, :, 16:240, 16:240]

        # Convert the image to tensor
        tensor_image = center_cropped_image.to(torch.float32)

        # Normalize the image
        normalized_image = (tensor_image / 255.0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])

        return normalized_image

    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)  
    input_batch.requires_grad = True
    with torch.enable_grad():
        output = model(input_batch)

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    probabilities = torch.nn.functional.softmax(output[0], dim=0) 
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = []
    for i in range(top5_prob.size(0)):
        result.append((i+1, categories[top5_catid[i]], round(top5_prob[i].item() * 100, 2)))

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

    top5_prob, top5_catid = torch.topk(adversarial_probabilities, 5)
    original_result = []
    for i in range(top5_prob.size(0)):
        original_result.append((i+1, categories[top5_catid[i]], round(top5_prob[i].item() * 100, 2)))
    
    # Get the class labels from the original result
    original_classes = [item[1] for item in original_result]

    # Find the indices of original classes in the adversarial probabilities
    original_indices = [categories.index(cls) for cls in original_classes]

    # Extract the probabilities of original classes from the adversarial probabilities
    adversarial_top5_class_probabilities = []
    for i, index in enumerate(original_indices):
        probability = adversarial_probabilities[index]
        class_name = original_result[i][1]
        adversarial_top5_class_probabilities.append((i + 1, class_name, round(probability.item() * 100, 1)))

    
    perturbation = perturbation.squeeze().permute(1, 2, 0).detach().numpy()
    multipled_perturbation = Image.fromarray((perturbation * 255 * 50).astype(np.uint8), mode='RGB')
    multipled_perturbation = multipled_perturbation.resize((512, 512))
    normal_perturbation = Image.fromarray((perturbation * 255).astype(np.uint8), mode='RGB')

    resized_perturbation = normal_perturbation.resize(input_image.size)

    # We can directly add using Pillow
    
    #resized_perturbation_rgb = resized_perturbation.convert("RGB")
    #input_image_rgb = input_image.convert("RGB")
    #adversarial_image = Image.blend(resized_perturbation_rgb, input_image_rgb, alpha=0.5)

    # OR WE CAN CONVERT TO NUMPY TO GET BETTER ADDITION
    # Convert images to NumPy arrays and perform addition of the two arrays
    adversarial_image_array = np.array(resized_perturbation).astype(np.uint16) + np.array(input_image).astype(np.uint16)
    # Ensure the resulting array is within the valid range [0, 255]
    adversarial_image_array = np.clip(adversarial_image_array, 0, 255)
    # Convert the adversarial_image_array  back to PIL.Image.Image
    adversarial_image = Image.fromarray(adversarial_image_array.astype(np.uint8))

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
    label_font = ImageFont.truetype("Arial.ttf", size=24)  # Set the font and size for the labels
    label_draw = ImageDraw.Draw(stacked_image)

    # Add label for the original image
    label_draw.text((10, 10), "Original", fill=(112, 238, 27, 1), font=label_font)

    # Add label for the adversarial image
    label_draw.text((input_image.width + 10, 10), "Adversarial", fill="red", font=label_font)

    # Save the stacked image
    stacked_save_path = f'del1.png'
    stacked_image.save(stacked_save_path)

    # Save the perturbation scaled by 50
    scaled_save_path = f'del2.png'
    multipled_perturbation.save(scaled_save_path)

    # Save the perturbed input image
    adversarial_image_path = f'del3.png'
    adversarial_image.save(adversarial_image_path)    

if __name__ == '__main__':
   print('here')
   #inference_result = my_fun()
   aa = adv_fun()
   print(aa)