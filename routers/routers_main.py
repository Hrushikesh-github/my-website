from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File, Request, APIRouter, Request
from fastapi.responses import HTMLResponse
import io
import os
from PIL import Image
from utils import generate_summary, get_counts, get_inference_result, get_adversarial_result, increment_visit_count, delete_files_in_directory

# Create an instance of the Jinja2Templates class
templates = Jinja2Templates(directory="templates")

#UPLOAD_DIR = '/Users/hrushi/im/fastapi-image/static/uploads'
UPLOAD_DIR = 'static/uploads'


router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    visits = get_counts()
    return templates.TemplateResponse("home.html", {"request": request, "visits": visits})

@router.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    delete_files_in_directory(os.path.join('static', 'stacked'))
    delete_files_in_directory(os.path.join('static', 'perturbations'))
    delete_files_in_directory(os.path.join('static', 'uploads'))
    delete_files_in_directory(os.path.join('static', 'adversarial'))
    visits = get_counts()
    return templates.TemplateResponse("about.html", {"request": request, "visits": visits})

@router.get("/image_classify")
def image_upload_page(request: Request):
    #return templates.TemplateResponse("upload_bootstap.html", {"request": request})
    visits = get_counts()
    return templates.TemplateResponse("image_classification.html", {"request": request, "visits": visits})

@router.post("/process_image")
async def process_image(request: Request, image: UploadFile = File(...)):
    # Access the image data using `image.file` or `image.read`

    # Save the uploaded image to the specified directory
    file_location = os.path.join(UPLOAD_DIR, image.filename)
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    contents = await image.read()

    # Convert the uploaded image to an PIL.Image.Image
    image_bytes = io.BytesIO(contents)
    input_image = Image.open(image_bytes)
    input_image = input_image.convert('RGB')
    # Let's have a standard image size to avoid resizing with html
    input_image = input_image.resize((512, 512))
    input_image.save(file_location)
    
    # Perform inference using the get_inference_result function
    inference_result = get_inference_result(input_image)

    # Generate the URL for the uploaded image
    image_url = request.url_for("static", path=f"uploads/{image.filename}")
    visits = increment_visit_count()
    #return templates.TemplateResponse("upload_bootstap.html", {"request": request, "image_url": image_url, "inference_result": inference_result})
    return templates.TemplateResponse("image_classification.html", {"request": request, "image_url": image_url, "inference_result": inference_result, "visits": visits})

@router.get("/adversarial")
def upload_adversarial_page(request: Request):
    visits = get_counts()
    return templates.TemplateResponse("adversarial_stack.html", {"request": request, "visits": visits})

@router.post("/process_adversarial_image")
async def process_adversarial_image(request: Request, image: UploadFile = File(...)):

    # Save the uploaded image to the specified directory
    file_location = os.path.join(UPLOAD_DIR, image.filename)
    contents = await image.read()
    # with open(file_location, "wb") as file:
    #     contents = await image.read()
    #     file.write(contents)

    # Convert the uploaded image to an array
    image_bytes = io.BytesIO(contents)
    input_image = Image.open(image_bytes)
    input_image = input_image.convert('RGB')
    # Let's have a standard image size to avoid resizing with html
    input_image = input_image.resize((512, 512))
    input_image.save(file_location)

    # Convert the PIL image to a NumPy array
    #image_array = np.array(input_image)

    # Perform inference using the get_inference_result function
    original_result, adversarial_result, adversarial_top5_class_probabilities = get_adversarial_result(input_image, image.filename)
    # Generate the URL for the uploaded image
    image_urls = {
    'original_image_url' : request.url_for("static", path=f"uploads/{image.filename}"),
    'adversarial_image_url' : request.url_for("static", path=f"adversarial/{image.filename}"),
    'stacked_image_url' : request.url_for("static", path=f"stacked/{image.filename}"),
    'perturbation_image_url' : request.url_for("static", path=f"perturbations/{image.filename}"),
    }
    
    visits = increment_visit_count()

    return templates.TemplateResponse("adversarial_stack.html", {"request": request, "image_urls": image_urls, "inference_result": original_result, "adversarial_inference_result": adversarial_result, "adversarial_original_top5": adversarial_top5_class_probabilities, "visits": visits})

@router.post("/post_get_summary", response_class=HTMLResponse)
async def post_get_summary(request: Request):
    form_data = await request.form()
    url = form_data["url"]
    try:
        article_info = generate_summary(url)
        visits = increment_visit_count()
        return templates.TemplateResponse("summary.html", {"request": request, "url": url, "article_info":article_info, "visits": visits})
    except:
        article_info = {"Summary": "Failed to obtain the summary. The website's security measures prevent programmatic access to the data"}
        return templates.TemplateResponse("summary.html", {"request": request, "url": url, "article_info":article_info})
    
@router.get("/get_summary")
def summary_page(request: Request):
    #return templates.TemplateResponse("upload_bootstap.html", {"request": request})
    visits = get_counts()
    return templates.TemplateResponse("summary.html", {"request": request, "visits": visits})


@router.get("/old_get_summary", response_class=HTMLResponse)
async def old_obtain_summary(request: Request, url: str):
    print("INSIDE OBTAIN SUMMARY")
    #url = 'https://testdriven.io/'
    try:
        output_dict = generate_summary(url)
    except:
        return "Failed to obtain the summary. The website's security measures prevent programmatic access to the data."
    return output_dict
    # visits = get_counts()
    #return templates.TemplateResponse("home.html", {"request": request, "visits": visits})

@router.get("/telugu_words", response_class=HTMLResponse)
async def render_telugu_page(request: Request):
    visits = get_counts()
    return templates.TemplateResponse("telugu_words.html", {"request": request, "visits": visits})