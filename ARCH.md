= SPEC-001: Cross Stitch Pattern Generator Application =

== Background == This application enables users to generate cross stitch patterns from images providef via direct upload or via a shared link. Application backend processes the image to create a simplified cross-stitch or embroidery pattern using a variety of image analysis techniques. Application front-end allows users to preview and adjust patterns, while backend processes adjust the image according to user-specified parameters like color count, image type, and other customizations.

== Requirements ==

#### Must Haves:
1. Web-based front-end built with React that allows users to:
  . Upload images or provide an image link
  . Specify image type as "painting", "illustration", or "icon", where image types are described in greater detail as:
    . "Painting": Generates patterns using the entire image, where users specify how many unique colors (threads) should be included in the resulting pattern.
    .  "Icon": Generates single-color patterns, ideal for distinct shapes or lettering.
    .  "Illustration": Focuses on foreground subjects and uses background removal and dynamic cropping strategies to isolate as focal point.
  . Adjust pattern parameters, such as:
    . Color count (for "painting")
    . Thread brand for color matching (e.g., DMC, Anchor)
  . Preview generated cross stitch patterns in real-time.
  . Download patterns in a usable format (PDF, SVG, or PNG).
2. Backend API (using Flask or FastAPI) that handles:
  . Image uploads and processing.
  . Cross stitch pattern generation, based on image type and selected parameters.
  . Return of generated patterns to front-end for rendering and downloading.
3. Pattern generation strategies based on image type, for example:
  . "painting": Applies KMeans clustering to limit colors.
  . "icon": Generates single-color patterns.
  . "illustration": Applies content-aware cropping and employs varuiys background removal strategies, including:
    . Edge Detection for Bounding Box: Identify subject boundaries and crop.
    . Golden Ratio Cropping: Crop based on the aesthetically pleasing area around the focal point.
    . Aspect Ratio Preservation: Uses padding as necessary to preserve aspect ratio.
    . Background Subtraction: Uses color analysis to detect and remove backgrounds.

#### Should Haves:
1. User authentication for saving and managing generated patterns.
2. Real-time pattern updates according to adjusted parameters.
3. Fine-tuning of options to adjust the after initial generation (e.g., minor edits to color areas, thread selection).

#### Could Haves:
1. Machine learning-based saliency detection or object detection for automatic cropping and focal point detection in "illustration" mode.
2. Collaborative features for multiple users to work on patterns in future iterations.

#### Won't Have (for now):
Multi-user collaboration on patterns.


== Method ==  Technical solution consists of three key components:

  1. Front-End (React)
  2. Backend API (Flask/FastAPI)
  3. Pattern Generation Logic

Below is a detailed breakdown of each component.

### Front-End (React)
The front-end serves as (UI) user interface and allows users to upload images, select options, preview results, and download final patterns. React is used to build the UI and  communicate with backend API.

##### Key Features:

**Image Upload & Link Input** 
- Form component prompting users to upload images or input an image URL.
- Parameter Selectors represented by dropdowns and sliders for users to adjust settings, including:
  - Type of Image (Dropdown: "Painting", "Illustration", "Icon")
  - Color Count (Slider for "Painting")
  - Thread brand (Dropdown for color selection in "Painting" and "Illustration")
  - Preview: Display the generated pattern in real-time based on user input.
  - Download option for to download the final pattern in SVG, PNG, or PDF format.

**Component Breakdown**
_ImageUploader_: Manages image upload or link input.
_ParameterForm_: Allows users to configure parameters for pattern generation.
_PatternPreview_: Displays the generated pattern.
_DownloadButton_: Enables users to download the pattern in the selected format.

**API Communication**

React will communicate with the backend API using axios or fetch. The backend will return the generated pattern in a usable format (e.g., an SVG or PNG), which will be rendered in the PatternPreview component.
2. Backend API (Flask/FastAPI)
The backend will be responsible for image processing, pattern generation, and returning the results to the front-end. Flask or FastAPI will be used to expose RESTful API endpoints for communication with the front-end.

API Endpoints:

POST /upload: Accepts an image upload or URL, along with parameters such as type of image, color count, and thread brand.
GET /preview: Generates a cross stitch pattern based on the provided parameters and returns a preview image (SVG or PNG).
GET /download: Allows users to download the final cross stitch pattern in a specified format (PDF, SVG, or PNG).
Backend Workflow:

Image Upload/URL Fetch: The backend will either accept an uploaded image or fetch an image from a provided URL.
Image Type Selection:
"Painting": Use KMeans clustering to limit the color palette to the user-specified number of unique colors.
"Icon": Convert the image to a single-color pattern using edge detection to capture distinct shapes and letters.
"Illustration": Use content-aware cropping, background subtraction, and dynamic cropping techniques to isolate the subject.
Pattern Generation: The backend will process the image according to the user’s choices and generate a cross stitch pattern, returning the result as a preview.
Final Pattern Download: Users can request the final pattern in a chosen format (PDF, SVG, or PNG).
Pattern Generation Algorithms:

KMeans Clustering (for "Painting") will reduce the image's colors based on the user's specified number of threads.
Single-Color Pattern (for "Icon") will convert the image to a monochromatic format, using edge detection to outline distinct shapes.
Illustration Handling:
Content-Aware Cropping will remove unnecessary background space using methods like Edge Detection for Bounding Box or Golden Ratio Cropping.
Background Subtraction will detect and remove uniform backgrounds using color similarity and adaptive thresholding techniques.
Aspect Ratio Preservation will ensure the resulting pattern fits a standard frame or aspect ratio, padding the image if necessary.
3. Pattern Generation Logic
The logic for generating patterns will differ based on the image type:

For "Painting":

Apply KMeans clustering to reduce the number of colors.
Generate a pattern with user-specified thread brands for matching colors.
For "Icon":

Convert the image to a single-color pattern using edge detection techniques like the Canny Edge Detector.
For "Illustration":

Content-Aware Cropping:
Use edge detection or saliency detection to isolate the focal point.
Crop out unnecessary background areas.
Background Subtraction:
Identify regions of uniform color (e.g., adaptive thresholding) and remove them from the pattern.
Dynamic Cropping:
Automatically crop the image based on its content using ML techniques like saliency detection to find the most important areas.
Aspect Ratio Preservation:
If the image does not fit the desired aspect ratio, add padding to maintain a clean, centered pattern.

Data Flow:
plantuml
Copy code
@startuml
actor User
User -> React: Upload Image or Provide URL
React -> Flask/FastAPI: POST /upload
Flask/FastAPI -> Image Processor: Process Image (based on type)
Image Processor -> KMeans: Cluster colors (Painting)
Image Processor -> Edge Detection: Detect edges (Icon)
Image Processor -> Saliency Detection: Crop and adjust (Illustration)
Image Processor -> Flask/FastAPI: Return Processed Pattern
Flask/FastAPI -> React: GET /preview (Send Pattern)
React -> User: Display Pattern Preview
User -> React: Download Pattern
React -> Flask/FastAPI: GET /download (format selection)
Flask/FastAPI -> React: Return final downloadable file
@enduml

Architecture Overview:
Front-End: React (Axios for API calls)
Backend: Flask or FastAPI for REST API
Pattern Generation: Python algorithms using OpenCV, Scikit-learn, and possibly TensorFlow or PyTorch (for ML-based techniques)

== Implementation ==

1. Front-End Implementation (React)
Set up React Project:
Initialize a new React project using create-react-app:
bash
Copy code
npx create-react-app cross-stitch-generator
Install necessary dependencies such as axios for API requests:
bash
Copy code
npm install axios
Create Components:
ImageUploader: Handles uploading of images or URL input.
ParameterForm: Allows users to select options such as image type, color count, and thread brand.
PatternPreview: Displays the cross stitch pattern returned from the API.
DownloadButton: Provides the option to download the final pattern in various formats.
API Communication:
Use axios to make POST requests to the backend for image upload and parameters.
Implement GET requests to fetch the generated pattern preview and download links.
Rendering the Pattern:
Use React's useState and useEffect hooks to handle state management for previewing the pattern based on user inputs and API responses.
2. Backend Implementation (Flask/FastAPI)
Set Up Flask or FastAPI:
Install Flask or FastAPI with Python:
bash
Copy code
pip install fastapi uvicorn  # for FastAPI
pip install flask  # for Flask
For FastAPI, also install dependencies for handling image uploads:
bash
Copy code
pip install python-multipart Pillow
Create API Endpoints:
POST /upload: Accepts image data (upload or URL) and pattern parameters (image type, color count, thread brand).
GET /preview: Generates a pattern based on the user-specified parameters and returns a preview (as SVG/PNG).
GET /download: Provides a download link for the final pattern in the requested format (PDF/PNG/SVG).
Pattern Generation Logic:
Implement the pattern generation algorithms for each image type:
Painting: Use KMeans clustering from scikit-learn for color reduction.
Icon: Apply edge detection (e.g., Canny Edge Detection via OpenCV) to create a monochrome pattern.
Illustration: Use content-aware cropping, background subtraction, and dynamic cropping for isolating the main subject.
Integrate Libraries:
KMeans Clustering: Use scikit-learn for clustering colors.
python
Copy code
from sklearn.cluster import KMeans
Edge Detection: Use OpenCV for detecting edges.
bash
Copy code
pip install opencv-python
Saliency Detection: For machine learning-based cropping, you could implement pre-trained models for saliency detection.
3. Pattern Generation Logic (Python)
KMeans for "Painting" Type:
Convert the image to an array of RGB values and apply KMeans clustering to limit the number of colors based on user input.
python
Copy code
kmeans = KMeans(n_clusters=num_colors).fit(image_array)
Use the resulting centroids to map colors to thread brand equivalents.
Edge Detection for "Icon" Type:
Convert the image to grayscale and apply Canny Edge Detection to generate a single-color outline of the image.
python
Copy code
edges = cv2.Canny(image, 100, 200)
Content-Aware Cropping for "Illustration" Type:
Use Edge Detection for Bounding Box to crop the image around the main subject.
Apply background subtraction to remove uniform background regions.
python
Copy code
# Thresholding for background subtraction
_, thresh = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)
Aspect Ratio Preservation:
Ensure that the image aspect ratio is preserved by adding padding if necessary. This ensures that the generated pattern fits typical cross-stitch frame dimensions.
4. Testing and Debugging
Front-End Testing:
Use React Developer Tools to test component state and API response handling.
Test image uploads, parameter adjustments, and downloading patterns to ensure the UI behaves as expected.
Backend Testing:
Use Postman or curl to test API endpoints.
Validate that each image type (painting, icon, illustration) is processed correctly and that appropriate pattern previews and downloads are returned.
Integration Testing:
Ensure that the React front-end and Flask/FastAPI backend work seamlessly together.
Test multiple image types with various parameters to verify that the pattern generation logic is functioning as intended.
5. Deployment
Back-End Deployment:
Deploy the Flask/FastAPI application to a cloud provider such as Heroku, AWS, or DigitalOcean.
Use Docker if necessary to package the backend into a container for deployment.
Front-End Deployment:
Deploy the React app using Netlify, Vercel, or a similar platform.

== Milestones ==
Front-End Setup and Image Upload Feature:
Basic React app with image upload functionality.
API endpoint to handle image data.
Pattern Generation for 'Painting' and 'Icon' Types:
Backend implementation for KMeans clustering and edge detection.
Preview patterns in the front-end.
'Illustration' Type Pattern Generation:
Implement content-aware cropping, background removal, and aspect ratio preservation.
API integration with the front-end for illustration-type images.
Download Feature and Final Pattern Output (2 weeks):
Allow users to download the final cross stitch pattern in various formats (PDF/SVG/PNG).
User Authentication and Fine-Tuning of Patterns (4 weeks):
Add user accounts for saving and managing patterns.
Implement real-time updates for pattern adjustments.
== Gathering Results == Upon completion, results can be gathered by:

User feedback on the ease of use of the UI and accuracy of generated patterns.
Monitoring API performance to ensure image processing is efficient.
Testing the patterns for usability in actual cross stitch projects, ensuring the designs translate well into physical form.

