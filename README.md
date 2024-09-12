# Stitch Companion 2 - Cross Stitch Pattern Generator

Stitch Companion 2 is a web application that generates cross-stitch patterns from uploaded images. It features a React-based frontend and a Flask-based backend API written in Python. The backend processes images and generates patterns using algorithms such as KMeans clustering, image quantization, and color mapping to DMC threads.

## Features
- Upload images to generate cross-stitch patterns.
- Specify thread brand (e.g., DMC) and color count.
- Supports pattern generation for different image types (e.g., paintings, illustrations, and icons).
- Download patterns in formats such as PNG, SVG, or PDF.

## Project Structure

```bash
.
├── backend                   # Flask backend with pattern generation logic
│   ├── app/
│   │   ├── image_processing.py
│   │   ├── pattern_generation.py
│   │   ├── routes.py
│   ├── Dockerfile             # Docker configuration for backend
│   ├── requirements.txt       # Python dependencies
│   └── generated_patterns/    # Directory where generated patterns are stored
├── frontend                  # React frontend
│   ├── src/
│   ├── Dockerfile             # Docker configuration for frontend
│   └── package.json           # NPM dependencies
├── docker-compose.yml         # Docker Compose setup
└── README.md                  # This file

## Prerequisites

Have the following installed:

- Python 3.12+ (for running without Docker)
- node.js 18+ (for React frontend)
- docker & docker-compose (for running with Docker)

## Running Project in Docker

Run both the frontend and backend using docker and docker-compose:

1. Clone the repository
```bash
git clone https://github.com/yourusername/stitch_companion_2.git
cd stitch_companion_2
```

2. Build and start the Docker containers
Run the following commands to build and start the services:
```bash
docker-compose build --no-cache
docker-compose up
```

The frontend will be accessible at http://localhost:3000.
The backend API will be accessible at http://localhost:5001.

3. Stop the services
To stop the services, run:
```bash
docker-compose down
```

## Running the Project (without Docker)

Run the frontend and backend on your local machine:

### Backend (Flask API)
1. Navigate to the backend directory
```bash
cd backend
```

2. Create a virtual environment (optional, but recommended):
```bash
python3 -m venv env
source env/bin/activate  # For Windows, use `env\Scripts\activate`
```

3. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask API:
```bash
export FLASK_APP=app/routes.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5001
```

The Flask API will now be running on http://localhost:5001.


### Frontend (React)
1. Navigate to the frontend directory
```bash
cd frontend
```

2. Install the Node.js dependencies
```bash
npm install
```

3. Start the React development server
```bash
npm start
```

The frontend will now be running on http://localhost:3000.

### API Endpoints

POST `/upload`
Uploads an image for processing. This endpoint requires a `POST` request with `multipart/form-data` containing the following fields:

- `image`: The uploaded image file.
- `image_type`: Type of the image (e.g., painting, icon, illustration).
- `color_count`: Number of colors to use in the pattern.
- `thread_brand`: The brand of thread for color mapping (e.g., DMC).

#### Response
The response is a generated pattern image file.

## Deployment

### Docker
To deploy using Docker, ensure the `docker-compose.yml` file is configured for production, and then run:
```bash
docker-compose up --build -d
```

### Manual Deployment
1. Set up the Flask API on a server with Python installed.
2. Serve the React frontend using a tool like Nginx, Apache, or Netlify.
3. Ensure the frontend and backend are properly connected (adjust API URLs if needed).

## Contributing

1. Fork the repository.
2. Create a new branch for your feature (git checkout -b feature-name).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-name).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or inquiries, please reach out to k8portalatin@gmail.com.


### Key Sections Explained:

- **Project Overview**: Briefly describes the functionality of the project.
- **Prerequisites**: Lists tools and dependencies needed to run the project locally or in Docker.
- **Running the Project (Docker)**: Instructions to build and run the project in Docker.
- **Running the Project (Without Docker)**: Step-by-step guide for running both the backend and frontend independently.
- **API Endpoints**: Details about the `/upload` endpoint and how to interact with the backend API.
- **Testing**: Instructions for running tests for both frontend and backend.
- **Deployment**: Quick notes on deployment using Docker or manual setup.
- **Contributing**: Guidelines for contributing to the project.
- **License**: Specifies the project’s license.

You can customize the repository link, contact information, and other details to fit your project’s specifics. Let me know if you'd like to add or change anything!



