from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from .image_processing import process_image
from .pattern_generation import generate_pattern
import os


app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400

        image = request.files['image']

        # Corrected: Use the allowed_file function to check the file extension
        if not allowed_file(image.filename):
            return jsonify({'error': 'Unsupported file type'}), 400

        params = request.form
        image_type = params.get('image_type', 'painting')
        color_count = int(params.get('color_count', 10))
        thread_brand = params.get('thread_brand', 'DMC')

        # Debug log
        print(f"Image received: {image.filename}, Image type: {image_type}, Color count: {color_count}, Thread brand: {thread_brand}")

        processed_image = process_image(image, image_type)
        pattern = generate_pattern(processed_image, image_type, color_count, thread_brand)

        # Save and send pattern file to front-end
        save_dir = os.path.join(app.root_path, 'generated_patterns')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pattern_file = os.path.join(save_dir, "output_pattern.png")
        pattern.save(pattern_file)
        
        if os.path.exists(pattern_file):
            print(f"Pattern file saved at: {pattern_file}")
            return send_file(pattern_file, mimetype='image/png', as_attachment=False)
        else:
            return jsonify({'error': 'Pattern file not created'}), 500

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/preview', methods=['GET'])
def preview_pattern():
    # Preview logic
    pass


@app.route('/download', methods=['GET'])
def download_pattern():
    # Download logic
    pass


if __name__ == '__main__':
    app.run(debug=True)

