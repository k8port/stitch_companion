import React, { useState } from 'react';
import axios from 'axios';

function ImageUploader({ onUploadSuccess }) {
    const [selectedImage, setSelectedImage] = useState(null);

    // Handles the image upload selection from the input field
    const handleImageUpload = (e) => {
        setSelectedImage(e.target.files[0]);  // Store the selected image in state
    };

    // Submits the image to the backend server
    const submitImage = () => {
        // Check if an image is selected before making the request
        if (!selectedImage) {
            console.error("No image selected for upload");
            alert("Please select an image to upload.");
            return;  // Exit early if no image is selected
        }

        const formData = new FormData();
        formData.append('image', selectedImage);  // Append the selected image to the form data
        formData.append('image_type', 'painting');  // Add additional parameters (example: image_type)
        formData.append('color_count', 10);  // Add color count
        formData.append('thread_brand', 'DMC');  // Add thread brand

        // Make an Axios POST request to the server
        axios.post('http://localhost:5001/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            responseType: 'blob'  // Expect the server response to be a Blob
        })
        .then((response) => {
            console.log("Server response:", response);  // Log the entire response from the server
            console.log("Response data type is Blob:", response.data instanceof Blob);  // Check if the response data is a Blob

            // Handle the Blob response and create a URL for the image preview
            try {
                if (response.data instanceof Blob) {
                    const url = URL.createObjectURL(response.data);  // Create a URL for the Blob
                    onUploadSuccess(url);  // Pass the blob URL to the parent component for display
                } else {
                    console.error("Response data is not a Blob");
                }
            } catch (err) {
                console.error("Failed to create blob URL", err);
            }
        })
        .catch((error) => {
            console.error('Error uploading image:', error);
            alert("Error uploading image. Please try again.");
        });
    };

    return (
        <div>
            {/* File input for image selection */}
            <input type="file" onChange={handleImageUpload} />
            {/* Button to trigger image upload */}
            <button onClick={submitImage}>Upload</button>
        </div>
    );
}

export default ImageUploader;
