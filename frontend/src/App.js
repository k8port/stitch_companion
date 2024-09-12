import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';

function App() {
    const [imageSrc, setImageSrc] = useState(null);

    const handleUploadSuccess = (url) => {
        setImageSrc(url);  // Set the image source to the Blob URL
    };

    return (
        <div>
            <h1>Image Upload and Preview</h1>
            <ImageUploader onUploadSuccess={handleUploadSuccess} />
            {imageSrc && <img src={imageSrc} alt="Uploaded pattern" />}  {/* Display the uploaded image */}
        </div>
    );
}

export default App;
