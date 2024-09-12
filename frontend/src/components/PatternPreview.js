import React from 'react';

function PatternPreview({ pattern }) {
    return (
        <div>
            {pattern ? <img src={pattern} alt="Pattern Preview" /> : <p>No pattern generated yet.</p>}
        </div>
    );
}

export default PatternPreview;

