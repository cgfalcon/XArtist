import React, { useState, useEffect } from 'react';

import CanvasBlock from "@/app/components/CanvasBlock";
import ImageScroller from "@/app/components/ImageScroller";

function ArtistCanvas() {
const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [canvasLayout, setCanvasLayout] = useState("Single"); // "Grid" or "Single"

    // Fetch models from the API
    useEffect(() => {
        async function fetchModels() {
            try {
                const response = await fetch('http://127.0.0.1:5000/api/dynamic_block/get_models'); // Adjust the API endpoint as necessary
                if (!response.ok) throw new Error('Failed to fetch');
                const mdlist = await response.json();
                if (mdlist.data && mdlist.data.length > 0) {
                    setModels(mdlist.data);
                    setSelectedModel(mdlist.data[0].model_key); // Default to the first model
                } else {
                    console.error('No models found');
                }
            } catch (error) {
                console.error('Error fetching models:', error);
            }
        }

        fetchModels();
        // const ret = await fetchModels();
        // setModels(ret);
        // setSelectedModel(ret[0].model_key); // Default to the first model
    }, []);

    // Handle model selection change
    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
    };

    const renderCanvas = () => {
        if (canvasLayout === "Single") {
            return (
                <ImageScroller/>
            );
        } else {
            return (
                <div className="grid grid-cols-4 gap-2 center">
                    {Array(8).fill(0).map((_, idx) => (
                        <CanvasBlock key={idx} model={selectedModel}/>
                    ))}
                </div>
            );
        }
    };

    return (
        <div>
            {/* Dropdown for selecting model */}

            {renderCanvas()}
        </div>
    );


}

export default ArtistCanvas;