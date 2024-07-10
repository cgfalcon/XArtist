import React, { useState, useEffect } from 'react';
import CanvasBlock from './CanvasBlock';

const ImageScroller = () => {
    const [currentIndex, setCurrentIndex] = useState(1);
    const [displayedModels, setDisplayedModels] = useState([]);

    const models = [
        { model_key: 'impressionist_150', model_name: 'Impressionism' },
        { model_key: 'still_lift_300', model_name: 'Abstract Still-life' },
        { model_key: 'abstract_strip', model_name: 'Abstract Blocks' }
    ];

    useEffect(() => {
        setDisplayedModels(getDisplayModels(currentIndex));
    }, [currentIndex]);

    const handleArrowClick = (direction) => {
        if (direction === 'left') {
            setCurrentIndex((prevIndex) => (prevIndex > 0 ? prevIndex - 1 : models.length - 1));
        } else {
            setCurrentIndex((prevIndex) => (prevIndex + 1) % models.length);
        }
    };

    const getDisplayModels = () => {
        let start = currentIndex - 1;
        let end = currentIndex + 1;
        if (start < 0) {
            start += models.length;
        }
        if (end > models.length) {
            end -= models.length;
        }
        const displayModels = [];
        for (let i = start; i !== end; i = (i + 1) % models.length) {
            displayModels.push(models[i]);
        }
        return models;
    };

    return (
        <div className="relative flex items-center justify-between w-full overflow-hidden">
            <button onClick={() => handleArrowClick('left')} className="text-2xl absolute left-0 z-10 text-white">&lt;</button>
            <div className="flex space-x-4 w-full object-cover">
                {displayedModels.map((model, index) => (
                    <div key={model.model_key} className="flex flex-col items-center w-512 h-512">
                        <CanvasBlock model={model.model_key} playing={index === 1}/>
                        <p>{model.model_name}</p>
                    </div>
                ))}
            </div>
            <button onClick={() => handleArrowClick('right')} className="text-2xl absolute right-0 z-10 text-white">&gt;</button>
        </div>
    );
};

export default ImageScroller;