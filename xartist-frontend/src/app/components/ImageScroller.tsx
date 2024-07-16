import React, { useState, useEffect } from 'react';
import CanvasBlock from './CanvasBlock';

const ImageScroller = () => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [displayedModels, setDisplayedModels] = useState([]);

    const models = [
        { model_key: 'gan256_bce_impressionism_600', model_name: 'Impressionism' },
        { model_key: 'gan256_hinge_impressionism_600', model_name: 'Abstract Still-life' },
        { model_key: 'abstract_strip', model_name: 'Abstract Blocks' },
        { model_key: 'gan256_hinge_landscape_660', model_name: 'Landscape' }
    ];

    const img_placeholders = [
        "./img_placeholder_1.jpeg",
        "./img_placeholder_2.jpeg",
        "./img_placeholder_3.jpeg",
        "./img_placeholder_4.jpeg"
    ]

    useEffect(() => {
        console.log("CurrentIndex", currentIndex);
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
        let start = currentIndex - 1 ;
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
        <>
         <div className="relative flex items-center w-full">
            <button onClick={() => handleArrowClick('left')} className="text-2xl absolute left-0 z-12 text-white">&lt;</button>
            <div className="relative flex space-x-1 w-full object-cover">
                {displayedModels.map((model, index) => (
                    <div key={model.model_key} className="w-[256] inline-block hover:scale-105 ease-in-out duration-200 hover:z-10">
                        <CanvasBlock model={model.model_key} playing={index === currentIndex} placeholder={img_placeholders[index]}/>
                        <p>{model.model_name}</p>
                    </div>
                ))}
            </div>
            <button onClick={() => handleArrowClick('right')} className="text-2xl absolute right-0 z-12 text-white">&gt;</button>
         </div>
        </>
    );
};

export default ImageScroller;