import React, { useState, useEffect } from 'react';
import CanvasBlock from './CanvasBlock';

const ImageScroller = () => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [displayedModels, setDisplayedModels] = useState([]);

    const models = [
        { model_key: 'gan256_bce_impressionism_600', model_name: 'Impressionism', model_desc: "Collection of impressionism" },
        { model_key: 'gan256_hinge_impressionism_600', model_name: 'Abstract Still-life', model_desc: "A mixture of impressionism & still life" },
        { model_key: 'abstract_strip', model_name: 'Abstract Blocks', model_desc: "Strips patterns learned by GAN" },
        { model_key: 'gan256_hinge_artist_2400', model_name: 'Artists Mix', model_desc: "Mix styles of wellknown artists" },
        // { model_key: 'gan256_hinge_portrait_420', model_name: 'Portrait', model_desc: "Portrait of WikiArt" },
        { model_key: 'gan256_cubism_landscape_030', model_name: 'Cubism Landscape', model_desc: "Landscape applied by Cubism" },
        { model_key: 'gan256_hinge_landscape_660', model_name: 'Landscape', model_desc: "Landscape of WikiArt" },
    ];

    const img_placeholders = [
        "./img_placeholder_1.jpeg",
        "./img_placeholder_2.jpeg",
        "./img_placeholder_3.jpeg",
        "./img_placeholder_4.jpeg",
        "./img_placeholder_6.jpeg",
        "./img_placeholder_7.jpeg",
        "./img_placeholder_8.jpeg",
        "./img_placeholder_9.jpeg",
    ]

    // useEffect(() => {
    //     console.log("CurrentIndex", currentIndex);
    //     setDisplayedModels(getDisplayModels(currentIndex));
    // }, [currentIndex]);


    const handleArrowClick = (direction) => {
        var slider = document.getElementById("slider")
        if (direction === 'left') {
            slider.scrollLeft = slider.scrollLeft - 100
            setCurrentIndex((prevIndex) => (prevIndex > 0 ? prevIndex - 1 : models.length - 1));
        } else {
            slider.scrollLeft = slider.scrollLeft + 100
            setCurrentIndex((prevIndex) => (prevIndex + 1) % models.length);
        }
    };

    // const getDisplayModels = () => {
    //     let start = currentIndex - 1 ;
    //     let end = currentIndex + 1;
    //     if (start < 0) {
    //         start += models.length;
    //     }
    //     if (end > models.length) {
    //         end -= models.length;
    //     }
    //     const displayModels = [];
    //     for (let i = start; i !== end; i = (i + 1) % models.length) {
    //         displayModels.push(models[i]);
    //     }
    //     return models;
    // };


    return (
        <div className="relative flex items-center w-full h-screen">
            <button
                onClick={() => handleArrowClick('left')}
                className="text-4xl absolute left-4 z-20 text-white bg-black bg-opacity-50 rounded-full p-4"
            >
                &lt;
            </button>
            <div id="slider"
                 className="flex items-center justify-start overflow-x-scroll whitespace-nowrap scroll-smooth w-full h-full transition-transform ease-in-out duration-500"
                 style={{display: 'flex', alignItems: 'center'}}
            >
                {models.map((model, index) => (
                    <div key={model.model_key}
                         className={`inline-block transform scale-95 hover:scale-100 ease-in-out duration-300`}
                    >
                        <CanvasBlock id={index} model={model}
                                     playing={false}
                                     placeholder={img_placeholders[index]}/>
                    </div>
                ))}
            </div>
            <button
                onClick={() => handleArrowClick('right')}
                className="text-4xl absolute right-4 z-20 text-white bg-black bg-opacity-50 rounded-full p-4"
            >
                &gt;
            </button>
        </div>
    );
};

export default ImageScroller;