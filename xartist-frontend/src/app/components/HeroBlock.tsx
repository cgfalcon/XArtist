import React, {useEffect, useState} from "react";

import {fetchImages} from "@/app/api/route";

function HeroBlock() {

    const tmp_images = [
        // '/wikiart_resnet_60.gif',
        '/wikiart_resnet_150.gif',
        '/wikiart_sndcgan_stilllife_300.gif',
        '/wikiart_sngan_genra_150.gif',
        '/wikiart_frelu_impressionism_050.gif',
        '/wikiart_frelu_impressionism_100.gif',
        '/wikiart_impressionist_animation100.gif',
        '/wikiart_impressionist_animation50.gif',
        '/wikiart_abstract_animation160.gif',
        '/wikiart_artist_animation.gif',
    ]; // Replace paths with your image paths or URLs

    const [index, setIndex] = useState(0);
    const [images, setImages] = useState([]);

    // Set looping index
    useEffect(() => {
        if (tmp_images.length > 0) {
            console.log('Loaded index:', index);
            const interval = setInterval(() => {
                setIndex((prevIndex) => (prevIndex + 1) % tmp_images.length);
            }, 10000); // Change image every 3000 milliseconds (3 seconds)

            return () => clearInterval(interval);
        }
    }, [tmp_images.length]);

    const cacheImages = (newImages) => {
        const cachedImages = JSON.parse(localStorage.getItem('cachedImages') || '[]');
        const updatedCache = [...cachedImages, ...newImages];
        localStorage.setItem('cachedImages', JSON.stringify(updatedCache));
    }

    return (
        <div className="">
            {tmp_images.length > 0 && (
                <img
                    src={tmp_images[index]}
                    style={{width: '100%', height: '100vh'}}
                    className="object-cover object-center w-full h-full max-w-full"
                />
            )}
        </div>
    )
}


export default HeroBlock;