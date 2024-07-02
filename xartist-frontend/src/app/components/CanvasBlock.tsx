import React, {useEffect, useState} from "react";

import {fetchImages} from "@/app/api/route";

const images = [
        '/wikiart_animation.gif',
        '/wikiart_artist_animation.gif',
    ]; // Replace paths with your image paths or URLs

function CanvasBlock() {

    const tmp_images = [
        '/wikiart_animation.gif',
        '/wikiart_artist_animation.gif',
    ]; // Replace paths with your image paths or URLs

    const [index, setIndex] = useState(0);
    const [images, setImages] = useState([]);

    // fetching images
    useEffect(() => {
        const inter_fetch = async () => {
            try {
                const img_b64 = await fetchImages();
                console.log('Fetched images:', img_b64);
                if (Array.isArray(img_b64) && img_b64.length > 0) {
                    setImages(prevImages => {
                        const newImages = [...prevImages, ...img_b64];
                        console.log('Updated images state:', newImages.length);
                        return newImages;
                    });
                } else {
                    console.error('Failed to fetch images: response is not an array');
                }
            } catch (error) {
                console.error('Failed to fetch images:', error);
            }
        };

        const intervalId = setInterval(inter_fetch, 1000);

        return () => clearInterval(intervalId);
    }, [])

    // Set looping index
    useEffect(() => {
        if (images.length > 0) {
            console.log('Loaded index:', index);
            const interval = setInterval(() => {
                setIndex((prevIndex) => (prevIndex + 1) % images.length);
            }, 90); // Change image every 3000 milliseconds (3 seconds)

            return () => clearInterval(interval);
        }
    }, [images.length]);

    const cacheImages = (newImages) => {
        const cachedImages = JSON.parse(localStorage.getItem('cachedImages') || '[]');
        const updatedCache = [...cachedImages, ...newImages];
        localStorage.setItem('cachedImages', JSON.stringify(updatedCache));
    }

    const nextImage = () => {
        setIndex((index + 1) % images.length);
    };

    const prevImage = () => {
        setIndex((index - 1 + images.length) % images.length);
    };

    return (
        <div className="">
            {images.length > 0 && (
                <img
                    src={`data:image/png;base64,${images[index]}`}
                    alt="Slideshow"
                    style={{ width: '256px', height: '256px' }}
                    className="object-cover object-center w-full h-256 max-w-full rounded-lg"
                />
            )}
        </div>
    )
}


export default CanvasBlock;