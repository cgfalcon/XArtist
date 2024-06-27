import React, {useEffect, useState} from "react";

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
                const rest = await fetch('http://localhost:5002/api/images');
                const data = await rest.json()
                setImages(prevImages => [...prevImages, ...data.images]);
                cacheImages(data.images)
            } catch (error) {
                console.error('Failed to fetch images:', error);
            }
        }

        const intervalId = setInterval(inter_fetch, 1000)

        return () => clearInterval(intervalId)
    }, [])

    // Set looping index
     useEffect(() => {
        const interval = setInterval(() => {
            setIndex((prevIndex) => (prevIndex + 1) % images.length);
        }, 8000); // Change image every 3000 milliseconds (3 seconds)

        return () => clearInterval(interval);
    }, []);

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
            <img src={tmp_images[index]} alt="Slideshow" style={{width: '256', height: '256'}}
                 className="object-cover object-center w-full h-256 max-w-full rounded-lg"/>
        </div>
    )
}


export default CanvasBlock;