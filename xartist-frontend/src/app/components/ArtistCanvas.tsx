import React, { useState, useEffect } from 'react';

function ArtistCanvas() {

    const images = [
        '/wikiart_animation.gif',
        '/wikiart_artist_animation.gif',
    ]; // Replace paths with your image paths or URLs

    const [index, setIndex] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setIndex((prevIndex) => (prevIndex + 1) % images.length);
        }, 8000); // Change image every 3000 milliseconds (3 seconds)

        return () => clearInterval(interval);
    }, []);

    const nextImage = () => {
        setIndex((index + 1) % images.length);
    };

    const prevImage = () => {
        setIndex((index - 1 + images.length) % images.length);
    };

    return (
        <div className="relative isolate px-6 pt-14 lg:px-8">
            <div>
                <img src={images[index]} alt="Slideshow" style={{width: '128px', height: '128px'}}
                     className="h-8 w-auto"/>
            </div>
            <div>
                <img src={images[index]} alt="Slideshow" style={{width: '128px', height: '128px'}}
                     className="h-8 w-auto"/>
            </div>
        </div>

    );
}

export default ArtistCanvas;