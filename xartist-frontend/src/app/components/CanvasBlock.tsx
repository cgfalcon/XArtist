import React, {useEffect, useState, useRef} from "react";

import {fetchImages} from "@/app/api/route";
import { Button } from "@material-tailwind/react";


function CanvasBlock() {

    const [index, setIndex] = useState(0);
    const [images, setImages] = useState([]);
    const [isPlaying, setIsPlaying] = useState(true);
    const intervalRef = useRef(null);

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
        if (isPlaying) {
            if (images.length > 0) {
                console.log('Loaded index:', index);
                intervalRef.current = setInterval(() => {
                    setIndex((prevIndex) => (prevIndex + 1) % images.length);
                }, 90); // Change image every 3000 milliseconds (3 seconds)


            }
        }
        return () => clearInterval(intervalRef.current);

    }, [isPlaying, images.length]);

    const togglePlayPause = () => {
        setIsPlaying(!isPlaying);
    };

    const handleDownload = () => {
        const link = document.createElement('a');
        link.href = `data:image/jpeg;base64,${images[index]}`;
        link.download = `downloadedImage-${index}.jpeg`; // Naming the downloaded file
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

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

    // Sample metadata for the artwork
    const artworkDetails = {
        artist: "Anonymous",
        dateCreated: "2024",
        materials: "GAN"
    };

    return (
        <div className="flex justify-center items-center h-screen">
            <div className="relative border-black p-10 bg-white shadow-sm group">
                {images.length > 0 && (
                    <>
                        <img
                            src={`data:image/png;base64,${images[index]}`}
                            alt="Art Animation"
                            style={{width: '500px', height: '500px'}}
                            className="object-cover object-center w-full h-256 max-w-full"
                            // onMouseEnter={() => setIsPlaying(false)}
                            // onMouseLeave={() => setIsPlaying(true)}
                        />
                        <div
                            className="absolute bottom-0 left-0 right-0 bg-white bg-opacity-20 p-3 text-sm flex justify-center space-x-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <Button
                                className="rounded-full flex items-center gap-3"
                                style={{backgroundColor: '#4f45e4', color: 'white'}}
                                buttonType="filled"
                                block={false}
                                iconOnly={false}
                                onClick={togglePlayPause}
                            >
                                {isPlaying ?
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                                         stroke-width="1.5"
                                         stroke="currentColor" className="size-4">
                                        <path strokeLinecap="round" strokeLinejoin="round"
                                              d="M15.75 5.25v13.5m-7.5-13.5v13.5"/>
                                    </svg>
                                    : <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                                           stroke-width="1.5"
                                           stroke="currentColor" className="size-4">
                                        <path strokeLinecap="round" strokeLinejoin="round"
                                              d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z"/>
                                    </svg>}

                                {isPlaying ? 'Pause' : 'Play'}
                            </Button>
                            {!isPlaying && (
                                <Button
                                    className="rounded-full flex items-center gap-3"
                                    buttonType="filled"
                                    block={false}
                                    iconOnly={false}
                                    onClick={handleDownload}
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                                         stroke-width="1.5"
                                         stroke="currentColor" className="size-4">
                                        <path strokeLinecap="round" strokeLinejoin="round"
                                              d="M19.5 13.5 12 21m0 0-7.5-7.5M12 21V3"/>
                                    </svg>
                                    Download
                                </Button>
                            )}
                        </div>
                    </>
                )}

            </div>
            <div className="p-5 bg-white border-l-2 border-gray-500 ">
                <p><strong>Artist:</strong> {artworkDetails.artist}</p>
                <p><strong>Date:</strong> {artworkDetails.dateCreated}</p>
                <p><strong>Material:</strong> {artworkDetails.materials}</p>
            </div>
        </div>
    )
}

export default CanvasBlock;