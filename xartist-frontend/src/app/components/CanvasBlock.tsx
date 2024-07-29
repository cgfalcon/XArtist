import React, {useEffect, useState, useRef} from "react";

import {fetchImages} from "@/app/api/route";
import { Button } from "@material-tailwind/react";
import { useError } from "../context/ErrorContext";
import {Simulate} from "react-dom/test-utils";


function CanvasBlock({ id, model, playing, placeholder }) {

    const [index, setIndex] = useState(0);
    const [images, setImages] = useState([]);
    const [isPlaying, setIsPlaying] = useState(false);
    const [loading, setLoading] = useState(false);
    const [hasBeenPlayed, setHasBeenPlayed] = useState(playing);
    const intervalRef = useRef(null);
    const { showError } = useError();

    const [fps, setFps] = useState(0);
    const frameCount = useRef(0);
    const lastUpdate = useRef(performance.now());
    const [fpsSeq, setFpsSeq] = useState([]);


    useEffect(() => {
        setIsPlaying(playing);

    }, [playing]);

    // fetching initializing images
    useEffect(() => {
        const fetchInitialImages = async () => {
            setLoading(true);
            try {
                const img_resp = await fetchImages(model.model_key);
                console.log('CanvasBlock[', id, '] Fetched initializing images:', img_resp);
                if (img_resp.success && Array.isArray(img_resp.data) && img_resp.data.length > 0) {
                    setImages(img_resp.data);
                } else {
                    console.error('Failed to fetch images: response is not an array');
                    showError(img_resp.error_msg);
                }
                setLoading(false); // Set loading to false after fetching images
            } catch (error) {
                console.error('Failed to fetch images:', error);
                showError(error.response?.data?.error_msg || 'Failed to fetch images');
                setLoading(false); // Set loading to false even if there's an error
            }
        };

        if (isPlaying) {
            fetchInitialImages();
            setHasBeenPlayed(true);
        }


    }, [model, showError]);

    // fetching images continously
    useEffect(() => {
        const inter_fetch = async () => {
            try {
                // if (!model) {
                //     model = 'impressionist_150'
                // }
                if (isPlaying) {
                    const img_resp = await fetchImages(model.model_key);
                    console.log('CanvasBlock[', id, '] Fetched images for model:', model.model_key);
                   if (img_resp.success && Array.isArray(img_resp.data) && img_resp.data.length > 0) {
                        setImages(prevImages => {
                            const newImages = [...prevImages, ...img_resp.data];
                            console.log('CanvasBlock[', id, '] Updated images state:', newImages.length);
                            return newImages;
                        });
                        setLoading(false);
                    } else {
                        console.error('CancasBlock[', id, '] Failed to fetch images: response is not an array');
                        showError('Failed to fetch images');
                        setLoading(false)
                    }
                }

            } catch (error) {
                console.error('Failed to fetch images:', error);
                showError(error.response?.data?.error_msg || 'Failed to fetch images');
            }
        };

        const intervalId = setInterval(inter_fetch, 2000);

        return () => clearInterval(intervalId);
    }, [model, isPlaying, showError])

    // Set looping index
    useEffect(() => {
        if (isPlaying) {
            if (images.length > 0) {
                intervalRef.current = setInterval(() => {
                    setIndex((prevIndex) => {
                        const newIndex = (prevIndex + 1) % images.length;

                        // FPS calculation
                        frameCount.current++;
                        const now = performance.now();
                        const elapsed = now - lastUpdate.current;

                        if (elapsed >= 1000) {
                            const calculatedFps = (frameCount.current / elapsed) * 1000;
                            const fpsValue = parseFloat(calculatedFps.toFixed(2));
                            setFps(fpsValue);
                            setFpsSeq((curr) => [...curr, fpsValue]);
                            lastUpdate.current = now;
                            frameCount.current = 0;
                            console.log('CanvasBlock[', id, '-', model.model_name, '] FPS:', fpsSeq)
                        }

                        return newIndex;
                    });
                    console.log('CancasBlock[', id, '] Loaded index:', index);
                }, 50); // Change image every 3000 milliseconds (3 seconds)
            } else {
                setLoading(true);
            }
        }
        return () => clearInterval(intervalRef.current);

    }, [isPlaying, images.length]);

    const togglePlayPause = () => {
        setIsPlaying(!isPlaying);
        setHasBeenPlayed(true);
    };

    const handleDownload = () => {
        const link = document.createElement('a');
        link.href = `data:image/jpeg;base64,${images[index]}`;
        link.download = `downloadedImage-${index}.jpeg`; // Naming the downloaded file
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // Sample metadata for the artwork
    const artworkDetails = {
        artist: "Anonymous",
        dateCreated: "2024",
        materials: "GAN"
    };

    return (
        <div className="flex justify-center items-center">
            <div
                className={`relative border-black p-0 shadow-sm group transition ease-in-out duration-500 `}
                style={{width: '350px', height: '350px'}}
            >
                {loading ? (
                    <div className="flex justify-center items-center w-full">
                        <svg className="animate-spin h-8 w-8 text-gray-600" xmlns="http://www.w3.org/2000/svg"
                             fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                    strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                        </svg>
                    </div>
                ) : (
                    <>
                        <img
                            src={hasBeenPlayed ? `data:image/png;base64,${images[index]}` : placeholder}
                            alt="Art Animation"
                            // style={{width: '350px', height: '350px'}}
                            className="object-cover object-center w-full h-full"
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
                <p className={`text-center ${isPlaying ? 'text-white' : 'text-gray-400'} font-bold mt-2`}>{model.model_name}</p>
                <p className="text-center text-gray-400 mt-1 text-left">{model.model_desc}</p>
            </div>
        </div>
    )
}

export default CanvasBlock;