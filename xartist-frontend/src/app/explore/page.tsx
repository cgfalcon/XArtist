"use client"

import Navbar from "../components/NavBar";
import React from "react";
import { Slider, Select, Option } from "@material-tailwind/react";


import { Component, useEffect, useState, useRef } from "react";
import ImageCanvas from "../components/ImageCanvas";  // for Latent Space Explorer
import XYPlot from "../components/XYPlot";  
import * as tf from "@tensorflow/tfjs";  // for Latent Space Explorer
import gaussian from "gaussian";  // for Latent Space Explorer
import encodedData from "../encoded.json";  // for Latent Space Explorer

import "../page.module.css";
import {imag} from "@tensorflow/tfjs";  // for Latent Space Explorer

const MODEL_PATH = "models/generatorjs/model.json";

interface Props {
}

const Explore = () => {
  const models = [
        { model_key: 'gan256_bce_impressionism_600', model_name: 'Impressionism', model_desc: "Collection of impressionism" },
        { model_key: 'gan256_hinge_impressionism_600', model_name: 'Abstract Still-life', model_desc: "A mixture of impressionism & still life" },
        { model_key: 'abstract_strip', model_name: 'Abstract Blocks', model_desc: "Strips patterns learned by GAN" },
        { model_key: 'gan256_hinge_artist_2400', model_name: 'Artists Mix', model_desc: "Mix styles of wellknown artists" },
        { model_key: 'gan256_hinge_portrait_420', model_name: 'Portrait', model_desc: "Portrait of WikiArt" },
        { model_key: 'gan256_cubism_landscape_030', model_name: 'Cubism Landscape', model_desc: "Landscape applied by Cubism" },
        { model_key: 'gan256_hinge_landscape_660', model_name: 'Landscape', model_desc: "Landscape of WikiArt" },
    ];


  const [selectedModel, setSelectedModel] = useState(models[0].model_key);
  const [dots, setDots] = useState([]);
  const [image, setImage] = useState(null);
  const [imagesBuffer, setImagesBuffer] = useState([]);
  const [index, setIndex] = useState(0);
  const [isHovering, setIsHovering] = useState(false);
  const [loading, setLoading] = useState(true);
  const [imgLoading, setImgLoading] = useState(true);
  const intervalRef = useRef(null);


  // *** Storage array and timer reference for scheduled processing ***
  const coordinatesRef = useRef([]);
  const fetchTimerRef = useRef(null);
  const displayTimerRef = useRef(null);

    // Handle model selection change
  const handleSelectChange = (value) => {
    setSelectedModel(value);
  };
  
  // // *** Buffer state to handle debouncing ***
  // const hoverTimeoutRef = useRef(null);

  useEffect(() => {
    // Generate 3,000 random dots
    const generatedDots = [];
    for (let i = 0; i < 1000; i++) {
      generatedDots.push({
        x: (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5 ) * 2,
      });
    }
    setDots(generatedDots);
    convertDotToImg(0.1, 0.1);
    setLoading(false);
  }, []);

  useEffect(() => {
    // Generate 3,000 random dots
    convertDotToImg(0.1, 0.1)
    // setImgLoading(false)
  }, []);

  // fetching images
  const convertDotToImg = async (x, y) => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/explorer/fetch_dots_to_img?1st_dot=${x * 2}&2nd_dot=${y * 2}&model_key=${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error('Failed to fetch');
      const imgResp = await response.json();
      console.log("Dots to img: ", imgResp);
      if (imgResp.data) {
        setImage(imgResp.data);
        // setImagesBuffer(prevBuffer => {
        //   const newBuffer = [...prevBuffer, imgResp.data];
        //   console.log("Image buffer size: ", newBuffer.length);
        //   return newBuffer;
        // });

      } else {
        console.error('No image data found');
      }
    } catch (error) {
      console.error('Error fetching image:', error);
    }
  };

   const batchConvertDotToImg = async (x1, y1, x2, y2) => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/explorer/fetch_dots_to_img_batch?x1=${x1 * 2}&y1=${y1 * 2}&x2=${x2 * 2}&y2=${y2 * 2}&model_key=${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) throw new Error('Failed to fetch');
      const imgResp = await response.json();
      console.log("Dots to img: ", imgResp);
      if (imgResp.data) {
        setImage(imgResp.data);
        // setImagesBuffer(prevBuffer => {
        //   const newBuffer = [...prevBuffer, imgResp.data];
        //   console.log("Image buffer size: ", newBuffer.length);
        //   return newBuffer;
        // });

      } else {
        console.error('No image data found');
      }
    } catch (error) {
      console.error('Error fetching image:', error);
    }
  }

  // *** Scheduled Timer Function ***
  // useEffect(() => {
  //   const processCoordinates = () => {
  //     if (coordinatesRef.current.length > 0) {
  //       const { x, y } = coordinatesRef.current.shift();
  //       convertDotToImg(x, y);
  //     }
  //   };
  //
  //   fetchTimerRef.current = setInterval(processCoordinates, 30);
  //   return () => clearInterval(fetchTimerRef.current);
  // }, []);
  //
  // *** Hover handler to store coordinates ***
  const handleHover = ({ x, y }) => {
    convertDotToImg(x, y);
    // console.log(x, y);
    // coordinatesRef.current.push({ x, y });
  };

  // Set looping index
  // useEffect(() => {
  //   if (imagesBuffer.length > 0) {
  //     displayTimerRef.current = setInterval(() => {
  //       setIndex((prevIndex) => Math.min(prevIndex + 1, imagesBuffer.length - 1));
  //       console.log('Loaded index:', index, ' , BufferSize: ', imagesBuffer.length);
  //     }, 10);
  //   }
  //
  //   return () => clearInterval(displayTimerRef.current);
  //
  // }, [imagesBuffer.length]);

  // // *** Debounced hover handler ***
  // const handleHover = ({ x, y }) => {
  //   console.log(x, y);
  //   if (hoverTimeoutRef.current) {
  //     clearTimeout(hoverTimeoutRef.current);
  //   }
  //   hoverTimeoutRef.current = setTimeout(() => {
  //     convertDotToImg(x, y);
  //   }, 3000); // 3 seconds delay
  // };

  return (
    <>
      {loading ? (<div className="flex justify-center items-center w-full">
        <svg className="animate-spin h-8 w-8 text-gray-600" xmlns="http://www.w3.org/2000/svg" fill="none"
             viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
        </svg>
      </div>) : (
          <div className="flex flex-col items-center justify-center w-full h-screen">
            <div className="flex mb-5 w-full px-10">
              <div className="w-full max-w-xs ml-40">
                <Select
                    label="Model"
                    labelProps={{
                      className: 'text-white text-ml',
                    }}
                    className="text-white"
                    value={selectedModel}
                    onChange={(e) => handleSelectChange(e)}
                >
                  {models.map((model) => (
                      <Option key={model.model_key} value={model.model_key}>
                        {model.model_name}
                      </Option>
                  ))}
                </Select>
              </div>
            </div>
            <div className="flex w-full justify-center px-10">
              <div className="flex items-center justify-center w-1/2 h-full">
                <XYPlot
                    data={dots}
                    width={400 - 10 - 10}
                    height={400 - 20 - 10}
                    xAccessor={(d) => d.x}
                    yAccessor={(d) => d.y}
                    margin={{top: 20, bottom: 10, left: 10, right: 10}}
                    onHover={handleHover}
                />
              </div>
              <div className="flex justify-center items-center w-1/2 h-full">
                <img
                    src={`data:image/png;base64,${image}`}
                    alt="Art Animation"
                    className="object-cover object-center w-96 h-96"
                />
              </div>
            </div>
          </div>
      )}
    </>

  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
    height: '100vh',
  },
  left: {
    width: '50%',
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  right: {
    width: '50%',
    height: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
};

export default Explore;