"use client"

import Navbar from "../components/NavBar";
import React from "react";
import { Slider } from "@material-tailwind/react";

import { Component, useEffect, useState, useRef } from "react"; 
import { useOpenCV } from "../../hooks/useOpenCV";  // for bicubic + sharpening
import { measureClassicalMethods } from "../../utils/imageProcessing";  // for bicubic + sharpening
import ImageCanvas from "../components/ImageCanvas";  // for Latent Space Explorer
import XYPlot from "../components/XYPlot";  
import * as tf from "@tensorflow/tfjs";  // for Latent Space Explorer
import gaussian from "gaussian";  // for Latent Space Explorer
import encodedData from "../encoded.json";  // for Latent Space Explorer

import "../page.module.css";  // for Latent Space Explorer

const MODEL_PATH = "models/generatorjs/model.json";

interface Props {
}

const Explore = () => {
  const [dots, setDots] = useState([]);
  const [image, setImage] = useState(null);
  const [isHovering, setIsHovering] = useState(false);
  const [loading, setLoading] = useState(true);

  // *** Storage array and timer reference for scheduled processing ***
  const coordinatesRef = useRef([]);
  const timerRef = useRef(null);

  const opencvLoaded = useOpenCV();  // for bicubic + sharpening
  
  // // *** Buffer state to handle debouncing ***
  // const hoverTimeoutRef = useRef(null);

  useEffect(() => {
    // Generate 3,000 random dots
    const generatedDots = [];
    for (let i = 0; i < 3000; i++) {
      generatedDots.push({
        x: (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5 ) * 2,
      });
    }
    setDots(generatedDots);
    console.log(generatedDots);
    setLoading(false)
  }, []);

  // fetching images
  const convertDotToImg = async (x, y) => {
    try {
      const x_f = x
      const y_f = y
      const response = await fetch(`http://127.0.0.1:5000/api/explorer/fetch_dots_to_img?1st_dot=${x_f}&2nd_dot=${y_f}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      }); // Adjust the API endpoint as necessary
      if (!response.ok) throw new Error('Failed to fetch');
      const imgResp = await response.json();
      console.log("Dots to img: ", imgResp);
      if (imgResp.data) {
        // setImage(imgResp.data);

        // *** Added for bicubic + sharpening ***
        const img = new Image();
        img.src = `data:image/png;base64,${imgResp.data}`;
        img.onload = async () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          const imageData = ctx.getImageData(0, 0, img.width, img.height);
          if (opencvLoaded) {
            const upscaledAndSharpened = await measureClassicalMethods(imageData, 2.0, 'bicubic');
            const outputCanvas = document.createElement('canvas');
            outputCanvas.width = upscaledAndSharpened.width;
            outputCanvas.height = upscaledAndSharpened.height;
            const outputCtx = outputCanvas.getContext('2d');
            outputCtx.putImageData(upscaledAndSharpened, 0, 0);
            setImage(outputCanvas.toDataURL());
          }
        };        
      } else {
        console.error('No image data found');
      }
    } catch (error) {
      console.error('Error fetching image:', error);
    }
  };

  // *** Scheduled Timer Function ***
  useEffect(() => {
    const processCoordinates = () => {
      if (coordinatesRef.current.length > 0) {
        const { x, y } = coordinatesRef.current.shift();
        convertDotToImg(x, y);
      }
    };

    timerRef.current = setInterval(processCoordinates, 1); // Process every 0.5 seconds

    return () => clearInterval(timerRef.current);
  }, [opencvLoaded]);

  // *** Hover handler to store coordinates ***
  const handleHover = ({ x, y }) => {
    // convertDotToImg(x, y);
    coordinatesRef.current.push({ x, y });
  };

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
      </div>) : (<div style={styles.container}>
      <div style={styles.left}>
        <XYPlot
          data={dots}
          width={500 - 10 - 10}
          height={500 - 20 - 10}
          xAccessor={d => d.x}
          yAccessor={d => d.y}
          // colorAccessor={d => d[2]}
          margin={{ top: 20, bottom: 10, left: 10, right: 10 }}
          onHover={handleHover} // *** Updated to use debounced handler ***
          // onHover={({ x, y }) => {
          //   console.log(x, y)
          //   convertDotToImg(x, y);
          // }}


        />
        {/* {isHovering && <div style={styles.hoverText}>Hovering</div>} */}
      </div>
      <div style={styles.right}>
        <img
          // src={`data:image/png;base64,${image}`}
          
          src={image}  // *** Added for bicubic + sharpening ***
          
          alt="Art Animation"
          style={{ width: '500px', height: '500px' }}
          className="object-cover object-center w-full h-256 max-w-full"
        // onMouseEnter={() => setIsPlaying(false)}
        // onMouseLeave={() => setIsPlaying(true)}
        />
        {/* <ImageCanvas /> */}
      </div>
    </div >)}
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


// class Explore extends Component<{}, { model, digitImg, mu, sigma:any}> {
//   constructor(props) {
//     super(props);
//     this.getImage = this.getImage.bind(this);

//     //this.norm = gaussian(0, 1);

//     this.state = {
//       model: null,
//       digitImg: tf.zeros([28, 28]),
//       mu: 0,
//       sigma: 0
//     };
//   }

//   // userEffect() -> {
//   //   data = fetchAllDataPoints()
//   //   XYPlot(data)
//   // }

//   componentDidMount() {
//     tf
//       .loadLayersModel(MODEL_PATH)
//       .then(model => this.setState({ model }))
//       .then(() => this.getImage())
//       .then(digitImg => this.setState({ digitImg }));
//   }

//   async getImage() {
//     const { model, mu, sigma } = this.state;
//     // fetch images 
//     // image_base64 = fetchImageWithDots(sigma, mu)
//     // setImage(image_base64)
//   }

//   render() {
//     return this.state.model === null ? (
//       <div>Loading, please wait</div>
//     ) : (
//       <div className="App">
//         <h1>Latent Space Explorer</h1>
//         <div className="ImageDisplay">
//           <ImageCanvas
//             width={500}
//             height={500}
//             imageData={this.state.digitImg}
//           />
//         </div>

//         <div className="ChartDisplay">
//           <XYPlot
//             data={encodedData}
//             width={500 - 10 - 10}
//             height={500 - 20 - 10}
//             xAccessor={d => d[0]}
//             yAccessor={d => d[1]}
//             colorAccessor={d => d[2]}
//             margin={{ top: 20, bottom: 10, left: 10, right: 10 }}
//             onHover={({ x, y }) => {
//               console.log(x,y )
//               //img = fetchImagesWithDot(x, y)
//               //setImage(img)
//               //this.setState({ sigma: y, mu: x });
//               //this.getImage().then(digitImg => this.setState({ digitImg }));
//             }}
//           />
//         </div>
//       </div>
//     );
//   }
// }


// Working Static V_1
// const Explore = () => {
//   // console.log(categories);
//   return (
//       <div className=" py-24 sm:py-32">
//         <div className="mx-auto grid max-w-7xl gap-x-10 gap-y-20 px-6 lg:px-8 xl:grid-cols-2">
//           <div className="max-w-2xl">
//             <div className="grid gap-x-8 gap-y-12">
//               <img
//                 className="object-cover object-center xl:col-span-1"
//                 src="https://www.researchgate.net/profile/Sergei-Astapov/publication/261467229/figure/fig2/AS:667663588143121@1536194812400/Three-clusters-in-a-three-dimensional-feature-space-Features-represent-energy-ratios-and.png"
//                 alt="feature space"
//                 style={{ height: 512, width: 512 }}
//               />
//               <div className="flex w-96 flex-col gap-2">
//                 <Slider size="lg" defaultValue={50} />
//                 <h3 className="text-base font-semibold leading-1 tracking-tight text-gray-900">Feature 1</h3>
//               </div>
//               <div className="flex w-96 flex-col gap-2">
//                 <Slider size="lg" color="blue" defaultValue={50} />
//                 <h3 className="text-base font-semibold leading-1 tracking-tight text-gray-900">Feature 2</h3>
//               </div>
//               <div className="flex w-96 flex-col gap-2">
//                 <Slider size="lg" color="green" defaultValue={50} />
//                 <h3 className="text-base font-semibold leading-1 tracking-tight text-gray-900">Feature 3</h3>
//               </div>
              
//               <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Control Panel</h2>
              
//             </div>
//           </div>
            
//           <div className="max-w-2xl">  
//             <img
//               className="object-cover object-center xl:col-span-1"
//               src="https://images.unsplash.com/photo-1682407186023-12c70a4a35e0?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2832&q=80"
//               alt="nature image"
//               style={{ height: 512, width: 512 }}
//             />
//             <h3 className="text-base font-semibold leading-1 tracking-tight text-gray-900">Output Image</h3>
//           </div>
//         </div>
//       </div>
//   );
// };

// export default Explore;