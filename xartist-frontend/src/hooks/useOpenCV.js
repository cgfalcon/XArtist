'use client';

import { useEffect, useState } from 'react';

export const useOpenCV = (onLoadCallback) => {
  const [opencvLoaded, setOpencvLoaded] = useState(false);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/3.4.0/opencv.js';
    script.async = true;
    script.onload = () => {
      console.log('OpenCV.js is ready');
      setOpencvLoaded(true);
      if (onLoadCallback) onLoadCallback();
    };
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, [onLoadCallback]);

  return opencvLoaded;
};