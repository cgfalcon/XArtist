// /utils/api.jsx
export async function measureClassicalMethods(imageData, scaleFactor, method, sigma = 1.0, strength = 1.3) {
  return new Promise((resolve, reject) => {
    try {
      console.log('Starting measureClassicalMethods');
      console.log('imageData:', imageData);

      // Validate imageData
      if (!imageData || !imageData.data || !imageData.width || !imageData.height) {
        throw new Error('Invalid imageData');
      }

      let src;
      try {
        src = cv.matFromImageData(imageData);
      } catch (err) {
        console.error('Error creating src mat from imageData:', err);
        reject(err);
        return;
      }

      console.log('Created src mat:', src);
      console.log('src properties:', { cols: src.cols, rows: src.rows, width: src.size().width, height: src.size().height });

      let dst = new cv.Mat();
      let dsize = new cv.Size(src.size().width * scaleFactor, src.size().height * scaleFactor);  // Use src.size() instead of src.width and src.height

      // Resize using the specified interpolation method
      let interpolation;
      switch (method) {
        case 'nearest':
          interpolation = cv.INTER_NEAREST;
          break;
        case 'bilinear':
          interpolation = cv.INTER_LINEAR;
          break;
        case 'bicubic':
          interpolation = cv.INTER_CUBIC;
          break;
        case 'lanczos':
          interpolation = cv.INTER_LANCZOS4;
          break;
        default:
          interpolation = cv.INTER_CUBIC;
      }

      cv.resize(src, dst, dsize, 0, 0, interpolation);
      console.log('Resized image:', dst);

      // Apply Gaussian Blur
      let ksize = new cv.Size(0, 0);
      cv.GaussianBlur(dst, dst, ksize, sigma);
      console.log('Applied Gaussian Blur:', dst);

      // Apply Unsharp Masking
      let sharpened = new cv.Mat();
      cv.addWeighted(dst, 1 + strength, dst, -strength, 0, sharpened);
      console.log('Applied Unsharp Masking:', sharpened);

      // Convert the result back to ImageData
      let imageDataResult = new ImageData(new Uint8ClampedArray(sharpened.data), sharpened.size().width, sharpened.size().height);
      resolve(imageDataResult);

      // Clean up
      src.delete();
      dst.delete();
      sharpened.delete();
    } catch (err) {
      console.error('Error in measureClassicalMethods:', err);
      reject(err);
    }
  });
}