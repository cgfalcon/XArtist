// /utils/api.jsx
export async function measureClassicalMethods(imageData, scaleFactor, method, sigma = 1.0, strength = 1.3) {
  return new Promise((resolve, reject) => {
    try {
      let src = cv.matFromImageData(imageData);
      let dst = new cv.Mat();
      let dsize = new cv.Size(src.cols * scaleFactor, src.rows * scaleFactor);

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

      // Apply Gaussian Blur
      let ksize = new cv.Size(0, 0);
      cv.GaussianBlur(dst, dst, ksize, sigma);

      // Apply Unsharp Masking
      let sharpened = new cv.Mat();
      cv.addWeighted(dst, 1 + strength, dst, -strength, 0, sharpened);

      // Convert the result back to ImageData
      let imageDataResult = new ImageData(new Uint8ClampedArray(sharpened.data), sharpened.cols, sharpened.rows);
      resolve(imageDataResult);

      // Clean up
      src.delete();
      dst.delete();
      sharpened.delete();
    } catch (err) {
      reject(err);
    }
  });
}
