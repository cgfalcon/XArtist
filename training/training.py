
import time
from PIL import Image
import src.gan_trainer as gan_trainer



if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None  # Completely removes the limit
    # or set to a higher limit, for example:
    Image.MAX_IMAGE_PIXELS = 100_000_000  # set to a more suitable limit
    start_ts = time.time()
    print(f'Training started ....')
    gan_trainer.run()
    print(f'Training finished ... ({(time.time() - start_ts)/1000} s)')