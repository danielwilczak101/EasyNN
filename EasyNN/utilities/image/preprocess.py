import numpy as np
from PIL import ImageOps, Image as pil_image

__all__ = ["Preprocess"]

class Preprocess:
    """Used for preprocessing images to a specific look.

    Args:
        image_path: Path to where the image is located relative to the executing file.
    """

    def __init__(self, image_path: str):
        try: 
            # We opened the image
            self.image = pil_image.open(image_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"{image_path} not found.") from None

    def format(
        self,
        grayscale=False,
        invert:bool=False,
        rotate:int=0,
        contrast:int=0,
        resize:list[int]=False,
        flatten:str='C',
        process: bool=False,
        convert: str=None,
        intensity: int=0
        ):
        """Used to specify how they want the format of the image to look like.
        
        Args:
            grayscale: Used to make an RGB image into grayscale format.
            invert: Light areas are mapped to dark, and dark areas are mapped to light.
            rotate: If the image needs to be rotated by a multiple of 90 degrees.
                Ex. rotate = 1 -> rotate image 90 degrees \n
                Ex. rotate = 3 -> rotate image 270 degrees
            contrast: How much addition pixel strength you want to remove from the array.
                Ex. original = [10,20,30] if offest = 5 then new_array = [5,15,25]
            resize: Specify an a desired pixel/image size.
                Ex. resize([28,28]) -> Image will now be a 28x28 pixel image.   
        """
        if convert:
            self.image.convert(convert)
        if grayscale:
            # Convert to grey scale.
            self.image = self.image.convert('L')
        if invert:
            # Inverted the color to look like our dataset images.
            self.image = ImageOps.invert(self.image)
        if resize:
            # Resize the iamge.
            self.image = self.image.resize(resize)

        # Convert to numpy array and not an image.
        self.image = np.array(self.image)
        
        if rotate:
            # Transofrm the image into a numpy array and apply rotation if needed.
            self.image = np.rot90(self.image, k=rotate)
        if process:
            # Apply contrast filter out all the grey noise in our image using the mean value.
            self.image = np.where(self.image < np.mean(self.image) + contrast,0,self.image)
        # Flatten the image so it can be used in the image and show functions.
        if flatten:
            self.image = self.image.flatten(flatten)
        if intensity:
            # Expose / brighten the pixels that are non zero.
            self.image[self.image!=0] += intensity
        
        return self.image

        