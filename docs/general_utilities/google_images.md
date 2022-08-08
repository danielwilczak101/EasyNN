Let EasyNN help you create a collection of images for your dataset.

### Examples:
```Python
from EasyNN.utilities.scrape import scrape_google
scrape_google("dog")
```
Downloads 5 images of dogs from google and puts the into images folder.
```Python
from EasyNN.utilities.scrape import scrape_google
scrape_google("dog", count=100)
```
Downloads 100 images of dog from google.

```Python
from EasyNN.utilities.scrape import scrape_google
scrape_google("dog", target_path='./my_new_folder')
```
Downloads 5 images from google of a dog and put it into a folder named "my_new_folder".

### Function parameters:
    
**search_term:** The term you want to scrape google images for.  

**count:** Number of image to scrape from google.  

**target_path:** Creates a directory with target_path name and puts all the images inside here.  

### Return:
Images folder with sub folder of the search term with images inside of there.