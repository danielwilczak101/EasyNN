EasyNN has downloading files from the internet built into the package. Downloading is even built into our examples to make it easier for our users to try out our examples.

#### To use the download utility:
```Python
from EasyNN.utilities import download

# Save as file name / Url
download("dress.jpg","https://bit.ly/3b7rsXF")
```

#### Ouput:
```
Downloading - dress.jpg:
[################################] 25/25 - 00:00:00
```

With this example the file dress.jpg will be downloaded and saved where ever your file was run from.