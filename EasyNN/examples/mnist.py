# To structure 2d array for images
import struct
import io

# Database connection
import sqlite3

# For plotting images
import matplotlib.pyplot as plt

# Download dataset if it doesnt exist
import requests

# Check if file exists functions
import os.path
from os import path

from clint.textui import progress

class mnist:
    """mnist dataset used in training neural networks about hand written letts."""

    def __init__(self):
        """Initialize the the mnist dataset with the file name,
         training,testing dataset."""

        self.conn = None
        self.file_name = "mnist.sqlite3"
        self.training  = None
        self.testing   = None

        if path.exists("mnist.sqlite3") == False:
            self.get_mnist_dataset()


    def get_mnist_dataset(self):
        """Since the file size is over 50MB we require the user to download it
        from the website https://www.datadb.dev/datadb/mnist/"""

        # Get the file
        print("Downloading MNIST dataset:")
        url = 'https://db1.datadb.dev/datadb/mnist/mnist.sqlite3?v=1'
        r = requests.get(url, stream=True)
        path = self.file_name
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()


    def bin_image_to_list(self,bin_image):
        """
        Converts 28x28 2-d array where an item is a single byte binary
        number from 0 to 255. Row major order.

        :param bin_image: binary buffer
        :return: 2-d list of numbers, ranging 0-255
        """
        io_reader = io.BytesIO(bin_image)
        image_list = []
        for y in range(0, 28):
            image_x = []
            for x in range(0, 28):
                pixel = struct.unpack('B', io_reader.read1(1))[0]
                image_x.append(pixel)
            image_list.append(image_x)
        return image_list


    def random_image(self):
        """Show random image from the mnist dataset."""
        # Connect to the downloaded db file.

        # Create a cursor to iterate through rows.
        c = self.conn.cursor()

        # Select which data you want.
        c.execute("""SELECT label, image
                     FROM train
                     ORDER BY random()
                     LIMIT 1""")

        # Fetch single row
        row = c.fetchone()

        image = self.bin_image_to_list(row[1])

        plt.title(f"Image Label: {row[0]}")
        plt.imshow(image)
        plt.show()

    def create_connection(self):
        """Create a database connection to the SQLite database
         specified by db_file."""

        try:
            self.conn = sqlite3.connect(self.file_name)
        except Error as e:
            self.conn = None
            print(e)

    #=====================================#
    # Setters and Getters:                #
    #=====================================#

    @property
    def training(self):
        """Getter function for training data"""

        # Return if the training data has already been set
        if self._training is not None:
            return self._training
        else:
            self.set_data()
            return self._training

    @property
    def conn(self):
        """Getter function for conn"""

        # Return if the connection has already been set
        if self._conn is not None:
            return self._conn

        # If the connection has not been set yet
        try:
            # Check if you can connect to the database
            self.create_connection()
            return self._conn

        except:
            # If the connection doesnt exist then print error
            raise Exception("""mnist database has not been downloaded yet.""")
