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
    """mnist dataset used in training neural networks about hand written letters."""

    def __init__(self):
        """Initialize the the mnist dataset with the file name,
         training,testing dataset."""

        self.conn = None
        self.file_name = "mnist.sqlite3"
        self.testing   = None
        self.training  = None
        if path.exists("mnist.sqlite3") == False:
            self.get_mnist_dataset()

    def get_mnist_dataset(self):
        """Since the file size is over 50MB we require the user to download it
        from the website https://www.datadb.dev/datadb/mnist/"""
        try:
            # Get the file
            print("Downloading MNIST dataset:")
            url = 'https://db1.datadb.dev/datadb/mnist/mnist.sqlite3?v=1'
            r = requests.get(url, stream=True)
            path = self.file_name
            with open(path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024),
                 expected_size=(total_length/1024) + 1):

                    if chunk:
                        f.write(chunk)
                        f.flush()
        except:
            print("""Error: Downloading of the MNIST Dataset requires an
                     ßinternet connection.""")
            print("Exiting....")
            exit()

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

    def show_image(self,image):
        """Function to show an image in matplot lib that is in the
        array format [image_label,image_array].
        """

        image_list = []
        for y in range(0, 28):
            image_x = []
            for x in range(0, 28):
                number = (y * 28) + x
                pixel = image[1][number]
                image_x.append(pixel)
            image_list.append(image_x)

        plt.title(f"Image Label: {image[0]}")
        plt.imshow(image_list)
        plt.show()


    def random_image(self):
        """Show random image from the mnist dataset."""

        data = self.query_one_item("""SELECT label, image
                                 FROM train
                                 ORDER BY random()
                                 LIMIT 1;""")

        image = self.bin_image_to_list(data[1])

        plt.title(f"Image Label: {data[0]}")
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


    def set_training_data(self):
        """Used to set the training object variable with the mnist training
        dataset."""

        # Get the training data
        data = self.query_all("""SELECT label, image FROM train""")

        # This may take a while so we want the user to know
        print("""Getting 50,000 training examples. This may take a minute:""")

        # Set the mnist objects training data variable to make it easy to acess.
        self.training = self.format_datset_data(data)

    def set_testing_data(self):
        """Used to set the testing object variable with the mnist testing
        dataset."""

        # Get the training data
        data = self.query_all("""SELECT label, image FROM train""")

        # This may take a while so we want the user to know
        print("""Getting 10,000 testing examples. This may take a minute:""")

        # Set the mnist objects training data variable to make it easy to acess.
        self.testing = self.format_datset_data(data)


    def format_datset_data(self,data):
        """Used to format the sqlite data of a tuple(label,
        image_byte_data) to array[image_label, [0,784] ]."""

        # Store all the formated arrays in a larger array.
        formated_data = []

        # For each (label, image in the training dataset)
        for image in data:

            temp = []
            io_reader = io.BytesIO(image[1])

            # Format the data into [0,784]
            for x in range(0, 784):
                pixel = struct.unpack('B', io_reader.read1(1))[0]
                temp.append(pixel)

                # Add the image label and translated image into a new array.
                translated_data = [image[0], temp]

            # Store into larger array
            formated_data.append(translated_data)

        return formated_data




    #=====================================#
    # Decorators:                         #
    #=====================================#

    def format_query_data(method):
        """Decorator used to format query data"""

        def new_method(self, config_id):
            query = method(self, config_id)

            # Unpack elements if they are lists with only 1 element
            if type(query[0]) in (list, tuple) and len(query[0]) == 1:
                query = [i[0] for i in query]

            # Unpack list if it is a list with only 1 element
            if type(query) in (list, tuple) and len(query) == 1:
                query = query[0]

            return query

        return new_method


    #=====================================#
    # Functions:                          #
    #=====================================#


    def create_table(self, create_table_sql):
        """Create a table from the create_table_sql statement."""

        try:
            c = self.conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)


    @format_query_data
    def query_all(self, query):
        """Query for muliple rows of data"""

        cur = self.conn.cursor()
        cur.execute(query)
        return cur.fetchall()


    @format_query_data
    def query_one_item(self, query):
        """Query for single data point"""

        cur = self.conn.cursor()
        cur.execute(query)
        return cur.fetchone()


    def remove_database(self):
        """Remove the current database file using the database_name attribute."""
        os.remove(self._database_name)


    @property
    def testing(self):
        """Getter function for testing"""
        # If the connection has not been set yet

        if self._testing is not None:
            return self._testing
        else:
            self.set_testing_data()
            return self._testing


    @testing.setter
    def testing(self, value_input):
        """Setter function for testing"""

        # Set the name in the ga attribute
        self._testing = value_input


    @property
    def training(self):
        """Getter function for conn"""
        # If the connection has not been set yet

        if self._training is not None:
            return self._training
        else:
            self.set_training_data()
            return self._training


    @training.setter
    def training(self, value_input):
        """Setter function for conn"""

        # Set the name in the ga attribute
        self._training = value_input

    @property
    def conn(self):
        """Getter function for conn"""

        # Return if the connection has already been set
        if self._conn is not None:
            return self._conn
        else:
        # If the connection has not been set yet
            try:
                # Check if you can connect to the database
                self.create_connection()
                return self._conn

            except:
                # If the connection doesnt exist then print error
                raise Exception("""mnist database has not been downloaded yet.""")

    @conn.setter
    def conn(self, value_input):
        """Setter function for conn"""

        # Set the name in the ga attribute
        self._conn = value_input
