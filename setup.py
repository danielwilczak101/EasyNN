import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name='EasyNN',
    version='0.0.6',
    description='EasyNN is a python package designed to provide an easy-to-use neural network. The package is designed to work right out of the box, while also allowing the user to customize features as they see fit.',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    url="https://github.com/danielwilczak101/EasyNN",
    author="Daniel Wilczak, Jack Nguyen, David Fadini, Nathan Foster, Liam Kehoe, Nathan Rose",
    author_email="danielwilczak101@gmail.com",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    classifier=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
        ],
    install_requires = ["matplotlib ~= 3.3.2",
                        "clint>=0.5.1",
                        "tabulate >=0.8.7"
                        ],
    )
