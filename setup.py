import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    description=
        'EasyNN is a python package designed to provide an easy-to-use neural '
        'network. The package is designed to work right out of the box, while '
        'also allowing the user to customize features as they see fit.',
)
