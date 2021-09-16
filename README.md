# Image Similarity with Transformer Neural Networks

Imsim project provides functionality for vectorizing images
for the subsequent implementation of the image search by these vectors.

Image processing is carried out by models of neural networks built with a transformer architecture.

+ [Project structure](#project_structure)
+ [Getting started](#getting_started)
+ [Authors](#authors)
+ [License](#license)
+ [Acknowledgments](#acknowledgments)

<div id="project_structure" />

### Project structure

+ **/ internal /** - CLI scripts that were required during the creation of the application
+ **/ models /** - Saved models
+ **/ src /** - Internal logic
  + **/ src / facades /** - Facades that encapsulate different parts of the application for ease of use
  + **/ src / mediators /** - Mediators used in the role of contracts to ensure consistency of data types passed in an application
  + **/ src / models /** - Model classes that provide the logic for delivering the model to RAM / GPU RAM
  + **/ src / utils /** - Utilities to help work the main business logic
  + **/ src / bl.py** - Business logic
  + **/ src / dtypes.py** - Data types used in the application
  + **/ src / exceptions.py** - Custom exceptions
  + **/ src / timer.py** - Not used in the application, but deliberately left decorator for profiling the speed of functions
+ **/ config.py** - Config file
+ **/ main.py** - Entry file

<div id="getting_started" />

## Getting started

The necessary information on using the program can be obtained from the
[FastApi](https://fastapi.tiangolo.com/tutorial/first-steps/)
framework documentation, since it is on it that the program runs.

<div id="authors" />

## Authors

+ **Tsotne Otanadze** ( [LinkedIn](https://www.linkedin.com/in/otanadzetsotne/) )

<div id="license" />

## License

This project is licensed under the [MIT License](https://en.wikipedia.org/wiki/MIT_License) - 
see the [LICENSE](https://github.com/otanadzetsotne/nn-image-similarity/blob/main/LICENSE) file for details.

<div id="acknowledgments" />

## Acknowledgments
+ Transformer classifiers are taken from *[lukemelas/PyTorch-Pretrained-ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT)*.
