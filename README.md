# Computer Vision & Image Processing Desktop Application

## Description

Welcome to our Image Processing Desktop Application! This application offers a comprehensive set of tools and features for performing various image processing tasks, making it an indispensable tool for professionals and enthusiasts alike. Whether you're a seasoned image processing expert or just getting started, our app provides an intuitive interface and powerful functionality to meet your needs.

With our desktop application, users can effortlessly load images, apply a wide range of image processing techniques, visualize the results in real-time, and save the processed images with ease. From basic operations like converting images to grayscale and binary formats to advanced features such as edge detection and object recognition, our app empowers users to explore and manipulate images in creative and innovative ways.

## Features


- **Noise Generation:**
  - Generate and apply different types of noise, including Gaussian, Uniform, and Salt & Pepper noise, to images.

   |         Uniform Noise          |         Guassian Noise          |        Salt & Pepper Noise         |
   | :----------------------------: | :-----------------------------: | :--------------------------------: |
   | img | img | img |

- **Image Filtering:**
  - Apply various filters including Average, Gaussian, and Median filters to enhance image quality and reduce noise.
    
   |         Average Filter          |         Gaussian Filter          |         Median Filter          |
   | :-----------------------------: | :------------------------------: | :----------------------------: |
   |  |  |  |

- **Histogram Visualization and Equalization:**
  - Visualize histograms and distribution curves of images to understand their pixel intensity distribution.
  - Perform histogram equalization and normalization to enhance image contrast and improve overall appearance.

   | Histogram | Normalization | Equalization |  
   | :----------------------------: | :-----------------------------: | :-----------------------------: |
   |  |  | |

- **Thresholding Techniques:**
  - Apply various thresholding techniques, including Global and Local Thresholding, to segment images and extract important features.
 
   | Local | Global |  
   | :----------------------------: | :-----------------------------: |
   | ![](Results/local-thresh.png) | ![](Results/global-thresh.png) |

- **Frequency Domain Filters:**
  - Apply frequency domain filters such as Low Pass and High Pass filters to manipulate image frequency components for image enhancement and noise reduction.
  - Create hybrid images by combining two images in the frequency domain to create visually appealing compositions.
    
   | High pass | Low pass |  
   | :----------------------------: | :-----------------------------: |
   | ![](Results/high-pass.png) | ![](Results/low-pass.png) |

    
 
  
- **Edge Detection:**
  - Utilize edge detection algorithms such as Sobel, Perwitt, Robert, and Canny for detecting edges and contours in images.
    
   | Sobel | Roberts | Perwitt | Canny |
   | :----------------------------: | :-----------------------------: | :--------------------------------: | :--------------------------------: |
   |  |  | |  |


- **Hough Transform:**
  - Detect lines, circles, and ellipses in images using the Hough Transform method for robust shape detection and recognition.
 
   | Line | Circle | Ellipse |  
   | :----------------------------: | :-----------------------------: | :-----------------------------: |
   |  |  |  |

- **Active Contour:**
  - Use active contour models (also known as snakes) to detect and track object boundaries in images.
  - Allow users to select an initial contour or region of interest (ROI) in the image.
  - Tune parameters such as alpha, beta, gamma, and iterations to control the behavior and convergence of the active contour algorithm.
  - Enable real-time visualization of the active contour evolution and final segmentation result.

   | Example 1 | Example 2 |  
   | :----------------------------: | :-----------------------------: |
   | | |

- **Feature Detection:**
  - Detect image features using methods like Harris Corner Detector and Lambda Corner Detector for point feature extraction.
  - Perform image matching and template matching to identify similarities and patterns in images.
    
   | Harris | Lambda |  
   | :----------------------------: | :-----------------------------: |
   |  | |

- **Feature Matching:**
  
   | Square Sum of Differences method | Cross-Correlation Method |  
   | :----------------------------: | :-----------------------------: |
   | | |

  
- **SIFT Descriptors:**
  - Utilize Scale-Invariant Feature Transform (SIFT) descriptors to detect and describe key points in images.
  - Compute keypoint matching between images for image registration and alignment tasks.

    <p align="center">
     <img src="img" />
   </p>

- **Thresholding Segmentation:**
  - Apply advanced thresholding segmentation techniques, including Otsu, Spectral, Optimum local, and global thresholding, to segment images into distinct regions and objects.
    
   | Local Otsu | Local Spectral | Local Optimum |  
   | :----------------------------: | :-----------------------------: | :-----------------------------: |
   |  |  |  |

   | Global Otsu | Global Spectral | Global Optimum |  
   | :----------------------------: | :-----------------------------: | :-----------------------------: |
   |  |  |  |

- **RGB Image Segmentation:**
  - Segment RGB images using K-means clustering, mean shift clustering, Agglomerative Segmentation, and Region Growing methods for semantic segmentation and object detection tasks.

    | K-Means Segmentation | Mean Shift Segmentation |
    | :----------------------------: | :-----------------------------: |
    | img |img |

    | Agglomerative Segmentation |         Region Growing          |
    | :------------------------: | :-----------------------------: |
    |  img  | img |

## Summary

Our Image Processing Desktop Application offers a comprehensive suite of features and tools for image manipulation, analysis, and enhancement. With its intuitive interface and powerful functionality, users can explore the world of digital imagery with ease and efficiency. Whether you're a professional photographer, researcher, or hobbyist, our app provides the tools you need to unleash your creativity and achieve stunning results.

## How to Run the Program

To run the Image Processing Desktop Application, follow these simple steps:

1. Clone or download the repository to your local machine.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Launch the application by running `python main.py` from the terminal or command prompt.
4. Explore the various features and tools available in the application's graphical user interface (GUI).
5. Load images, apply image processing techniques, visualize the results, and save the processed images as needed.
6. Enjoy the power and versatility of our Image Processing Desktop Application!

   ## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Muhannad159" target="_black">
    <img src="https://avatars.githubusercontent.com/u/104541242?v=4" width="150px;" alt="Muhannad Abdallah"/>
    <br />
    <sub><b>Muhannad Abdallah</b></sub></a>
    </td>
  <td align="center">
    <a href="https://github.com/AliBadran716" target="_black">
    <img src="https://avatars.githubusercontent.com/u/102072821?v=4" width="150px;" alt="Ali Badran"/>
    <br />
    <sub><b>Ali Badran</b></sub></a>
    </td>
     <td align="center">
    <a href="https://github.com/ahmedalii3" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110257687?v=4" width="150px;" alt="Ahmed Ali"/>
    <br />
    <sub><b>Ahmed Ali</b></sub></a>
    </td>
<td align="center">
    <a href="https://github.com/ossama971" target="_black">
    <img src="https://avatars.githubusercontent.com/u/40814982?v=4" width="150px;" alt="Hassan Hussein"/>
    <br />
    <sub><b>Osama Badawi</b></sub></a>
    </td>
      </tr>
 </table>



