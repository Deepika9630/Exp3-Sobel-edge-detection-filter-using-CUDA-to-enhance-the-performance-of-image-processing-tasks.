# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>ENTER YOUR NAME : DEEPIKA R</h3>
<h3>ENTER YOUR REGISTER NO : 212224230054</h3>
<h3>EX. NO: 3</h3>
<h3>DATE:24.08.2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```

<img width="1733" height="362" alt="image" src="https://github.com/user-attachments/assets/3dfd2823-2348-4895-8e6f-df145ee6e79b" />

```
!nvcc --version
```

<img width="1727" height="121" alt="image" src="https://github.com/user-attachments/assets/ac940354-c3cb-4055-a6d4-630f17c1fbb6" />

```
%load_ext nvcc4jupyter
```

<img width="1727" height="62" alt="image" src="https://github.com/user-attachments/assets/fee8b869-37cf-4023-8d18-5a497843696f" />

```
from pathlib import Path

file_path = Path('/absolute/path/to/images.jpeg')
if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")
```

<img width="1732" height="37" alt="image" src="https://github.com/user-attachments/assets/26902751-a2c6-476f-8ee9-a4bb05789d3e" />

```
import os
print("Current Working Directory:", os.getcwd())
```

<img width="1728" height="35" alt="image" src="https://github.com/user-attachments/assets/4aaa7124-28ba-49c7-9679-9c02d46afbff" />

```
from google.colab import files
uploaded = files.upload()
```

<img width="1720" height="95" alt="image" src="https://github.com/user-attachments/assets/9f29c652-87fc-4ba9-898f-51d5097c54af" />

```
from pathlib import Path

file_path = Path("/content/image.jpg")

if file_path.exists():
    print("File exists!")
else:
    print("File does not exist!")
```

<img width="1720" height="40" alt="image" src="https://github.com/user-attachments/assets/f2cefe5f-ab64-4560-9517-9db7f08c258f" />

```
pwd
```

<img width="1720" height="42" alt="image" src="https://github.com/user-attachments/assets/3ab6f5fd-9fbe-4fb4-96ce-80893aa73198" />

```
ls "/content/image.jpg"
```

<img width="1717" height="37" alt="image" src="https://github.com/user-attachments/assets/e55ab36f-e786-4a55-8e8d-53e39bcb2ad8" />

```
#ls -l /content/66666.jpg
import cv2
image = cv2.imread('/content/image.jpg')
if image is None:
    print("Error: Image not found or unable to read the image.")
else:
    print("Image read successfully.")
```

<img width="1708" height="36" alt="image" src="https://github.com/user-attachments/assets/b81f3dea-4fe6-4191-8702-de56d8bf6591" />

```
%%writefile sobelEdgeDetectionFilter.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage,
                            unsigned char *dstImage,
                            unsigned int width,
                            unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float Kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    float Ky[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    // Only threads inside the valid image area
    if (x >= 1 && x < width - 1 &&
        y >= 1 && y < height - 1)
    {
        // Gradient in x-direction
        float Gx = 0;

        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                float fl = srcImage[
                    (y + ky) * width + (x + kx)
                ];

                Gx += fl * Kx[ky + 1][kx + 1];
            }
        }

        float Gx_abs = Gx < 0 ? -Gx : Gx;

        // Gradient in y-direction
        float Gy = 0;

        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                float fl = srcImage[
                    (y + ky) * width + (x + kx)
                ];

                Gy += fl * Ky[ky + 1][kx + 1];
            }
        }

        float Gy_abs = Gy < 0 ? -Gy : Gy;

        float result = Gx_abs + Gy_abs;

        if (result > 255)
            result = 255;

        dstImage[y * width + x] =
            (unsigned char)result;
    }
    else if (x < width && y < height)
    {
        dstImage[y * width + x] = 0;
    }
}

void checkCudaErrors(cudaError_t r)
{
    if (r != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA Error: %s\n",
                cudaGetErrorString(r));

        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Read input image
    Mat image = imread(
        "/content/66666.jpg",
        IMREAD_GRAYSCALE
    );

    if (image.empty())
    {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize =
        width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage =
        (unsigned char *)malloc(imageSize);

    if (h_outputImage == nullptr)
    {
        fprintf(stderr,
                "Failed to allocate host memory\n");

        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    checkCudaErrors(
        cudaMalloc(&d_inputImage, imageSize)
    );

    checkCudaErrors(
        cudaMalloc(&d_outputImage, imageSize)
    );

    checkCudaErrors(
        cudaMemcpy(
            d_inputImage,
            image.data,
            imageSize,
            cudaMemcpyHostToDevice
        )
    );

    // Define CUDA events for timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + 15) / 16,
        (height + 15) / 16
    );

    cudaEventRecord(start);

    sobelFilter<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        width,
        height
    );

    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    // Copy result back to host
    checkCudaErrors(
        cudaMemcpy(
            h_outputImage,
            d_outputImage,
            imageSize,
            cudaMemcpyDeviceToHost
        )
    );

    // Write output image
    Mat outputImage(
        height,
        width,
        CV_8UC1,
        h_outputImage
    );

    imwrite(
        "/content/output_sobel.jpeg",
        outputImage
    );

    // Free memory
    free(h_outputImage);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n",
           milliseconds);

    return 0;
}
```


<img width="1711" height="35" alt="image" src="https://github.com/user-attachments/assets/099e55bf-42b3-4116-abe7-07f898174bf2" />


```
!nvcc -o sobelEdgeDetectionFilter sobelEdgeDetectionFilter.cu `pkg-config --cflags --libs opencv4`
```


<img width="1728" height="218" alt="image" src="https://github.com/user-attachments/assets/4c31ace5-c506-406f-b82a-de2faf71be79" />


```
!./sobelEdgeDetectionFilter
```

<img width="1537" height="46" alt="image" src="https://github.com/user-attachments/assets/9a75dd67-0e76-4c0a-a195-892ef83b1920" />


```
import cv2
from matplotlib import pyplot as plt
```

```
output_image_path = '/content/image.jpg'
output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)  # Use IMREAD_GRAYSCALE if it's a single-channel image
```
```
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detection Output')
plt.axis('off')  # Hide the axes
plt.show()
```

<img width="1187" height="533" alt="image" src="https://github.com/user-attachments/assets/fc564a84-8524-4557-bc7d-c4c7a69c4665" />




## RESULT:
Thus the program has been executed by using CUDA to Edge detection.

Questions:

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

