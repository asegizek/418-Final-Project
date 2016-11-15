#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <thrust/scan.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "grid.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct global_constants {
  int grid_width;
  int grid_height;
  grid_elem* grid;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ global_constants const_params;

// kernelClearGrid --  (CUDA device code)
//
// Clear the grid, setting all cells to 0
__global__ void kernel_clear_grid() {

  // cells at border are not modified
  int image_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int image_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

  int width = const_params.grid_width;
  int height = const_params.grid_width;

  // cells at border are not modified
  if (image_x >= width - 1 || image_y >= height - 1)
      return;

  int offset = image_y*width + image_x;

  // write to global memory
  *(grid_elem*)(&global_constants.grid[offset]) = DEAD;
}

#define THREAD_DIMX 32
#define THREAD_DIMY 32


// kernelSingleCycle -- (CUDA device code)
//
// fill in each array of blockCircles so that it represents which circles
// overlap with each block
__global__ void kernalComputeLocalCircles(int *blockCircles, int arraySize) {

    // index of this block in blockCircles
    int blockIndex = blockIdx.y*gridDim.x + blockIdx.x;
    // circle number of the circle this thread will deal with
    int circleIndex = blockIdx.z*blockDim.x + threadIdx.x;

    int numCircles = cuConstRendererParams.numCircles;

    // only continue if circleIndex is valid
    if (circleIndex < numCircles) {

        float3 p = *(float3*) (&cuConstRendererParams.position[circleIndex*3]);
        float  rad = cuConstRendererParams.radius[circleIndex];
        short imageWidth = cuConstRendererParams.imageWidth;
        short imageHeight = cuConstRendererParams.imageHeight;

        int minPixelX = THREAD_DIMX*blockIdx.x;
        int maxPixelX = THREAD_DIMX*(blockIdx.x + 1);
        int minPixelY = THREAD_DIMY*blockIdx.y;
        int maxPixelY = THREAD_DIMY*(blockIdx.y + 1);

        // compute wether circle can intersect with the current block
        int inRange = 1;
        inRange = inRange &&
            maxPixelY > static_cast<short>(imageHeight * (p.y - rad));
        inRange = inRange &&
            minPixelY <=  static_cast<short>(imageHeight * (p.y + rad));
        inRange = inRange &&
            maxPixelX > static_cast<short>(imageWidth * (p.x - rad));
        inRange = inRange &&
            minPixelX <=  static_cast<short>(imageWidth * (p.x + rad));

        // in order to signal the last position in the array, the last
        // circle being rendered is put into every array in blockCircles
        inRange = inRange || (circleIndex == numCircles - 1);
        blockCircles[blockIndex*arraySize + circleIndex] = inRange;
    }


}

// create the keys vector
__global__ void
kernalComputeKeys(int *keys, int arraySize) {

    // get the keys array specific to this block
    keys += blockIdx.y*arraySize;
    // index of this thread in the keys array
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < arraySize) {
        keys[index] = blockIdx.y;
    }
}

__global__ void
kernalCreateCircleList(int *blockCircles, int *blockCirclesScan,
                        int *circleList, int arraySize) {
    // get the arrays specific to this block
    blockCircles += blockIdx.y*arraySize;
    blockCirclesScan += blockIdx.y*arraySize;
    circleList += blockIdx.y*arraySize;

    // index of this thread in any of the arrays
    int listIndex = blockIdx.x*blockDim.x + threadIdx.x;

    int numCircles = cuConstRendererParams.numCircles;

    if (listIndex < numCircles) {
        if (blockCircles[listIndex]) {
            circleList[blockCirclesScan[listIndex]] = listIndex;
        }

    }
}

////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce GTX 480") == 0
            || name.compare("GeForce GTX 670") == 0
            || name.compare("GeForce GTX 780") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA GTX 480, 670 or 780.\n");
        printf("---------------------------------------------------------\n");
    }

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaThreadSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaThreadSynchronize();
}

void
CudaRenderer::render() {

    // block/grid size for the pixel kernal
    dim3 pixelBlockDim(THREAD_DIMX, THREAD_DIMY);
    dim3 pixelGridDim((image->width + pixelBlockDim.x - 1) / pixelBlockDim.x,
                (image->height + pixelBlockDim.y - 1) / pixelBlockDim.y);

    // size of each circle array
    int arraySize = numCircles;

    // skip parts of the implementation if there aren't many circles
    int quick = (numCircles < 30);

    // initialize blockCircles so it has enough space to hold each array
    int numBlocks = pixelGridDim.x*pixelGridDim.y;

    if (!quick) {
        // contains the arrays of circles which can overlap with each block
        thrust::device_ptr<int> blockCircles =
            thrust::device_malloc<int>(numBlocks*arraySize);
        // will contain the exclusive scan of blockCircles
        thrust::device_ptr<int> blockCirclesScan =
            thrust::device_malloc<int>(numBlocks*arraySize);
        // contains keys for the exclusive scan
        thrust::device_ptr<int> keys =
            thrust::device_malloc<int>(numBlocks*arraySize);
        // will contain arrays of all circles that overlap with each block
        thrust::device_ptr<int> circleList =
            thrust::device_malloc<int>(numBlocks*arraySize);

        // block/gird size for setting up blockCircles
        dim3 circleBlockDim(512);
        dim3 circleGridDim(pixelGridDim.x, pixelGridDim.y,
                (numCircles + circleBlockDim.x - 1) / circleBlockDim.x);

        // fill in each array of blockCircles so that it represents which
        // circles overlap with each block
        kernalComputeLocalCircles<<<circleGridDim, circleBlockDim>>>
                (blockCircles.get(), arraySize);
        cudaThreadSynchronize();

        // block/gird size for setting up keys
        dim3 keyBlockDim(512);
        dim3 keyGridDim((arraySize + keyBlockDim.x - 1)/keyBlockDim.x,
                numBlocks);

        // create the keys vector
        kernalComputeKeys<<<keyGridDim, keyBlockDim>>>(keys.get(), arraySize);

        thrust::exclusive_scan_by_key(keys, keys+arraySize*numBlocks,
                blockCircles, blockCirclesScan);

        // block/gird size for circleList
        dim3 listBlockDim(512);
        dim3 listGridDim((numCircles + listBlockDim.x - 1) / listBlockDim.x,
                numBlocks);

        // fill in circeList
        kernalCreateCircleList<<<listGridDim, listBlockDim>>>
            (blockCircles.get(), blockCirclesScan.get(),
             circleList.get(), arraySize);
        cudaThreadSynchronize();

        // shade the pixels based on which circles overlap with them
        kernelRenderPixels<<<pixelGridDim, pixelBlockDim>>>
            (circleList.get(), arraySize);

        cudaThreadSynchronize();
        thrust::device_free(circleList);
        thrust::device_free(keys);
        thrust::device_free(blockCirclesScan);
        thrust::device_free(blockCircles);
    } else {
        kernelRenderSmall<<<pixelGridDim, pixelBlockDim>>>();
        cudaThreadSynchronize();
    }
}
