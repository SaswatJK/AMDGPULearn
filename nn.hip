#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand/hiprand_kernel.h>
#include <hiprand/hiprand.h>

typedef uint8_t   u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t    i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float    f32;
typedef double   f64;
typedef u8      byte;

typedef enum{
    TA_NONE,
    TA_TANHF,
    TA_SIGMOID,
    TA_RELU,
    TANUM,
}TYPE_ACTIVATION;

typedef enum{
    TL_INPUT,
    TL_HIDDEN,
    TL_OUTPUT,
    TLNUM
}TYPE_LAYER;

typedef struct Layer Layer;

struct Layer{
    Layer*          nextLayer;
    Layer*          prevLayer; // This will be used to get the input values from.
    float*          hiddenValues; // Should set this to 0 at the start.
    float*          activatedValues; // Should set this to 0 at the start.
    float*          weights; // Number of nodes in the prev layer * number of nodes in this layer. Each node has prev layer of nodes conected to it, and the weights are together like that. So each node's weights to the other nodes are contiguous.
    float*          bias; // Number of nodes in this layer.
    float*          weightGradients;
    float*          biasGradients;
    u16             weightNum;
    u16             numNodes;
    TYPE_LAYER      layerType;
    TYPE_ACTIVATION activationType;
};

typedef struct{
    Layer*  layers;
    u16     numLayers;
}NeuralNetwork;

typedef struct { uint64_t state; } RNG;

Layer CreateLayer(TYPE_LAYER layerType, TYPE_ACTIVATION activationType, u16 numNodes, Layer* prevLayer){
    Layer tempLayer;
    u16 prevNodes = 1;
    tempLayer.numNodes = numNodes;
    tempLayer.prevLayer = prevLayer;
    tempLayer.activationType = activationType;
    tempLayer.weightNum = numNodes * prevNodes;
    if(layerType != TL_INPUT)
        prevNodes = prevLayer->numNodes;
    hipError_t error;
    error = hipMalloc(&tempLayer.hiddenValues, numNodes * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the hidden values: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&tempLayer.activatedValues, numNodes * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the activated values: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&tempLayer.weights, tempLayer.weightNum * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the weights: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&tempLayer.bias, numNodes * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the biases: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&tempLayer.weightGradients, tempLayer.weightNum * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the weight gradients: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&tempLayer.biasGradients, numNodes * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the bias gradients: \%sn", hipGetErrorName(error));
        abort();
    }
    return tempLayer;
}

uint32_t rngNext(RNG* rng){
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = ((old >> 18u) ^ old) >> 27u;
    uint32_t rot = old >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float MSE(float value, float realValue){
    return ((value - realValue) * (value - realValue));
}

float rngNextf(RNG* rng){
    return (float)rngNext(rng) / (float)0xFFFFFFFF;
}

__device__ f32 activationTan(f32 input){
    return tanhf(input);
}

__device__ f32 gradTan(f32 input){
    return (1 - (tanhf(input) * tanhf(input)));
}

/*

    The way weights and biases works here is that each node's bias and weights are associated with the incoming connections.
    The weights are also contiguous, so all the input connections are contiguous to one singular node, instead of outgoing being contiguous.

*/

__global__ void processLayer(f32* inputs, i16 numInputs, f32* weights, f32* biases, f32* hiddenValues, f32* activatedValues, i16 numNodes){
    i16 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;
    i32 currentWeightNum = i * numInputs;
    for(i16 j = 0; j < numInputs; j++){
        hiddenValues[i] += weights[currentWeightNum + j] * inputs[j];
    }
    hiddenValues[i] += biases[i];
    activatedValues[i] = activationTan(hiddenValues[i]);
}

__global__ void computeGradientOfLayer(f32* inputs, i16 numInputs, f32* hiddenValues, f32* activatedValues, f32* weightGradients, f32* biasGradients, f32* weightsNextLayer, f32* weightGradientsNextLayer, f32* biasGradientsNextLayer, i16 numNodesNextLayer, i16 numNodes){
    i16 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;
    i32 nextLayerWeightCurrent = i * numNodesNextLayer;
    f32 dlda = 0;
    for(i16 j = 0; j < numNodesNextLayer; j++){
        dlda += (weightGradientsNextLayer[j] * weightsNextLayer[nextLayerWeightCurrent + j] / activatedValues[i]); // We use a trick. Basically, if my guts is correct, the gradient of the next layer will be a * next layer stuff(input / activated value) due to it being = d(wa + b)/dw * other stuff, so if I just do a / a * w, then I get w * other stuff, which is what I need!
    }
    f32 dldh = gradTan(hiddenValues[i]) * dlda;
    i32 inputLayerWeightGradient = i * numInputs;
    for(i16 j = 0; j < numInputs; j++){
        weightGradients[inputLayerWeightGradient + j] = inputs[j] * dldh;
    }
    biasGradients[i] = dldh;
}

__global__ void learnLayer(i16 numInputs, f32* weights, f32* biases, f32* weightGradients, f32* biasGradients, i16 numNodes, f32 learningRate){
    i16 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;
    i32 currentWeight = i * numInputs;
    for(i16 j = 0; j < numInputs; j++){
        weights[currentWeight + j] = weights[currentWeight + j] - learningRate * weightGradients[currentWeight + j];
        weightGradients[currentWeight + j] = 0.0f;
    }
    biases[i] = biases[i] - learningRate * biasGradients[i];
    biases[i] = 0.0f;
}

__global__ void initWeights(f32* inputWeights, i16 numNodes, i16 numInputs){
    i16 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;
    i32 currentWeight = i * numInputs;
    for(i16 j = 0; j < numInputs; j++){
        inputWeights[currentWeight + j] = 0.0f;
    }
}
