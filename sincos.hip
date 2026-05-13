#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand/hiprand_kernel.h>
#include <hiprand/hiprand.h>

typedef struct { uint64_t state; } RNG;

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

__device__ float activationTan(float input){
    return tanhf(input);
}

__device__ void inputToHiddenLayer(int i, float input, float* weight, float* biases, float* hiddenValues, float* activatedValues, int n){
    if (i >= n) return;
    hiddenValues[i] = weight[i] * input + biases[i];
    activatedValues[i] = activationTan(hiddenValues[i]);
}

__device__ void hiddenToOutputLayer(int i, float* weight, float* activatedValues, float* output, int n){ // For the best cache coherency, the weights for each node are next-to-eachother.
    if (i >= n) return;
    int j = i * 2;
    float intermeddiateValueSin = activatedValues[i] * weight[j];
    float intermeddiateValueCos = activatedValues[i] * weight[j+1];
    atomicAdd(&output[0], intermeddiateValueSin);
    atomicAdd(&output[1], intermeddiateValueCos);
}

__global__ void addOutputBias(float* output, float* outputBias, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] += outputBias[i];
}

__global__ void computeGradient(float* outputWeights, float* inputWeightGradients, float* inputBiasGradients, float* outputWeightGradients, float* outputBiasGradients, float* hiddenValues, float* activatedValues, int n, float input, float* outputSinCos, float expectedValueSin, float expectedValueCos){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = i * 2;
    float dldoSin = 2.0f * (outputSinCos[0] - expectedValueSin);
    inputWeightGradients[i] += dldoSin * outputWeights[j] * (1 - (tanhf(hiddenValues[i]) * tanhf(hiddenValues[i]))) * input;
    outputWeightGradients[j] = dldoSin * activatedValues[i];
    inputBiasGradients[i] += dldoSin * outputWeights[j] * (1 - (tanhf(hiddenValues[i]) * tanhf(hiddenValues[i])));
    if (i == 0) outputBiasGradients[0] = dldoSin;
    j++;
    float dldoCos = 2.0f * (outputSinCos[1] - expectedValueCos);
    inputWeightGradients[i] += dldoCos * outputWeights[j] * (1 - (tanhf(hiddenValues[i]) * tanhf(hiddenValues[i]))) * input;
    outputWeightGradients[j] = dldoCos * activatedValues[i];
    inputBiasGradients[i] += dldoCos * outputWeights[j] * (1 - (tanhf(hiddenValues[i]) * tanhf(hiddenValues[i])));
    if (i == 0) outputBiasGradients[1] = dldoCos;
}

__global__ void learn(float* inputWeights, float* inputBiases, float* inputWeightGradients, float* inputBiasGradients, float* outputWeights, float* outputBiases, float* outputWeightGradients, float* outputBiasGradients, int n, float learningRate){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = i * 2;
    inputWeights[i] = inputWeights[i] - learningRate * inputWeightGradients[i];
    inputBiases[i] = inputBiases[i] - learningRate * inputBiasGradients[i];
    outputWeights[j] = outputWeights[j] - learningRate * outputWeightGradients[j];
    j++;
    outputWeights[j] = outputWeights[j] - learningRate * outputWeightGradients[j];
    if (i == 0) outputBiases[0] = outputBiases[0] - learningRate * outputBiasGradients[0];
    if (i == 0) outputBiases[1] = outputBiases[1] - learningRate * outputBiasGradients[1];
}

__global__ void initWeights(float* inputWeights, float* inputBiases, float* outputWeights, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = i * 2;
    inputWeights[i] = 2.0f * inputWeights[i] - 1.0f;
    outputWeights[j] = 2.0f * outputWeights[j] - 1.0f;
    j++;
    outputWeights[j] = 2.0f * outputWeights[j] - 1.0f;
    inputBiases[i] = 0.0f;
}

__global__ void runNN(float input, float* inputWeights, float* inputBiases, float* outputWeights, float* outputBias, float* hiddenNeurons, float* activatedNeurons, float* output, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inputToHiddenLayer(i, input, inputWeights, inputBiases, hiddenNeurons, activatedNeurons, n);
    hiddenToOutputLayer(i, outputWeights, activatedNeurons, output, n);
}

int main() {
    float* inputWeights;
    float* outputWeightSinCos;
    float* inputBiases;
    float* outputBiases;
    float* hiddenNeuronValues;
    float* activatedNeuronValues;
    float* inputWeightGradients;
    float* outputWeightGradients;
    float* inputBiasGradients;
    float* outputBiasGradients;
    float* outputSinCos;
    int n = 256;

    int threadsPerBlock = 256;
    int BlockForProgram = ((n - 1)/ 256) + 1;

    hipError_t error;
    error = hipMalloc(&inputWeights, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the input weights: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputWeightSinCos, 2 * n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output weights: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&inputBiases, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the input biases: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputBiases, 2 * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output bias: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&hiddenNeuronValues, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for hidden neurons: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&activatedNeuronValues, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for activated neurons: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&inputWeightGradients, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the input weight gradients: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputWeightGradients, 2 * n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output weight gradients: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&inputBiasGradients, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the input bias gradients: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputBiasGradients, 2 * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output bias gradient: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputSinCos, 2 * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the outputs: \%sn", hipGetErrorName(error));
        abort();
    }

    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    hiprandGenerateUniform(gen, inputWeights, n);
    hiprandGenerateUniform(gen, outputWeightSinCos, 2 * n);
    hiprandDestroyGenerator(gen);

    RNG inputRNG;
    inputRNG.state = 1 * 1973 + 13 * 9277 + 3001 * 2699;
    rngNext(&inputRNG);
    float input = rngNextf(&inputRNG); // in radians.
    input = input * 3.1415 * 2;
    float expectedValueSin = sinf(input);
    float expectedValueCos = cosf(input);
    float learningRate = 0.0015;
    float cpuOutputSin;
    float cpuOutputCos;

    hipLaunchKernelGGL(initWeights, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, inputWeights, inputBiases, outputWeightSinCos, n);
    error = hipMemset(outputBiases, 0, 2 * sizeof(float));
    error = hipMemset(inputWeightGradients, 0, n * sizeof(float));
    error = hipMemset(inputBiasGradients, 0, n * sizeof(float));
    for(uint i = 0; i < 2500; i++){
        error = hipMemset(outputSinCos, 0, 2 * sizeof(float));
        hipLaunchKernelGGL(runNN, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, input, inputWeights, inputBiases, outputWeightSinCos, outputBiases, hiddenNeuronValues, activatedNeuronValues, outputSinCos, n);
        hipLaunchKernelGGL(addOutputBias, dim3(1), dim3(2), 0, 0, outputSinCos, outputBiases, 2);
        hipLaunchKernelGGL(computeGradient, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, outputWeightSinCos, inputWeightGradients, inputBiasGradients, outputWeightGradients, outputBiasGradients, hiddenNeuronValues, activatedNeuronValues, n, input, outputSinCos, expectedValueSin, expectedValueCos);
        hipLaunchKernelGGL(learn, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, inputWeights, inputBiases, inputWeightGradients, inputBiasGradients, outputWeightSinCos, outputBiases, outputWeightGradients, outputBiasGradients, n, learningRate);
        input = rngNextf(&inputRNG); // in radians.
        input = input * 3.1415 * 2;
        expectedValueSin = sinf(input);
        expectedValueCos = cosf(input);
        error = hipMemset(inputWeightGradients, 0, n * sizeof(float));
        error = hipMemset(inputBiasGradients, 0, n * sizeof(float));

    }
    error = hipMemset(outputSinCos, 0, 2 * sizeof(float));
    hipLaunchKernelGGL(runNN, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, input, inputWeights, inputBiases, outputWeightSinCos, outputBiases, hiddenNeuronValues, activatedNeuronValues, outputSinCos, n);
    hipLaunchKernelGGL(addOutputBias, dim3(1), dim3(2), 0, 0, outputSinCos, outputBiases, 2);
    error = hipMemcpy(&cpuOutputSin, &outputSinCos[0], 4, hipMemcpyDeviceToHost);
    error = hipMemcpy(&cpuOutputCos, &outputSinCos[1], 4, hipMemcpyDeviceToHost);
    if(error != hipSuccess){
        fprintf(stderr, "Error couldn't copy output from GPU to CPU memory: \%sn", hipGetErrorName(error));
        abort();
    }
    for(uint i = 0; i < 5; i++){
        input = rngNextf(&inputRNG); // in radians.
        input = input * 3.1415 * 2;
        expectedValueSin = sinf(input);
        expectedValueCos = cosf(input);
        error = hipMemset(outputSinCos, 0, 2 * sizeof(float));
        hipLaunchKernelGGL(runNN, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, input, inputWeights, inputBiases, outputWeightSinCos, outputBiases, hiddenNeuronValues, activatedNeuronValues, outputSinCos, n);
        hipLaunchKernelGGL(addOutputBias, dim3(1), dim3(2), 0, 0, outputSinCos, outputBiases, 2);
        error = hipMemcpy(&cpuOutputSin, &outputSinCos[0], 4, hipMemcpyDeviceToHost);
        error = hipMemcpy(&cpuOutputCos, &outputSinCos[1], 4, hipMemcpyDeviceToHost);
        fprintf(stderr, "Output for sin(%f)| NN: %f | Arithematic: %f | MSE = %f\n", input, cpuOutputSin, expectedValueSin, MSE(cpuOutputSin, expectedValueSin));
        fprintf(stderr, "Output for cos(%f)| NN: %f | Arithematic: %f | MSE = %f\n", input, cpuOutputCos, expectedValueCos, MSE(cpuOutputCos, expectedValueCos));
    }

    error = hipFree(inputWeights);
    error = hipFree(outputWeightSinCos);
    error = hipFree(inputBiases);
    error = hipFree(outputBiases);
    error = hipFree(hiddenNeuronValues);
    error = hipFree(activatedNeuronValues);
    error = hipFree(inputWeightGradients);
    error = hipFree(outputWeightGradients);
    error = hipFree(inputBiasGradients);
    error = hipFree(outputBiasGradients);
    error = hipFree(outputSinCos);

    return 0;
}
