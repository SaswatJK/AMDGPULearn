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

float rngNextf(RNG* rng){
    return (float)rngNext(rng) / (float)0xFFFFFFFF;
}

__device__ float activationTan(float input){
    return tanhf(input);
}

__device__ void inputToHiddenLayer(int i, float input, float* weight, float* biases, float* hiddenValues, float* activatedValues, int n) {
    if (i >= n) return;
    hiddenValues[i] = weight[i] * input + biases[i];
    activatedValues[i] = activationTan(hiddenValues[i]);
}

__device__ void hiddenToOutputLayer(int i, float* weight, float* activatedValues, float* output, int n) {
    if (i >= n) return;
    float intermeddiateValue = activatedValues[i] * weight[i];
    atomicAdd(output, intermeddiateValue);
}

__global__ void addOutputBias(float* output, float* outputBias) {
    *output += *outputBias;
}

__global__ void computeGradient(float* outputWeights, float* inputWeightGradients, float* inputBiasGradients, float* outputWeightGradients, float* outputBiasGradient, float* hiddenValues, float* activatedValues, int n, float input, float* output, float expectedValue){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dldo = 2.0f * (*output - expectedValue);
    inputWeightGradients[i] = dldo * outputWeights[i] * (1 - (tanhf(hiddenValues[i]) * tanhf(hiddenValues[i]))) * input;
    outputWeightGradients[i] = dldo * activatedValues[i];
    inputBiasGradients[i] = dldo * outputWeights[i] * (1 - (tanhf(hiddenValues[i]) * tanhf(hiddenValues[i])));
    if (i == 0) *outputBiasGradient = dldo;
}

__global__ void learn(float* inputWeights, float* inputBiases, float* inputWeightGradients, float* inputBiasGradients, float* outputWeights, float* outputBias, float* outputWeightGradients, float* outputBiasGradient, int n, float learningRate){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inputWeights[i] = inputWeights[i] - learningRate * inputWeightGradients[i];
    inputBiases[i] = inputBiases[i] - learningRate * inputBiasGradients[i];
    outputWeights[i] = outputWeights[i] - learningRate * outputWeightGradients[i];
    if (i == 0) *outputBias = *outputBias - learningRate * *outputBiasGradient;
}

__global__ void initWeights(float* inputWeights, float* inputBiases, float* outputWeights, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inputWeights[i] = 2.0f * inputWeights[i] - 1.0f;
    outputWeights[i] = 2.0f * outputWeights[i] - 1.0f;
    inputBiases[i] = 0.0f;
}

__global__ void runNN(float input, float* inputWeights, float* inputBiases, float* outputWeights, float* outputBias, float* hiddenNeurons, float* activatedNeurons, float* output, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inputToHiddenLayer(i, input, inputWeights, inputBiases, hiddenNeurons, activatedNeurons, n);
    hiddenToOutputLayer(i, outputWeights, activatedNeurons, output, n);
}

float MSE(float value, float realValue){
    return ((value - realValue) * (value - realValue));
}

int main() {
    float* inputWeights;
    float* outputWeights;
    float* inputBiases;
    float* outputBias;
    float* hiddenNeuronValues;
    float* activatedNeuronValues;
    float* inputWeightGradients;
    float* outputWeightGradients;
    float* inputBiasGradients;
    float* outputBiasGradient;
    float* output;
    int n = 256;

    int threadsPerBlock = 256;
    int BlockForProgram = ((n - 1)/ 256) + 1;

    hipError_t error;
    error = hipMalloc(&inputWeights, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the input weights: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputWeights, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output weights: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&inputBiases, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the input biases: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputBias, sizeof(float));
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
    error = hipMalloc(&outputWeightGradients, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output weight gradients: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&inputBiasGradients, n * sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the input bias gradients: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&outputBiasGradient, sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output bias gradient: \%sn", hipGetErrorName(error));
        abort();
    }
    error = hipMalloc(&output, sizeof(float));
    if(error != hipSuccess){
        fprintf(stderr, "Error allocating space for the output: \%sn", hipGetErrorName(error));
        abort();
    }

    hiprandGenerator_t gen;
    hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT);
    hiprandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    hiprandGenerateUniform(gen, inputWeights, n);
    hiprandGenerateUniform(gen, outputWeights, n);
    hiprandDestroyGenerator(gen);

    float learningRate = 0.0015;
    float cpuOutput;

    RNG inputRNG;
    inputRNG.state = 1 * 1973 + 13 * 9277 + 3001 * 2699;
    rngNext(&inputRNG);
    float input = rngNextf(&inputRNG); // in radians.
    hipLaunchKernelGGL(initWeights, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, inputWeights, inputBiases, outputWeights, n);
    error = hipMemset(outputBias, 0, sizeof(float));
    float expectedValue = sinf(input); // sin x
    for(uint i = 0; i < 5128; i++){
        error = hipMemset(output, 0, sizeof(float));
        hipLaunchKernelGGL(runNN, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, input, inputWeights, inputBiases, outputWeights, outputBias, hiddenNeuronValues, activatedNeuronValues, output, n);
        hipLaunchKernelGGL(addOutputBias, dim3(1), dim3(1), 0, 0, output, outputBias);
        hipLaunchKernelGGL(computeGradient, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, outputWeights, inputWeightGradients, inputBiasGradients, outputWeightGradients, outputBiasGradient, hiddenNeuronValues, activatedNeuronValues, n, input, output, expectedValue);
        hipLaunchKernelGGL(learn, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, inputWeights, inputBiases, inputWeightGradients, inputBiasGradients, outputWeights, outputBias, outputWeightGradients, outputBiasGradient, n, learningRate);
        /*
        error = hipMemcpy(&cpuOutput, output, 4, hipMemcpyDeviceToHost);
        if(error != hipSuccess){
            fprintf(stderr, "Erorr couldn't copy output from GPU to CPU memory: \%sn", hipGetErrorName(error));
            abort();
        }
        fprintf(stderr, "For %d iteration, the MSE is:%f\n", i, MSE(cpuOutput, expectedValue));
        */
        input = rngNextf(&inputRNG); // in radians.
        expectedValue = sinf(input); // sin x
    }
    error = hipMemcpy(&cpuOutput, output, 4, hipMemcpyDeviceToHost);
    if(error != hipSuccess){
        fprintf(stderr, "Error couldn't copy output from GPU to CPU memory: \%sn", hipGetErrorName(error));
        abort();
    }
    fprintf(stderr, "Output for sin(%f)| NN: %f | Arithematic: %f", input, cpuOutput, expectedValue);
    fprintf(stderr, "The MSE is:%f\n", MSE(cpuOutput, expectedValue));
    input = rngNextf(&inputRNG);
    expectedValue = sinf(input);
    error = hipMemset(output, 0, sizeof(float));
    hipLaunchKernelGGL(runNN, dim3(BlockForProgram), dim3(threadsPerBlock), 0, 0, input, inputWeights, inputBiases, outputWeights, outputBias, hiddenNeuronValues, activatedNeuronValues, output, n);
    hipLaunchKernelGGL(addOutputBias, dim3(1), dim3(1), 0, 0, output, outputBias);
    error = hipMemcpy(&cpuOutput, output, 4, hipMemcpyDeviceToHost);
    if(error != hipSuccess){
        fprintf(stderr, "Error couldn't copy output from GPU to CPU memory: \%sn", hipGetErrorName(error));
        abort();
    }
    fprintf(stderr, "Output for sin(%f)| NN: %f | Arithematic: %f", input, cpuOutput, expectedValue);
    fprintf(stderr, "The MSE is:%f\n", MSE(cpuOutput, expectedValue));

    error = hipFree(inputWeights);
    error = hipFree(outputWeights);
    error = hipFree(inputBiases);
    error = hipFree(outputBias);
    error = hipFree(hiddenNeuronValues);
    error = hipFree(activatedNeuronValues);
    error = hipFree(inputWeightGradients);
    error = hipFree(outputWeightGradients);
    error = hipFree(inputBiasGradients);
    error = hipFree(outputBiasGradient);
    error = hipFree(output);

    return 0;
}
