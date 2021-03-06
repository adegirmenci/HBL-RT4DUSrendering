/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <helper_math.h>

const unsigned int nTimeFrames = 1;

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray[nTimeFrames];
cudaArray *d_transferFuncArray;

typedef unsigned char VolumeType;
//typedef unsigned short VolumeType;

cudaChannelFormatDesc channelDesc;

texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, float lowerThresh)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from back to front, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tfar;
    float3 pos = eyeRay.o + eyeRay.d*t;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
        //sample *= 64.0f;    // scale for 10-bit data

        // threshold the sample
        if(sample <= lowerThresh)
            sample = 0.0f;
//        else if(sample >= upperThresh)
//            sample = 255.0f;

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
        col.w *= density;
        
		float depth = norm3df(pos.x, pos.y, pos.z)/(tfar - tnear);
		col.x += 0.2f*depth;
		//col.y += 0.1f*depth;
		col.z += 0.2f*(0.5f-depth);
        // depth cueing
        // R, G, B
        // float depth = ((float)(maxSteps-i))/maxSteps -0.5f;
        // float invDepth = 0.75f - depth;
        // col.x += 0.5f*invDepth;
        // col.y += 0.5f*invDepth;
        // col.z += 1.0f*depth;
        
        // "under" operator for back-to-front blending
        sum = lerp(sum, col, col.w);
        
        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);
        
        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t -= tstep;

        if (t < tnear) break;

        pos -= step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);

    __syncthreads();
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCuda(void **h_volume, cudaExtent volumeSize)
{

	for (unsigned int i = 0; i < nTimeFrames; ++i)
	{
		printf("Loading %d...", i);
		// create 3D array
		channelDesc = cudaCreateChannelDesc<VolumeType>();
		checkCudaErrors(cudaMalloc3DArray(&d_volumeArray[i], &channelDesc, volumeSize));
		printf("Malloc success...", i);
	}
	for (unsigned int i = 0; i < nTimeFrames; ++i)
	{
        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr(h_volume[i], volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
        copyParams.dstArray = d_volumeArray[i];
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
		printf("Copying data...", i);
        checkCudaErrors(cudaMemcpy3D(&copyParams));
		printf("Loaded %d.\n", i);
    }
    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture - we'll need to do a texture bind cyclically
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray[0], channelDesc));

//    //grayscale
    float4 transferFunc[] = {
        {0.0,       0.0,       0.0,       0.0000, },
        {0.2500,    0.2500,    0.2500,    1.0000, },
        {0.5000,    0.5000,    0.5000,    1.0000, },
        {0.7500 ,   0.7500,    0.7500,    1.0000, },
        {1.0000 ,   1.0000,    1.0000 ,   1.0000, },
    };
    
//    //grayscale
//    float4 transferFunc[] = {
//        {0.0,       0.0,       0.0,       0.0000, },
//        {0.2500 ,   0.2500,    0.2500 ,   0.5000, },
//        {0.5000 ,   0.5000,    0.5000,    0.5000, },
//        {0.7500,    0.7500,    0.7500,    0.5000, },
//        {1.0000,    1.0000,    1.0000,    0.5000, },
//    };

    // create transfer function texture
    //float4 transferFunc[] =
    //{
    //    {  0.0, 0.0, 0.0, 0.0, },
    //    {  1.0, 0.0, 0.0, 1.0, },
    //    {  1.0, 0.5, 0.0, 1.0, },
    //    {  1.0, 1.0, 0.0, 1.0, },
    //    {  0.0, 1.0, 0.0, 1.0, },
    //    {  0.0, 1.0, 1.0, 1.0, },
    //    {  0.0, 0.0, 1.0, 1.0, },
    //    {  1.0, 0.0, 1.0, 1.0, },
    //    {  0.0, 0.0, 0.0, 0.0, },
    //};

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void reinitCuda(void **h_volume, cudaExtent volumeSize)
{
    // free old mem
    for (int i = 0; i < nTimeFrames; ++i)
    {
        std::cout << "Trying to free array..." << std::endl;
        checkCudaErrors(cudaFreeArray(d_volumeArray[i]));
        std::cout << "Success." << std::endl;
    }
    // allocate new mem
    for (unsigned int i = 0; i < nTimeFrames; ++i)
    {
        std::cout << "Loading " << i << std::endl;
        //printf("Loading %d...", i);
        // create 3D array
        channelDesc = cudaCreateChannelDesc<VolumeType>();
        checkCudaErrors(cudaMalloc3DArray(&d_volumeArray[i], &channelDesc, volumeSize));
        //printf("Malloc success...");
        std::cout << "Malloc success... " << std::endl;
    }
    for (unsigned int i = 0; i < nTimeFrames; ++i)
    {
        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr(h_volume[i], volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
        copyParams.dstArray = d_volumeArray[i];
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        //printf("Copying data...", i);
        std::cout << "Copying data... " << std::endl;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
        std::cout << "Loaded " << i << std::endl;
        //printf("Loaded %d.\n", i);
    }
    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture - we'll need to do a texture bind cyclically
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray[0], channelDesc));
}

extern "C"
void freeCudaBuffers()
{
    for (int i = 0; i < nTimeFrames; ++i)
    {
        checkCudaErrors(cudaFreeArray(d_volumeArray[i]));
    }
    //checkCudaErrors(cudaFree(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale, float lowerThresh, int currFrame)
{
    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // int nextFrame = currFrame % nTimeFrames;
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray[currFrame], channelDesc));

    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
                                      brightness, transferOffset, transferScale, lowerThresh);
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
