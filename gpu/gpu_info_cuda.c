#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include <string.h>

#include "gpu_info_cuda.h"

void cuda_init(char *cuda_lib_path, cuda_init_resp_t *resp) {
  resp->err = NULL;
  resp->ch.lib_t = LIBUNKNOWN;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;
  int version;

  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"nvmlInit_v2", (void *)&resp->ch.nvmlInit_v2},
      {"nvmlShutdown", (void *)&resp->ch.nvmlShutdown},
      {"nvmlDeviceGetHandleByIndex", (void *)&resp->ch.nvmlDeviceGetHandleByIndex},
      {"nvmlDeviceGetMemoryInfo", (void *)&resp->ch.nvmlDeviceGetMemoryInfo},
      {"nvmlDeviceGetCount_v2", (void *)&resp->ch.nvmlDeviceGetCount_v2},
      {"nvmlDeviceGetCudaComputeCapability", (void *)&resp->ch.nvmlDeviceGetCudaComputeCapability},
      {"nvmlSystemGetDriverVersion", (void *)&resp->ch.nvmlSystemGetDriverVersion},
      {"nvmlDeviceGetName", (void *)&resp->ch.nvmlDeviceGetName},
      {"nvmlDeviceGetSerial", (void *)&resp->ch.nvmlDeviceGetSerial},
      {"nvmlDeviceGetVbiosVersion", (void *)&resp->ch.nvmlDeviceGetVbiosVersion},
      {"nvmlDeviceGetBoardPartNumber", (void *)&resp->ch.nvmlDeviceGetBoardPartNumber},
      {"nvmlDeviceGetBrand", (void *)&resp->ch.nvmlDeviceGetBrand},
      {"cudaSetDevice", (void *)&resp->ch.cudaSetDevice},
      {"cudaDeviceReset", (void *)&resp->ch.cudaDeviceReset},
      {"cudaMemGetInfo", (void *)&resp->ch.cudaMemGetInfo},
      {"cudaGetDeviceCount", (void *)&resp->ch.cudaGetDeviceCount},
      {"cudaDeviceGetAttribute", (void *)&resp->ch.cudaDeviceGetAttribute},
      {"cudaDriverGetVersion", (void *)&resp->ch.cudaDriverGetVersion},
      {NULL, NULL},
  };

  resp->ch.handle = LOAD_LIBRARY(cuda_lib_path, RTLD_LAZY);
  if (!resp->ch.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", cuda_lib_path, msg);
    snprintf(buf, buflen,
             "Unable to load %s library to query for Nvidia GPUs: %s",
             cuda_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  LOG(resp->ch.verbose, "wiring nvidia management library functions in %s\n", cuda_lib_path);
  
  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    LOG(resp->ch.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->ch.handle, l[i].s);
    if (!l[i].p) {
      resp->ch.handle = NULL;
      char *msg = LOAD_ERR();
      LOG(resp->ch.verbose, "dlerr: %s \n", msg);
      UNLOAD_LIBRARY(resp->ch.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
               msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }
  
  cudartReturn_t cudart_ret;
  nvmlReturn_t nvml_ret;
  
  // Trying libcudart.so first, fallback to nvml
  if (resp->ch.cudaSetDevice != NULL) {
    cudart_ret = (*resp->ch.cudaSetDevice)(0);
    if (cudart_ret != CUDART_SUCCESS) {
      LOG(resp->ch.verbose, "cudaSetDevice err: %d\n", cudart_ret);
      UNLOAD_LIBRARY(resp->ch.handle);
      resp->ch.handle = NULL;
      snprintf(buf, buflen, "cudart vram init failure: %d", cudart_ret);
      resp->err = strdup(buf);
      return;
    }
    resp->ch.lib_t = LIBCUDART;
  } else if (resp->ch.nvmlInit_v2 != NULL) {
    nvml_ret = (*resp->ch.nvmlInit_v2)();
    if (nvml_ret != NVML_SUCCESS) {
      LOG(resp->ch.verbose, "nvmlInit_v2 err: %d\n", nvml_ret);
      UNLOAD_LIBRARY(resp->ch.handle);
      resp->ch.handle = NULL;
      snprintf(buf, buflen, "nvml vram init failure: nvml error %d", nvml_ret);
      resp->err = strdup(buf);
      return;
    }
    resp->ch.lib_t = LIBNVIDIAML;
  }

  cudartDriverVersion_t driverVersion;
  switch (resp->ch.lib_t) {
    case LIBCUDART:
      // Report driver version if we're in verbose mode, ignore errors
      version = 0;
      driverVersion.major = 0;
      driverVersion.minor = 0;
      cudart_ret = (*resp->ch.cudaDriverGetVersion)(&version);
      if (cudart_ret != CUDART_SUCCESS) {
        LOG(resp->ch.verbose, "cudaDriverGetVersion failed: %d\n", cudart_ret);
      }
      driverVersion.major = version / 1000;
      driverVersion.minor = (version - (driverVersion.major * 1000)) / 10;
      LOG(resp->ch.verbose, "CUDA driver version: %d-%d\n", driverVersion.major, driverVersion.minor);
      break;
    case LIBNVIDIAML:
      nvml_ret = (*resp->ch.nvmlSystemGetDriverVersion)(buf, buflen);
      if (nvml_ret != NVML_SUCCESS) {
        LOG(resp->ch.verbose, "nvmlSystemGetDriverVersion failed: %d\n", nvml_ret);
      } else {
        LOG(resp->ch.verbose, "CUDA driver version: %s\n", buf);
      }
    default:
      resp->ch.handle = NULL;
      LOG(resp->ch.verbose, "unknown cuda initialization error\n");
      UNLOAD_LIBRARY(resp->ch.handle);
      snprintf(buf, buflen, "unknown error: dlsym succeded but function pointers are unassigned");
      resp->err = strdup(buf);
      return;
  }
}

void cuda_check_vram(cuda_handle_t h, mem_info_t *resp) {
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;
  cudartReturn_t cudart_ret;
  nvmlDevice_t device;
  nvmlReturn_t nvml_ret;

  if (h.handle == NULL) {
    resp->err = strdup("cuda and nvml handle isn't initialized");
    return;
  }

  switch (h.lib_t) {
    case LIBCUDART:
      cudartMemory_t cudart_memInfo = {0};
      cudart_ret = (*h.cudaGetDeviceCount)(&resp->count);
      if (cudart_ret != CUDART_SUCCESS) {
        snprintf(buf, buflen, "unable to get device count: %d", cudart_ret);
        resp->err = strdup(buf);
        return;
      }

      resp->total = 0;
      resp->free = 0;
      for (i = 0; i < resp->count; i++) {
        
        cudart_ret = (*h.cudaSetDevice)(i);
        if (cudart_ret != CUDART_SUCCESS) {
          snprintf(buf, buflen, "unable to get device handle %d: %d", i, cudart_ret);
          resp->err = strdup(buf);
          return;
        }
        cudart_ret = (*h.cudaMemGetInfo)(&cudart_memInfo.free, &cudart_memInfo.total);
        if (cudart_ret != CUDART_SUCCESS) {
          snprintf(buf, buflen, "device memory info lookup failure %d: %d", i, cudart_ret);
          resp->err = strdup(buf);
          return;
        }
        LOG(h.verbose, "[%d] CUDA totalMem %ld\n", i, cudart_memInfo.total);
        LOG(h.verbose, "[%d] CUDA usedMem %ld\n", i, cudart_memInfo.free);

        resp->total += cudart_memInfo.total;
        resp->free += cudart_memInfo.free;
      }
      break;
    
    case LIBNVIDIAML:
      nvmlMemory_t nvml_memInfo = {0};
      nvml_ret = (*h.nvmlDeviceGetCount_v2)(&resp->count);
      if (nvml_ret != NVML_SUCCESS) {
        snprintf(buf, buflen, "unable to get device count: %d", nvml_ret);
        resp->err = strdup(buf);
        return;
      }

      resp->total = 0;
      resp->free = 0;

      for (i = 0; i < resp->count; i++) {

        nvml_ret = (*h.nvmlDeviceGetHandleByIndex)(i, &device);
        if (nvml_ret != NVML_SUCCESS) {
          snprintf(buf, buflen, "unable to get device handle %d: %d", i, nvml_ret);
          resp->err = strdup(buf);
          return;
        }

        nvml_ret = (*h.nvmlDeviceGetMemoryInfo)(device, &nvml_memInfo);
        if (nvml_ret != NVML_SUCCESS) {
          snprintf(buf, buflen, "device memory info lookup failure %d: %d", i, nvml_ret);
          resp->err = strdup(buf);
          return;
        }
        
        if (h.verbose) {
          nvmlBrandType_t brand = 0;
          // When in verbose mode, report more information about
          // the card we discover, but don't fail on error
          nvml_ret = (*h.nvmlDeviceGetName)(device, buf, buflen);
          if (nvml_ret != NVML_SUCCESS) {
            LOG(h.verbose, "nvmlDeviceGetName failed: %d\n", nvml_ret);
          } else {
            LOG(h.verbose, "[%d] CUDA device name: %s\n", i, buf);
          }
          nvml_ret = (*h.nvmlDeviceGetBoardPartNumber)(device, buf, buflen);
          if (nvml_ret != NVML_SUCCESS) {
            LOG(h.verbose, "nvmlDeviceGetBoardPartNumber failed: %d\n", nvml_ret);
          } else {
            LOG(h.verbose, "[%d] CUDA part number: %s\n", i, buf);
          }
          nvml_ret = (*h.nvmlDeviceGetSerial)(device, buf, buflen);
          if (nvml_ret != NVML_SUCCESS) {
            LOG(h.verbose, "nvmlDeviceGetSerial failed: %d\n", nvml_ret);
          } else {
            LOG(h.verbose, "[%d] CUDA S/N: %s\n", i, buf);
          }
          nvml_ret = (*h.nvmlDeviceGetVbiosVersion)(device, buf, buflen);
          if (nvml_ret != NVML_SUCCESS) {
            LOG(h.verbose, "nvmlDeviceGetVbiosVersion failed: %d\n", nvml_ret);
          } else {
            LOG(h.verbose, "[%d] CUDA vbios version: %s\n", i, buf);
          }
          nvml_ret = (*h.nvmlDeviceGetBrand)(device, &brand);
          if (nvml_ret != NVML_SUCCESS) {
            LOG(h.verbose, "nvmlDeviceGetBrand failed: %d\n", nvml_ret);
          } else {
            LOG(h.verbose, "[%d] CUDA brand: %d\n", i, brand);
          }
        }

        LOG(h.verbose, "[%d] CUDA totalMem %ld\n", i, nvml_memInfo.total);
        LOG(h.verbose, "[%d] CUDA usedMem %ld\n", i, nvml_memInfo.free);

        resp->total += nvml_memInfo.total;
        resp->free += nvml_memInfo.free;
      }
      break;
    
    default:
      LOG(h.verbose, "unknown library loaded: %d \n", h.lib_t);
      snprintf(buf, buflen, "error detecting loaded library: %d", h.lib_t);
      resp->err = strdup(buf);
      return;
  }
}

void cuda_compute_capability(cuda_handle_t h, cuda_compute_capability_t *resp) {
  resp->err = NULL;
  resp->major = 0;
  resp->minor = 0;
  cudartReturn_t cudart_ret;
  nvmlDevice_t device;
  nvmlReturn_t nvml_ret;
  int major = 0;
  int minor = 0;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  if (h.handle == NULL) {
    resp->err = strdup("cuda handle not initialized");
    return;
  }

  switch (h.lib_t) {
    case LIBNVIDIAML:
      unsigned int devices;
      nvml_ret = (*h.nvmlDeviceGetCount_v2)(&devices);
      if (nvml_ret != NVML_SUCCESS) {
        snprintf(buf, buflen, "unable to get device count: %d", nvml_ret);
        resp->err = strdup(buf);
        return;
      }

      for (i = 0; i < devices; i++) {
        nvml_ret = (*h.nvmlDeviceGetHandleByIndex)(i, &device);
        if (nvml_ret != NVML_SUCCESS) {
          snprintf(buf, buflen, "unable to get device handle %d: %d", i, nvml_ret);
          resp->err = strdup(buf);
          return;
        }

        nvml_ret = (*h.nvmlDeviceGetCudaComputeCapability)(device, &major, &minor);
        if (nvml_ret != NVML_SUCCESS) {
          snprintf(buf, buflen, "device compute capability lookup failure %d: %d", i, nvml_ret);
          resp->err = strdup(buf);
          return;
        }
        // Report the lowest major.minor we detect as that limits our compatibility
        if (resp->major == 0 || resp->major > major ) {
          resp->major = major;
          resp->minor = minor;
        } else if ( resp->major == major && resp->minor > minor ) {
          resp->minor = minor;
        }
      }
      break;
    case LIBCUDART:
      int devices;
      cudart_ret = (*h.cudaGetDeviceCount)(&devices);
      if (cudart_ret != CUDART_SUCCESS) {
        snprintf(buf, buflen, "unable to get tegra device count: %d", cudart_ret);
        resp->err = strdup(buf);
        return;
      }

      for (i = 0; i < devices; i++) {
        cudart_ret = (*h.cudaSetDevice)(i);
        if (cudart_ret != CUDART_SUCCESS) {
          snprintf(buf, buflen, "unable to get device handle %d: %d", i, cudart_ret);
          resp->err = strdup(buf);
          return;
        }

        cudart_ret = (*h.cudaDeviceGetAttribute)(&major, cudaDevAttrComputeCapabilityMajor, i);
        if (cudart_ret != CUDART_SUCCESS) {
          snprintf(buf, buflen, "device compute capability lookup failure %d: %d", i, cudart_ret);
          resp->err = strdup(buf);
          return;
        }
        cudart_ret = (*h.cudaDeviceGetAttribute)(&minor, cudaDevAttrComputeCapabilityMinor, i);
        if (cudart_ret != CUDART_SUCCESS) {
          snprintf(buf, buflen, "device compute capability lookup failure %d: %d", i, cudart_ret);
          resp->err = strdup(buf);
          return;
        }
        
        // Report the lowest major.minor we detect as that limits our compatibility
        if (resp->major == 0 || resp->major > major ) {
          resp->major = major;
          resp->minor = minor;
        } else if ( resp->major == major && resp->minor > minor ) {
          resp->minor = minor;
        }
      }

      break;
    default:
      LOG(h.verbose, "unknown library loaded: %d \n", h.lib_t);
      snprintf(buf, buflen, "error detecting loaded library: %d", h.lib_t);
      resp->err = strdup(buf);
      return;
  }
}

#endif  // __APPLE__