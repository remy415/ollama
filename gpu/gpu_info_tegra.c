#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include <string.h>
#include "gpu_info_tegra.h"

void tegra_init(char *tegra_lib_path, tegra_init_resp_t *resp) {
  tegraReturn_t ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"cudaSetDevice", (void *)&resp->th.cudaSetDevice},
      {"cudaDeviceReset", (void *)&resp->th.cudaDeviceReset},
      {"cudaMemGetInfo", (void *)&resp->th.cudaMemGetInfo},
      {"cudaGetDeviceCount", (void *)&resp->th.cudaGetDeviceCount},
      {"cudaDeviceGetAttribute", (void *)&resp->th.cudaDeviceGetAttribute},
      {"cudaDriverGetVersion", (void *)&resp->th.cudaDriverGetVersion},
      {NULL, NULL},
  };

  resp->th.handle = LOAD_LIBRARY(tegra_lib_path, RTLD_LAZY);
  if (!resp->th.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->th.verbose, "library %s load err: %s\n", tegra_lib_path, msg);
    snprintf(buf, buflen,
            "Unable to load %s library to query for Nvidia Tegra GPUs: %s",
            tegra_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  LOG(resp->th.verbose, "wiring nvidia/tegra management library functions in %s\n", tegra_lib_path);
  
  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    LOG(resp->th.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->th.handle, l[i].s);
    if (!l[i].p) {
      char *msg = LOAD_ERR();
      LOG(resp->th.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->th.handle);
      resp->th.handle = NULL;
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
              msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  ret = (*resp->th.cudaSetDevice)(0);
  if (ret != TEGRA_SUCCESS) {
    LOG(resp->th.verbose, "cudaSetDevice(0) err: %d\n", ret);
    UNLOAD_LIBRARY(resp->th.handle);
    resp->th.handle = NULL;
    snprintf(buf, buflen, "cuda runtime api init failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  int version = 0;
  tegraDriverVersion_t driverVersion;
  driverVersion.major = 0;
  driverVersion.minor = 0;

  // Report driver version if we're in verbose mode, ignore errors
  ret = (*resp->th.cudaDriverGetVersion)(&version);
  if (ret != TEGRA_SUCCESS) {
    LOG(resp->th.verbose, "cudaDriverGetVersion failed: %d\n", ret);
  } else {
    driverVersion.major = version / 1000;
    driverVersion.minor = (version - (driverVersion.major * 1000)) / 10;
    LOG(resp->th.verbose, "CUDA driver version: %d-%d\n", driverVersion.major, driverVersion.minor);
  }
}

void tegra_check_vram(tegra_handle_t th, mem_info_t *resp) {
  resp->err = NULL;
  tegraMemory_t memInfo = {0};
  tegraReturn_t ret;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  if (th.handle == NULL) {
    resp->err = strdup("tegra handle isn't initialized");
    return;
  }

  int deviceCount;
  ret = (*th.cudaGetDeviceCount)(&deviceCount);
  if (ret != TEGRA_SUCCESS) {
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    resp->err = strdup(buf);
    return;
  } else {
    resp->count = (unsigned int)deviceCount;
  }

  resp->total = 0;
  resp->free = 0;

  ret = (*th.cudaMemGetInfo)(&resp->free, &resp->total);
  if (ret != TEGRA_SUCCESS) {
    snprintf(buf, buflen, "tegra device memory info lookup failure %d", ret);
    resp->err = strdup(buf);
    return;
  }

  if (th.verbose) {
    tegraBrandType_t brand = 0;
    // When in verbose mode, report more information about
    // the card we discover, but don't fail on error
    ret = TEGRA_UNSUPPORTED;
    if (ret != TEGRA_SUCCESS) {
      LOG(th.verbose, "nvmlDeviceGetName unsupported on Tegra: %d\n", ret);
    } else {
      LOG(th.verbose, "[%d] CUDA device name: %s\n", i, buf);
    }
    if (ret != TEGRA_SUCCESS) {
      LOG(th.verbose, "nvmlDeviceGetBoardPartNumber unsupported on Tegra: %d\n", ret);
    } else {
      LOG(th.verbose, "[%d] CUDA part number: %s\n", i, buf);
    }
    if (ret != TEGRA_SUCCESS) {
      LOG(th.verbose, "nvmlDeviceGetSerial unsupported on Tegra: %d\n", ret);
    } else {
      LOG(th.verbose, "[%d] CUDA S/N: %s\n", i, buf);
    }
    if (ret != TEGRA_SUCCESS) {
      LOG(th.verbose, "nvmlDeviceGetVbiosVersion unsupported on Tegra: %d\n", ret);
    } else {
      LOG(th.verbose, "[%d] CUDA vbios version: %s\n", i, buf);
    }
    if (ret != TEGRA_SUCCESS) {
      LOG(th.verbose, "nvmlDeviceGetBrand unsupported on Tegra: %d\n", ret);
    } else {
      LOG(th.verbose, "[%d] CUDA brand: %d\n", i, brand);
    }
  }

  LOG(th.verbose, "[%d] CUDA totalMem %ld\n", i, resp->total);
  LOG(th.verbose, "[%d] CUDA freeMem %ld\n", i, resp->free);

}

void tegra_compute_capability(tegra_handle_t th, tegra_compute_capability_t *resp) {
  resp->err = NULL;
  resp->major = 0;
  resp->minor = 0;
  tegraReturn_t ret;
  const int buflen = 256;
  char buf[buflen + 1];

  if (th.handle == NULL) {
    resp->err = strdup("tegra handle not initialized");
    return;
  }

  int devices;
  ret = (*th.cudaGetDeviceCount)(&devices);
  if (ret != TEGRA_SUCCESS) {
    snprintf(buf, buflen, "unable to get tegra device count: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  int devId = 0; // Tegra device id is always 0
  int major = 0;
  int minor = 0;

  ret = (*th.cudaDeviceGetAttribute)(&major, cudaDevAttrComputeCapabilityMajor, devId);
  if (ret != TEGRA_SUCCESS) {
    snprintf(buf, buflen, "device compute capability lookup failure %d: %d", devId, ret);
    resp->err = strdup(buf);
    return;
  }
  ret = (*th.cudaDeviceGetAttribute)(&minor, cudaDevAttrComputeCapabilityMinor, devId);
  if (ret != TEGRA_SUCCESS) {
    snprintf(buf, buflen, "device compute capability lookup failure %d: %d", devId, ret);
    resp->err = strdup(buf);
    return;
  }
 
  LOG(th.verbose, "Tegra Compute Capability version: %d.%d\n", major, minor);
  
  // Report the lowest major.minor we detect as that limits our compatibility
  if (resp->major == 0 || resp->major > major ) {
    resp->major = major;
    resp->minor = minor;
  } else if ( resp->major == major && resp->minor > minor ) {
    resp->minor = minor;
  }
}

#endif  // __APPLE__
