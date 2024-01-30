#ifndef __APPLE__
#ifndef __GPU_INFO_TEGRA_H__
#define __GPU_INFO_TEGRA_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum tegraReturn_enum {
  TEGRA_SUCCESS = 0,
  TEGRA_UNSUPPORTED = 1,
  // Other values omitted for now...
} tegraReturn_t;

typedef enum cudaDeviceAttr_enum {
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
} cudaDeviceAttr_t;

typedef void *tegraDevice_t;  // Opaque is sufficient
typedef struct tegraMemory_st {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} tegraMemory_t;

typedef struct tegraDriverVersion {
  int major;
  int minor;
} tegraDriverVersion_t;

typedef enum tegraBrandType_enum
{
    TEGRA_BRAND_UNKNOWN          = 0,
} tegraBrandType_t;

typedef struct tegra_handle {
  void *handle;
  uint16_t verbose;
  tegraReturn_t (*cudaSetDevice)(int device);
  tegraReturn_t (*cudaDeviceReset)(void);
  tegraReturn_t (*cudaMemGetInfo)(size_t *, size_t *);
  tegraReturn_t (*cudaGetDeviceCount)(int *);
  tegraReturn_t (*cudaDeviceGetAttribute)(int* value, cudaDeviceAttr_t attr, int device);
  tegraReturn_t (*cudaDriverGetVersion) (int *driverVersion);
} tegra_handle_t;

typedef struct tegra_init_resp {
  char *err;  // If err is non-null handle is invalid
  tegra_handle_t th;
} tegra_init_resp_t;

typedef struct tegra_compute_capability {
  char *err;
  int major;
  int minor;
} tegra_compute_capability_t;


void tegra_init(char *tegra_lib_path, tegra_init_resp_t *resp);
void tegra_check_vram(tegra_handle_t th, mem_info_t *resp);
void tegra_compute_capability(tegra_handle_t th, tegra_compute_capability_t *tcc);

#endif  // __GPU_INFO_TEGRA_H__
#endif  // __APPLE__
