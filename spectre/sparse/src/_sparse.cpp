#include "canonical.h"
#include "convolve.h"
#include "maxclip.h"
#include "python.h"
#include "rolling.h"
#include "stdev.h"
#include <pybind11/pybind11.h>

using i32 = npy_int32;
using i64 = npy_int64;
using f32 = npy_float32;
using f64 = npy_float64;

PYBIND11_MODULE(_sparse, m)
{
  using namespace spectre;

  spectre::import_numpy();

  m.def("is_canonical_coo", is_canonical_coo<i32>);
  m.def("is_canonical_coo", is_canonical_coo<i64>);
  m.def("is_canonical_csr", is_canonical_csr<i32>);
  m.def("is_canonical_csr", is_canonical_csr<i64>);

  m.def("rolling_alloc_csr", rolling_alloc_csr<i32, i32>);
  m.def("rolling_alloc_csr", rolling_alloc_csr<i32, i64>);
  m.def("rolling_alloc_csr", rolling_alloc_csr<i64, i32>);
  m.def("rolling_alloc_csr", rolling_alloc_csr<i64, i64>);

  m.def("rolling_min_csr", rolling_csr<i32, i32, f32, min_kernel>);
  m.def("rolling_min_csr", rolling_csr<i32, i32, f64, min_kernel>);
  m.def("rolling_min_csr", rolling_csr<i32, i64, f32, min_kernel>);
  m.def("rolling_min_csr", rolling_csr<i32, i64, f64, min_kernel>);
  m.def("rolling_min_csr", rolling_csr<i64, i64, f32, min_kernel>);
  m.def("rolling_min_csr", rolling_csr<i64, i64, f64, min_kernel>);

  m.def("rolling_max_csr", rolling_csr<i32, i32, f32, max_kernel>);
  m.def("rolling_max_csr", rolling_csr<i32, i32, f64, max_kernel>);
  m.def("rolling_max_csr", rolling_csr<i32, i64, f32, max_kernel>);
  m.def("rolling_max_csr", rolling_csr<i32, i64, f64, max_kernel>);
  m.def("rolling_max_csr", rolling_csr<i64, i64, f32, max_kernel>);
  m.def("rolling_max_csr", rolling_csr<i64, i64, f64, max_kernel>);

  m.def("rolling_mean_csr", rolling_csr<i32, i32, f32, mean_kernel>);
  m.def("rolling_mean_csr", rolling_csr<i32, i32, f64, mean_kernel>);
  m.def("rolling_mean_csr", rolling_csr<i32, i64, f32, mean_kernel>);
  m.def("rolling_mean_csr", rolling_csr<i32, i64, f64, mean_kernel>);
  m.def("rolling_mean_csr", rolling_csr<i64, i64, f32, mean_kernel>);
  m.def("rolling_mean_csr", rolling_csr<i64, i64, f64, mean_kernel>);

  m.def("rolling_median_csr", rolling_csr<i32, i32, f32, median_kernel>);
  m.def("rolling_median_csr", rolling_csr<i32, i32, f64, median_kernel>);
  m.def("rolling_median_csr", rolling_csr<i32, i64, f32, median_kernel>);
  m.def("rolling_median_csr", rolling_csr<i32, i64, f64, median_kernel>);
  m.def("rolling_median_csr", rolling_csr<i64, i64, f32, median_kernel>);
  m.def("rolling_median_csr", rolling_csr<i64, i64, f64, median_kernel>);

  m.def("std_csr", stdev_csr<i32, f32>);
  m.def("std_csr", stdev_csr<i32, f64>);
  m.def("std_csr", stdev_csr<i64, f32>);
  m.def("std_csr", stdev_csr<i64, f64>);

  m.def("convolve_csr_dv", convolve_csr_dv<i32, i32, f32>);
  m.def("convolve_csr_dv", convolve_csr_dv<i32, i32, f64>);
  m.def("convolve_csr_dv", convolve_csr_dv<i32, i64, f32>);
  m.def("convolve_csr_dv", convolve_csr_dv<i32, i64, f64>);
  m.def("convolve_csr_dv", convolve_csr_dv<i64, i64, f32>);
  m.def("convolve_csr_dv", convolve_csr_dv<i64, i64, f64>);

  m.def("maxclip_csr_spmat_plus_dvec_nonnegative",
        maxclip_csr_spmat_plus_dvec_nonnegative<i32, f32>);
  m.def("maxclip_csr_spmat_plus_dvec_nonnegative",
        maxclip_csr_spmat_plus_dvec_nonnegative<i32, f64>);
  m.def("maxclip_csr_spmat_plus_dvec_nonnegative",
        maxclip_csr_spmat_plus_dvec_nonnegative<i64, f32>);
  m.def("maxclip_csr_spmat_plus_dvec_nonnegative",
        maxclip_csr_spmat_plus_dvec_nonnegative<i64, f64>);
}
