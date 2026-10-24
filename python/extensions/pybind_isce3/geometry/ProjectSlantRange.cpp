#include "ProjectSlantRange.h"

#include <limits>

#include <optional>
#include <pybind11/stl.h>

#include <isce3/core/Constants.h>
#include <isce3/core/blockProcessing.h>
#include <isce3/io/Raster.h>

// #include <isce3/geocode/GeocodeCov.h>

#include <pybind_isce3/core/Constants.h>

namespace py = pybind11;

using isce3::core::parseDataInterpMethod;
using isce3::geometry::ProjectSlantRange;
using isce3::core::GeocodeMemoryMode;
using isce3::io::Raster;
using isce3::product::RadarGridParameters;

template<typename T>
void addbinding(py::class_<ProjectSlantRange<T>>& pyProjectSlantRange)
{
    pyProjectSlantRange.def(py::init<>())
            .def_property("orbit", nullptr, &ProjectSlantRange<T>::orbit)
            .def_property("doppler", nullptr, &ProjectSlantRange<T>::doppler)
            .def_property("native_doppler", nullptr, &ProjectSlantRange<T>::nativeDoppler)
            .def_property("ellipsoid", nullptr, &ProjectSlantRange<T>::ellipsoid)
            .def_property("threshold_geo2rdr", nullptr,
                          &ProjectSlantRange<T>::thresholdGeo2rdr)
            .def_property("numiter_geo2rdr", nullptr,
                          &ProjectSlantRange<T>::numiterGeo2rdr)
            .def_property("radar_block_margin", nullptr,
                    &ProjectSlantRange<T>::radarBlockMargin)
            .def_property_readonly(
                    "geogrid_start_x", &ProjectSlantRange<T>::geoGridStartX)
            .def_property_readonly(
                    "geogrid_start_y", &ProjectSlantRange<T>::geoGridStartY)
            .def_property_readonly(
                    "geogrid_spacing_x", &ProjectSlantRange<T>::geoGridSpacingX)
            .def_property_readonly(
                    "geogrid_spacing_y", &ProjectSlantRange<T>::geoGridSpacingY)
            .def_property_readonly("geogrid_width", &ProjectSlantRange<T>::geoGridWidth)
            .def_property_readonly("geogrid_length", &ProjectSlantRange<T>::geoGridLength)
            .def("update_geogrid", &ProjectSlantRange<T>::updateGeoGrid,
                    py::arg("radar_grid"), py::arg("dem_raster"))
            .def("geogrid", &ProjectSlantRange<T>::geoGrid, py::arg("x_start"),
                    py::arg("y_start"), py::arg("x_spacing"),
                    py::arg("y_spacing"), py::arg("width"), py::arg("length"),
                    py::arg("epsg"))
            .def("project", &ProjectSlantRange<T>::project, py::arg("radar_grid"),
                    py::arg("input_raster"), py::arg("output_raster"),
                    py::arg("dem_raster"),
                    py::arg("exponent") = 0,
                    py::arg("az_time_correction") = isce3::core::LUT2d<double>(),
                    py::arg("slant_range_correction") = isce3::core::LUT2d<double>(),
                    py::arg("memory_mode") = GeocodeMemoryMode::Auto,
                    py::arg("min_block_size") =
                            isce3::core::DEFAULT_MIN_BLOCK_SIZE,
                    py::arg("max_block_size") =
                            isce3::core::DEFAULT_MAX_BLOCK_SIZE,
                    py::arg("dem_interp_method") =
                            isce3::core::BIQUINTIC_METHOD,
                    R"(
                    Project data from map coordinates to the slant-range geometry.

                    Parameters
                    ----------
                    radar_grid: isce3.product.RadarGridParameters
                        Radar grid
                    input_raster: isce3.io.Raster
                        Input raster. Can be real-
                        or complex-valued. If the input raster is complex-valued
                        and the output is real-valued, it is assumed that the
                        input raster represents single-look complex (SLC) data.
                        In such cases, the complex data
                        is converted to real-valued backscatter, which is proportional to
                        power or intensity. This conversion is performed by taking the
                        square of the SLC magnitudes, and it is applied before geocoding.
                    output_raster: isce3.io.Raster
                        Output raster. Can be real-
                        or complex-valued. This module provides options to perform
                        absolute radiometric calibration, through `abs_cal_factor`
                        and radiometric terrain correction (RTC). To apply these
                        calibrations, it is assumed that complex-valued output rasters
                        represent single-look complex (SLC) data.
                        Both `abs_cal_factor` and RTC normalization values are
                        defined in terms of power or intensity. Therefore, when applied
                        to SLC data, the square roots of these values are used to properly
                        calibrate the magnitude of the complex signal.
                        If the output is a complex interferogram, the user should ensure that
                        RTC correction and absolute radiometric calibration are disabled.
                        This can be done by not providing the `flag_apply_rtc` or
                        `abs_cal_factor parameters`, which default to `false` and `1`,
                        respectively.
                    dem_raster: isce3.io.Raster
                        Input DEM raster
                    exponent: int, optional
                        Exponent to be applied to the input data. The value 0
                        indicates that the exponent is based on the data type
                        of the input raster (1 for real and 2 for complex rasters).
                    az_time_correction: isce3.core.LUT2d
                        Azimuth additive correction, in seconds,
                        as a function of azimuth and range
                    slant_range_correction: isce3.core.LUT2d
                        Slant range additive correction, in meters,
                        as a function of azimuth and range
                    geocode_memory_mode: isce3.core.GeocodeMemoryMode
                        Select memory mode
                    min_block_size: int, optional
                        Minimum block size (per thread)
                    max_block_size: int, optional
                        Maximum block size (per thread)
                    dem_interp_method: isce3.core.DataInterpMethod, optional
                        DEM interpolation method
                    )");
}

template void addbinding(py::class_<ProjectSlantRange<float>>&);
template void addbinding(py::class_<ProjectSlantRange<double>>&);
template void addbinding(py::class_<ProjectSlantRange<std::complex<float>>>&);
template void addbinding(py::class_<ProjectSlantRange<std::complex<double>>>&);
// end of file
