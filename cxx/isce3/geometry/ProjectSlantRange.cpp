#include "ProjectSlantRange.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cpl_virtualmem.h>
#include <limits>
#include <ios>

#include <isce3/core/Basis.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Projections.h>
#include <isce3/core/TypeTraits.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Utilities.h>
#include <isce3/geometry/RTC.h>
#include <isce3/geometry/loadDem.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/geometry/geometry.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/signal/Looks.h>
#include <isce3/signal/signalUtils.h>

#include <isce3/geocode/GeocodeHelpers.h>
#include <isce3/math/complexOperations.h>

using isce3::core::OrbitInterpBorderMode;
using isce3::core::Vec3;
using isce3::core::GeocodeMemoryMode;

namespace isce3 { namespace geometry {

template<class T>
void ProjectSlantRange<T>::updateGeoGrid(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& dem_raster)
{

    pyre::journal::info_t info("isce.geometry.ProjectSlantRange.updateGeoGrid");

    if (_epsgOut == 0)
        _epsgOut = dem_raster.getEPSG();

    if (std::isnan(_geoGridSpacingX))
        _geoGridSpacingX = dem_raster.dx();

    if (std::isnan(_geoGridSpacingY))
        _geoGridSpacingY = dem_raster.dy();

    if (std::isnan(_geoGridStartX) || std::isnan(_geoGridStartY) ||
        _geoGridLength <= 0 || _geoGridWidth <= 0) {
        std::unique_ptr<isce3::core::ProjectionBase> proj(
                isce3::core::createProj(_epsgOut));
        isce3::geometry::BoundingBox bbox =
                isce3::geometry::getGeoBoundingBoxHeightSearch(
                        radar_grid, _orbit, proj.get(), _doppler);
        _geoGridStartX = bbox.MinX;
        if (_geoGridSpacingY < 0)
            _geoGridStartY = bbox.MaxY;
        else
            _geoGridStartY = bbox.MinY;

        _geoGridWidth = (bbox.MaxX - bbox.MinX) / _geoGridSpacingX;
        _geoGridLength = std::abs((bbox.MaxY - bbox.MinY) / _geoGridSpacingY);
    }
}

template<class T>
void ProjectSlantRange<T>::geoGrid(double geoGridStartX, double geoGridStartY,
                         double geoGridSpacingX, double geoGridSpacingY,
                         int width, int length, int epsgcode)
{

    // the starting coordinate of the output geocoded grid in X direction.
    _geoGridStartX = geoGridStartX;

    // the starting coordinate of the output geocoded grid in Y direction.
    _geoGridStartY = geoGridStartY;

    // spacing of the output geocoded grid in X
    _geoGridSpacingX = geoGridSpacingX;

    // spacing of the output geocoded grid in Y
    _geoGridSpacingY = geoGridSpacingY;

    // number of lines (rows) in the geocoded grid (Y direction)
    _geoGridLength = length;

    // number of columns in the geocoded grid (Y direction)
    _geoGridWidth = width;

    // Save the EPSG code
    _epsgOut = epsgcode;
}



template<class T>
int ProjectSlantRange<T>::_geo2rdr(const isce3::product::RadarGridParameters& radar_grid,
        double x, double y, double& azimuthTime, double& slantRange,
        isce3::geometry::DEMInterpolator& demInterp,
        isce3::core::ProjectionBase* proj, float& dem_value)
{
    // coordinate in the output projection system
    const Vec3 xyz {x, y, 0.0};

    // transform the xyz in the output projection system to llh
    Vec3 llh = proj->inverse(xyz);

    // interpolate the height from the DEM for this pixel
    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

    // assign interpolated DEM value to returning variable
    dem_value = llh[2];

    // Perform geo->rdr iterations
    int converged = isce3::geometry::geo2rdr(llh, _ellipsoid, _orbit, _doppler,
            azimuthTime, slantRange, radar_grid.wavelength(),
            radar_grid.lookSide(), _threshold, _numiter, 1.0e-8);

    // Check convergence
    if (converged == 0) {
        azimuthTime = std::numeric_limits<double>::quiet_NaN();
        slantRange = std::numeric_limits<double>::quiet_NaN();
    }
    return converged;
}

template<class T, class T_out>
void _getInputDataBlock(
        std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>>& inputData,
        isce3::io::Raster& input_raster, size_t xidx, size_t yidx,
        size_t size_x, size_t size_y,
        GeocodeMemoryMode geocode_memory_mode, const long long min_block_size,
        const long long max_block_size, pyre::journal::info_t& info)
{
    int nbands = input_raster.numBands();
    inputData.reserve(nbands);
    bool flag_parallel_radargrid_read = geocode_memory_mode ==
            GeocodeMemoryMode::BlocksGeogridAndRadarGrid;

    int radargrid_nblocks, radar_block_length;
    const int n_threads_per_radargrid_block = 1;

    for (int band = 0; band < nbands; ++band) {
        if (!flag_parallel_radargrid_read) {
            info << "reading input raster band: " << band + 1
                 << pyre::journal::endl;
        }

        inputData.emplace_back(
                std::make_unique<isce3::core::Matrix<T_out>>(size_y, size_x));

        if (std::is_same<T, T_out>::value &&
                !flag_parallel_radargrid_read) {
            /*
            Enter here if:
                1. No type convertion is required (input and output have same
                   types);
                2. Not parallel (which allows messages to be printed to stdout).
            */

            isce3::core::getBlockProcessingParametersY(size_y, size_x, 1,
                    sizeof(T), nullptr, &radar_block_length, &radargrid_nblocks,
                    min_block_size, max_block_size, n_threads_per_radargrid_block);

            for (size_t block = 0; block < (size_t) radargrid_nblocks;
                 ++block) {

                int this_radar_block_length = radar_block_length;
                if ((block + 1) * radar_block_length > size_y) {
                    this_radar_block_length = size_y % radar_block_length;
                }
                if (radargrid_nblocks > 1) {
                    std::cout << "reading band " << band + 1 << " progress: "
                              << static_cast<int>(
                                         (100.0 * block) / radargrid_nblocks)
                              << "% \r";
                    std::cout.flush();
                }
                auto ptr = inputData[band]->data();
                input_raster.getBlock(ptr +
                                      block * radar_block_length * size_x,
                                      xidx, block * radar_block_length + yidx, size_x,
                                      this_radar_block_length, band + 1);
            }

            if (radargrid_nblocks > 1) {
                std::cout << "reading band " << band + 1
                          << " progress: 100%" << std::endl;
            }
        }
        else if (std::is_same<T, T_out>::value) {
            /*
            Enter here if:
                1. No type convertion is required (input and output have same
                   types);
                2. Is parallel (which does not allow messages to be printed to
                   stdout).
            */
            _Pragma("omp critical")
            {
            input_raster.getBlock(inputData[band]->data(), xidx, yidx, size_x,
                                  size_y, band + 1);
            }
        }
        else {
            /*
            Enter here if:
                1. Type convertion is required (input and output have different
                   types).
            */
            isce3::core::Matrix<T> radar_data_in(size_y, size_x);
            if (flag_parallel_radargrid_read) {
                _Pragma("omp critical")
                {
                input_raster.getBlock(radar_data_in.data(), xidx, yidx, size_x,
                                       size_y, band + 1);
                }
            } else {
                input_raster.getBlock(radar_data_in.data(), xidx, yidx, size_x,
                        size_y, band + 1);
            }

            /*
            Iteratively converts input pixel (ptr_1) to output pixel (ptr_2).
            In this case, the input type T (complex) is different than T_out
            (real).
            The conversion from complex (e.g. SLC) to real (SAR backscatter)
            in the context of a covariance matrix (diagonal elements) is done by
            squaring the modulus of the complex input. This operation is handled
            by _convertToOutputType
            */
            auto ptr_1 = radar_data_in.data();
            auto ptr_2 = inputData[band]->data();
            for (size_t k = 0; k < size_y * size_x; ++k) {
                isce3::geocode::_convertToOutputType(*ptr_1++, *ptr_2++);
            }
        }
    }
}

static int _geo2rdrWrapper(const Vec3& inputLLH, const Ellipsoid& ellipsoid,
        const Orbit& orbit, const LUT2d<double>& doppler, double& aztime,
        double& slantRange, double wavelength, LookSide side,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        double threshold, int maxIter, double deltaRange,
        bool flag_edge = true)
{
    int flag_converged;
    for (int i = 0; i <= static_cast<int>(flag_edge); ++i) {
        /*
          Run geo2rdr twice for border edge pixels. This is
          required because initial guesses (a11 and r11)
          are not as good for edge elements. Without it,
          the edge solutions are slightly different than the
          corresponding solutions from single-block processing.
       */
        flag_converged = isce3::geometry::geo2rdr(inputLLH, ellipsoid, orbit,
                doppler, aztime, slantRange, wavelength, side, threshold,
                maxIter, deltaRange);

        if (!flag_converged) {
            return flag_converged;
        }
    }
    // apply timing corrections
    if (az_time_correction.contains(aztime, slantRange)) {
        const auto aztimeCor = az_time_correction.eval(aztime, slantRange);
        aztime += aztimeCor;
    }

    if (slant_range_correction.contains(aztime, slantRange)) {
        const auto srangeCor = slant_range_correction.eval(aztime, slantRange);
        slantRange += srangeCor;
    }

    return flag_converged;
}


/**
* This function fills up a GCOV raster block with NaNs if the block is
# invalid (e.g., outside of the DEM coverage).
*
* @param[in]  block_x            Number of the current block in the X-direction
* @param[in]  block_size_x       Processing block size in the X direction
* @param[in]  block_y            Number of the current block in the Y-direction
* @param[in]  block_size_y       Processing block size in the Y direction
* @param[in]  this_block_size_x  Size of the current block in the X direction
* @param[in]  this_block_size_y  Size of the current block in the Y direction
* @param[out] output_raster      Output raster
*
*/
template<class T>
inline void _fillGcovBlocksWithNans(
    int block_x, int block_size_x, int block_y,
    int block_size_y, int this_block_size_x, int this_block_size_y,
    isce3::io::Raster* output_raster)
{

    // The output raster may be optional (e.g., off-diagonal raster). If
    // it is `nullptr`, return.
    if (output_raster == nullptr) {
        return;
    }

    // declare matrix that will hold the NaNs
    isce3::core::Matrix<T> data_block(this_block_size_y, this_block_size_x);

    // declare variable to hold NaN values according to the templateT,
    // i.e. real (NaN) or complex (NaN, NaN)
    using T_real = typename isce3::real<T>::type;
    T nan_t = 0;
    nan_t *= std::numeric_limits<T_real>::quiet_NaN();

    // fill matrix with NaN
    data_block.fill(nan_t);

    const int nbands = output_raster->numBands();
    for (int band = 0; band < nbands; ++band) {
        _Pragma("omp critical")
        {
            // set block with the matrix `data_block` that
            // is filled with NaNs
            output_raster->setBlock(
                data_block.data(), block_x * block_size_x,
                block_y * block_size_y, this_block_size_x,
                this_block_size_y, band + 1);
        }
    }
}


template<class T>
bool ProjectSlantRange<T>::_checkLoadEntireRslcCorners(const double y0, const double x0,
        const double yf, const double xf,
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::core::ProjectionBase* proj,
        const std::function<Vec3(double, double,
                const isce3::geometry::DEMInterpolator&,
                isce3::core::ProjectionBase*)>& getDemCoords,
        isce3::geometry::DEMInterpolator& dem_interp, int margin_pixels)
{
    /*
     Check if a geogrid bounding box (y0, x0, yf, xf) fully
     covers the RSLC (represented by the radar_grid).
     */

    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    const double r0 = radar_grid.startingRange() - 0.5 * dr;

    double a_min = std::numeric_limits<double>::quiet_NaN();
    double r_min = std::numeric_limits<double>::quiet_NaN();
    double a_max = std::numeric_limits<double>::quiet_NaN();
    double r_max = std::numeric_limits<double>::quiet_NaN();

    std::vector<std::pair<float, float>> vertices_positions = {
            std::make_pair(y0, x0), std::make_pair(y0, xf),
            std::make_pair(yf, x0), std::make_pair(yf, xf)};

    for (auto [dem_y, dem_x] : vertices_positions) {

        double az_time = radar_grid.sensingMid();
        double range_distance = radar_grid.midRange();

        // Convert DEM coordinates (`dem_x` and `dem_y`) from _epsgOut to DEM
        // EPSG coordinates x and y, interpolate height (z), and return:
        // dem_pos_vect = {x, y, z}
        Vec3 dem_pos_vect = getDemCoords(dem_x, dem_y, dem_interp, proj);

        const int converged = isce3::geometry::geo2rdr(
                dem_interp.proj()->inverse(dem_pos_vect), _ellipsoid, _orbit,
                _doppler, az_time, range_distance, radar_grid.wavelength(),
                radar_grid.lookSide(), _threshold, _numiter, 1.0e-8);
        // if it didn't converge, return false
        if (!converged) {
            return false;
        }

        // Convert az. time and range distance to pixel indexes
        double idx_a = (az_time - start) / pixazm;
        double idx_r = (range_distance - r0) / dr;

        /*
        If there is at least one point inside the radar grid,
        do not load entire RSLC
        */
        if (idx_a > margin_pixels &&
                idx_a < radar_grid.length() - 1 - margin_pixels &&
                idx_r > margin_pixels &&
                idx_r < radar_grid.width() - 1 - margin_pixels) {
            return false;
        }

        if (std::isnan(a_min) || idx_a < a_min)
            a_min = idx_a;
        if (std::isnan(a_max) || idx_a > a_max)
            a_max = idx_a;
        if (std::isnan(r_min) || idx_r < r_min)
            r_min = idx_r;
        if (std::isnan(r_max) || idx_r > r_max)
            r_max = idx_r;
    }

    /*
    If no point is inside the RSLC radar grid, we still need to test
    if the bounding box covers the RSLC completely.

    Notice that all points could be located at one side (e.g. East)
    of the radar grid and the previous check would fail to detect
    that the area of interest has no intersection with the RSLC.
    */

    const bool flag_load_entire_rslc =
            (a_min <= margin_pixels && r_min <= margin_pixels &&
                    a_max >= radar_grid.length() - 1 - margin_pixels &&
                    r_max >= radar_grid.width() - 1 - margin_pixels);

    return flag_load_entire_rslc;
}

template<class T>
void ProjectSlantRange<T>::_getRadarPositionBorder(
        const double y0, const double x0, const double yf, const double xf,
        double* a_min, double* r_min, double* a_max, double* r_max,
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::core::ProjectionBase* proj,
        const std::function<Vec3(double, double,
                const isce3::geometry::DEMInterpolator&,
                isce3::core::ProjectionBase*)>& getDemCoords,
        isce3::geometry::DEMInterpolator& dem_interp,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction) {
    /*
    Get radar grid boundaries, i.e. min and max rg. and az. indexes, using
    the border of a geogrid bounding box.
    */

    // TODO fix this
    const int imax = _geoGridLength;
    const int jmax = _geoGridWidth;

    double az_time = radar_grid.sensingMid();
    double range_distance = radar_grid.midRange();

    bool flag_direction_line = true, flag_save_vectors = false;
    bool flag_compute_min_max = true;

    _getRadarPositionVect(y0, 0, jmax, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);

    _getRadarPositionVect(yf, 0, jmax, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);

    // pre-compute radar positions on the left side of the geogrid
    flag_direction_line = false;

    int i_start = 1;
    int i_end = imax - 1;

    _getRadarPositionVect(x0, i_start, i_end, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);

    _getRadarPositionVect(xf, i_start, i_end, &az_time,
            &range_distance, a_min, r_min, a_max, r_max, radar_grid, proj,
            dem_interp, getDemCoords, flag_direction_line, flag_save_vectors,
            flag_compute_min_max, az_time_correction, slant_range_correction);
}

template<class T>
void ProjectSlantRange<T>::_getRadarGridBoundaries(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
        isce3::core::ProjectionBase* proj,
        isce3::core::dataInterpMethod dem_interp_method, int* offset_y,
        int* offset_x, int* grid_size_y, int* grid_size_x)
{
    /*
    Get radar grid boundaries (offsets and window size) based on
    the ProjectSlantRange object geogrid attributes.
    */

    double y0 = _geoGridStartY;
    double x0 = _geoGridStartX;
    double yf = _geoGridStartY + _geoGridLength * _geoGridSpacingY;
    double xf = _geoGridStartX + _geoGridWidth * _geoGridSpacingX;

    isce3::geometry::DEMInterpolator dem_interp(0, dem_interp_method);

    auto error_code =
            loadDemFromProj(dem_raster, x0, xf, y0, yf, &dem_interp, proj);

    if (error_code != isce3::error::ErrorCode::Success) {
        throw isce3::except::RuntimeError(
                ISCE_SRCINFO(), "ERROR invalid DEM for given area");
    }

    int margin_pixels = 50;

    std::function<Vec3(double, double, const isce3::geometry::DEMInterpolator&,
            isce3::core::ProjectionBase*)>
            getDemCoords;

    if (proj->code() == dem_raster.getEPSG()) {
        getDemCoords = isce3::geometry::getDemCoordsSameEpsg;
    } else {
        getDemCoords = isce3::geometry::getDemCoordsDiffEpsg;
    }

    bool flag_load_entire_rslc = _checkLoadEntireRslcCorners(y0, x0, yf, xf,
            radar_grid, proj, getDemCoords, dem_interp, margin_pixels);

    /*
    If the four courners surround the RSLC, load entire RSLC
    */
    if (flag_load_entire_rslc) {
        *offset_y = 0;
        *offset_x = 0;
        *grid_size_y = radar_grid.length();
        *grid_size_x = radar_grid.width();
        return;
    }

    double a_min = std::numeric_limits<double>::quiet_NaN();
    double r_min = std::numeric_limits<double>::quiet_NaN();
    double a_max = std::numeric_limits<double>::quiet_NaN();
    double r_max = std::numeric_limits<double>::quiet_NaN();

    /*
    Otherwise, use the geogrid bounding box perimeter (borders) to obtain
    the minimum and maximum az. and rg. values
    */
    _getRadarPositionBorder(y0, x0, yf, xf, &a_min, &r_min,
            &a_max, &r_max, radar_grid, proj, getDemCoords, dem_interp);

    // azimuth block boundary
    *offset_y = std::min(
            std::max(static_cast<int>(std::floor(a_min) - margin_pixels), 0),
            static_cast<int>(input_raster.length() - 1));
    const int ybound = std::min(
            std::max(static_cast<int>(std::ceil(a_max) + margin_pixels), 0),
            static_cast<int>(input_raster.length() - 1));

    *grid_size_y = ybound - *offset_y + 1;

    // range block boundary
    *offset_x = std::min(
            std::max(static_cast<int>(std::floor(r_min) - margin_pixels), 0),
            static_cast<int>((input_raster.width() - 1)));

    const int xbound = std::min(
            std::max(static_cast<int>(std::floor(r_max) + margin_pixels), 0),
            static_cast<int>((input_raster.width() - 1)));
    *grid_size_x = xbound - *offset_x + 1;
}


template<class T>
void ProjectSlantRange<T>::project(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
        isce3::io::Raster& dem_raster, int exponent,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        GeocodeMemoryMode geocode_memory_mode, const long long min_block_size,
        const long long max_block_size,
        isce3::core::dataInterpMethod dem_interp_method)
{
    bool flag_complex_to_real = isce3::signal::verifyComplexToRealCasting(
            input_raster, output_raster, exponent);

    if (!flag_complex_to_real)
        _project<T>(radar_grid, input_raster, output_raster,
                dem_raster, az_time_correction,
                slant_range_correction, geocode_memory_mode,
                min_block_size, max_block_size,
                dem_interp_method);
    else if (std::is_same<T, double>::value ||
             std::is_same<T, std::complex<double>>::value)
        _project<double>(radar_grid, input_raster, output_raster,
                dem_raster, az_time_correction,
                slant_range_correction, geocode_memory_mode,
                min_block_size, max_block_size,
                dem_interp_method);
    else
        _project<float>(radar_grid, input_raster, output_raster,
                dem_raster, az_time_correction,
                slant_range_correction, geocode_memory_mode,
                min_block_size, max_block_size,
                dem_interp_method);

}

template<class T>
template<class T_out>
void ProjectSlantRange<T>::_project(
        const isce3::product::RadarGridParameters& radar_grid,
        isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
        isce3::io::Raster& dem_raster,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        GeocodeMemoryMode geocode_memory_mode, const long long min_block_size,
        const long long max_block_size,
        isce3::core::dataInterpMethod dem_interp_method)
{

    pyre::journal::info_t info("isce.geometry.ProjectSlantRange.project");
    pyre::journal::info_t warning("isce.geometry.ProjectSlantRange.project");
    pyre::journal::error_t error("isce.geometry.ProjectSlantRange.project");

    auto start_time = std::chrono::high_resolution_clock::now();

    // number of bands in the input raster
    int nbands = input_raster.numBands();

    // create projection based on epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(_epsgOut));

    // TODO fix this
    const int imax = _geoGridLength;
    const int jmax = _geoGridWidth;

    int offset_y, offset_x, grid_size_y, grid_size_x;

    _getRadarGridBoundaries(radar_grid, output_raster, dem_raster, proj.get(),
            dem_interp_method,
            &offset_y, &offset_x, &grid_size_y, &grid_size_x);
        
    if (offset_y != 0 || offset_x != 0 || grid_size_y != radar_grid.length() ||
        grid_size_x != radar_grid.width()) {
        warning << "input image does not cover the radargrid entirely"
                << pyre::journal::endl;
    }

    isce3::product::RadarGridParameters radar_grid_cropped =
            radar_grid.offsetAndResize(
                    offset_y, offset_x, grid_size_y, grid_size_x);

    bool is_radar_grid_single_block =
            (geocode_memory_mode !=
                    GeocodeMemoryMode::BlocksGeogridAndRadarGrid);

    // number of bands in the input raster
    info << "nbands: " << nbands << pyre::journal::newline;

    info << "radar grid width: " << radar_grid_cropped.width()
         << ", length: " << radar_grid_cropped.length()
         << pyre::journal::newline;

    int epsgcode = dem_raster.getEPSG();

    info << "DEM EPSG: " << epsgcode << pyre::journal::endl;
    if (epsgcode < 0) {
        std::string error_msg = "invalid DEM EPSG";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }
    info << "output EPSG: " << _epsgOut << pyre::journal::endl;

    info << "reproject DEM (0: false, 1: true): "
         << std::to_string(_epsgOut != dem_raster.getEPSG())
         << pyre::journal::newline;

    info << "apply azimuth offset (0: false, 1: true): "
            << std::to_string(az_time_correction.haveData())
            << pyre::journal::newline;
    info << "apply range offset (0: false, 1: true): "
            << std::to_string(slant_range_correction.haveData())
            << pyre::journal::newline;

    const long long progress_block = ((long long) imax) * jmax / 100;

    _print_parameters(info, geocode_memory_mode, min_block_size,
                      max_block_size);

    info << "is radar-grid single block: "
         << std::boolalpha  << is_radar_grid_single_block << std::noboolalpha 
         << pyre::journal::newline;

    /*
    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> inputData;

    if (is_radar_grid_single_block) {

        int geo_offset_x = 0;
        int geo_offset_y = 0;

        _getInputDataBlock<T, T_out>(inputData, input_raster,
                geo_offset_x, geo_offset_y, jmax, imax,
                geocode_memory_mode, min_block_size, max_block_size, info);
    }
    */

    int block_size_x, nblocks_x;
    int block_size_y, nblocks_y;
    if (geocode_memory_mode == GeocodeMemoryMode::SingleBlock) {

        nblocks_x = 1;
        block_size_x = _geoGridWidth;

        nblocks_y = 1;
        block_size_y = _geoGridLength;
    } else {
        isce3::core::getBlockProcessingParametersXY(
                imax, jmax, nbands, sizeof(T_out),
                &info, &block_size_y, &nblocks_y, 
                &block_size_x, &nblocks_x,
                min_block_size, max_block_size);
        
    }

    long long numdone = 0;

    info << "nblocks X: " << nblocks_x << pyre::journal::newline;
    info << "block size X: " << block_size_x << pyre::journal::newline;

    info << "nblocks Y: " << nblocks_y << pyre::journal::newline;
    info << "block size Y: " << block_size_y << pyre::journal::newline;

    info << "starting slant-range projection" << pyre::journal::endl;

    _Pragma("omp parallel for schedule(dynamic)")
    for (int block_y = 0; block_y < nblocks_y; ++block_y) {
        for (int block_x = 0; block_x < nblocks_x; ++block_x) {
            _runBlock<T_out>(
                radar_grid_cropped,
                is_radar_grid_single_block,
                // inputData,
                block_size_y, block_y,
                block_size_x, block_x,
                numdone, progress_block,
                nbands,
                dem_interp_method,
                dem_raster,
                proj.get(),
                az_time_correction, slant_range_correction,
                input_raster,
                offset_y, offset_x,
                output_raster,
                geocode_memory_mode,
                min_block_size, max_block_size,
                info);
        }
    }

    printf("\rslant-range projection progress: 100%%\n");

    auto elapsed_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    float elapsed_time = ((float) elapsed_time_milliseconds.count()) / 1e3;
    info << "elapsed time (slant-range projection) [s]: " << elapsed_time << pyre::journal::endl;
}

template<class T>
void ProjectSlantRange<T>::_getRadarPositionVect(double dem_pos_1, const int k_start,
        const int k_end, double* az_time,
        double* range_distance, double* y_min, double* x_min, double* y_max,
        double* x_max, const isce3::product::RadarGridParameters& radar_grid,
        isce3::core::ProjectionBase* proj,
        isce3::geometry::DEMInterpolator& dem_interp_block,
        const std::function<Vec3(double, double,
                const isce3::geometry::DEMInterpolator&,
                isce3::core::ProjectionBase*)>& getDemCoords,
        bool flag_direction_line, bool flag_save_vectors,
        bool flag_compute_min_max,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        std::vector<double>* a_vect, std::vector<double>* r_vect,
        std::vector<Vec3>* dem_vect) {
    /*
    Compute radar positions (az, rg, DEM vect.) for a geogrid vector
    (e.g. geogrid border) in X or Y direction (defined by flag_direction_line).
    If flag_compute_min_max is True, the function also return the min/max
    az. and rg. positions
    */

    double pixazm = 0.0, start = 0.0, dr = 0.0, r0 = 0.0;

    if (flag_compute_min_max) {
        // start (az) and r0 at the outer edge of the first pixel
        pixazm = radar_grid.azimuthTimeInterval();
        start = radar_grid.sensingStart() - 0.5 * pixazm;
        dr = radar_grid.rangePixelSpacing();
        r0 = radar_grid.startingRange() - 0.5 * dr;
    }

    for (int kk = k_start; kk <= k_end; ++kk) {
        const int k = kk - k_start;

        Vec3 dem_pos_vect;
        // Convert DEM coordinates (`dem_x` and `dem_y`) from _epsgOut to DEM
        // EPSG coordinates x and y, interpolate height (z), and return:
        // dem_pos_vect = {x, y, z}
        if (flag_direction_line) {
            // flag_direction_line == true: y fixed, varies x
            const double dem_pos_2 =
                    _geoGridStartX + _geoGridSpacingX * kk;
            dem_pos_vect =
                    getDemCoords(dem_pos_2, dem_pos_1, dem_interp_block, proj);
        } else {
            // flag_direction_line == false: x fixed, varies y
            const double dem_pos_2 =
                    _geoGridStartY + _geoGridSpacingY * kk;
            dem_pos_vect =
                    getDemCoords(dem_pos_1, dem_pos_2, dem_interp_block, proj);
        }

        // coarse geo2rdr
        int converged =
                _geo2rdrWrapper(dem_interp_block.proj()->inverse(dem_pos_vect),
                        _ellipsoid, _orbit, _doppler, *az_time, *range_distance,
                        radar_grid.wavelength(), radar_grid.lookSide(),
                        az_time_correction, slant_range_correction,
                        _threshold, _numiter, 1.0e-8, true);

        // if it didn't converge, reset initial solution and continue
        if (!converged) {
            *az_time = radar_grid.sensingMid();
            *range_distance = radar_grid.midRange();
            continue;
        }

        // otherwise, save solution
        if (flag_save_vectors) {
            a_vect->operator[](k) = *az_time;
            r_vect->operator[](k) = *range_distance;
            dem_vect->operator[](k) = dem_pos_vect;
        }

        if (!flag_compute_min_max)
            continue;

        // compute min/max pixel indexes
        double y = (*az_time - start) / pixazm;
        double x = (*range_distance - r0) / dr;

        // update min and max rg. and az. indexes
        if (std::isnan(*y_min) || y < *y_min)
            *y_min = y;
        if (std::isnan(*y_max) || y > *y_max)
            *y_max = y;
        if (std::isnan(*x_min) || x < *x_min)
            *x_min = x;
        if (std::isnan(*x_max) || x > *x_max)
            *x_max = x;
    }
}

template<class T>
template<class T_out>
void ProjectSlantRange<T>::_runBlock(
        const isce3::product::RadarGridParameters& radar_grid,
        bool is_radar_grid_single_block,
        // std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>>& inputData,
        int block_size_y, int block_y,
        int block_size_x, int block_x,
        long long& numdone, const long long& progress_block,
        int nbands,
        isce3::core::dataInterpMethod dem_interp_method,
        isce3::io::Raster& dem_raster,
        isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& az_time_correction,
        const isce3::core::LUT2d<double>& slant_range_correction,
        isce3::io::Raster& input_raster,
        int raster_offset_y, int raster_offset_x,
        isce3::io::Raster& output_raster,
        GeocodeMemoryMode geocode_memory_mode,
        const long long min_block_size, const long long max_block_size,
        pyre::journal::info_t& info)
{

    using isce3::math::complex_operations::operator*;
    using isce3::math::complex_operations::operator/;

    // start (az) and r0 at the outer edge of the first pixel
    const double pixazm = radar_grid.azimuthTimeInterval();
    const double start = radar_grid.sensingStart() - 0.5 * pixazm;
    const double dr = radar_grid.rangePixelSpacing();
    const double r0 = radar_grid.startingRange() - 0.5 * dr;

    // set NaN values according to T_out, i.e. real (NaN) or complex (NaN, NaN)
    using T_out_real = typename isce3::real<T_out>::type;
    T_out nan_t_out = 0;
    nan_t_out *= std::numeric_limits<T_out_real>::quiet_NaN();

    int this_block_size_y = block_size_y;
    if ((block_y + 1) * block_size_y > _geoGridLength)
        this_block_size_y = _geoGridLength % block_size_y;

    int this_block_size_x = block_size_x;
    if ((block_x + 1) * block_size_x > _geoGridWidth)
        this_block_size_x = _geoGridWidth % block_size_x;

    // TODO fix this
    int ii_0 = block_y * block_size_y;
    int jj_0 = block_x * block_size_x;

    isce3::geometry::DEMInterpolator dem_interp_block(0, dem_interp_method);

    double minX =
            _geoGridStartX +
            (static_cast<double>(jj_0) * _geoGridSpacingX);
    double maxX = _geoGridStartX +
                  std::min(static_cast<double>(jj_0) +
                                   this_block_size_x,
                          static_cast<double>(_geoGridWidth)) *
                          _geoGridSpacingX;

    double minY =
            _geoGridStartY +
            (static_cast<double>(ii_0) * _geoGridSpacingY);
    double maxY = _geoGridStartY +
                  std::min(static_cast<double>(ii_0) +
                                   this_block_size_y,
                          static_cast<double>(_geoGridLength)) *
                          _geoGridSpacingY;

    std::function<Vec3(double, double, const isce3::geometry::DEMInterpolator&,
            isce3::core::ProjectionBase*)>
            getDemCoords;

    if (_epsgOut == dem_raster.getEPSG()) {
        getDemCoords = isce3::geometry::getDemCoordsSameEpsg;
    } else {
        getDemCoords = isce3::geometry::getDemCoordsDiffEpsg;
    }

    // Load DEM using the block geogrid extents
    auto error_code = loadDemFromProj(dem_raster, minX, maxX, minY, maxY,
            &dem_interp_block, proj);

    if (error_code != isce3::error::ErrorCode::Success) {

        /*
        _fillGcovBlocksWithNans<T_out>(block_x, block_size_x, block_y,
            block_size_y, this_block_size_x, this_block_size_y,
            &output_raster);
        */

        return;
    }

    double a11 = radar_grid.sensingMid();
    double r11 = radar_grid.midRange();
    Vec3 dem11;

    // pre-compute radar positions on the top of the geogrid
    bool flag_direction_line = true, flag_save_vectors = true;
    bool flag_compute_min_max = !is_radar_grid_single_block;

    double a_idx_min = std::numeric_limits<double>::quiet_NaN();
    double r_idx_min = std::numeric_limits<double>::quiet_NaN();
    double a_idx_max = std::numeric_limits<double>::quiet_NaN();
    double r_idx_max = std::numeric_limits<double>::quiet_NaN();

    double dem_y1 =
            _geoGridStartY + _geoGridSpacingY * ii_0;
    std::vector<double> a_last(this_block_size_x + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_last(this_block_size_x + 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_last(this_block_size_x + 1,
                               {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()});
    _getRadarPositionVect(dem_y1, jj_0,
            jj_0 + this_block_size_x, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_last, &r_last, &dem_last);

    // pre-compute radar positions on the bottom of the geogrid
    dem_y1 = (_geoGridStartY +
              (_geoGridSpacingY * (ii_0 + this_block_size_y)));

    std::vector<double> a_bottom(this_block_size_x + 1,
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_bottom(this_block_size_x + 1,
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_bottom(this_block_size_x + 1,
                                 {std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()});
    _getRadarPositionVect(dem_y1, jj_0,
            jj_0 + this_block_size_x, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_bottom, &r_bottom, &dem_bottom);

    // pre-compute radar positions on the left side of the geogrid
    flag_direction_line = false;
    std::vector<double> a_left(this_block_size_y - 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_left(this_block_size_y - 1,
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_left(this_block_size_y - 1,
                               {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()});

    int i_start = (ii_0 + 1);
    int i_end = ii_0 + this_block_size_y - 1;

    double dem_x1 =
            _geoGridStartX + _geoGridSpacingX * jj_0;

    _getRadarPositionVect(dem_x1, i_start, i_end, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_left, &r_left, &dem_left);

    // pre-compute radar positions on the right side of the geogrid
    std::vector<double> a_right(this_block_size_y - 1,
                                std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r_right(this_block_size_y - 1,
                                std::numeric_limits<double>::quiet_NaN());
    std::vector<Vec3> dem_right(this_block_size_y - 1,
                                {std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()});

    dem_x1 = (_geoGridStartX +
              (_geoGridSpacingX * (jj_0 + this_block_size_x)));

    _getRadarPositionVect(dem_x1, i_start, i_end, &a11,
            &r11, &a_idx_min, &r_idx_min, &a_idx_max, &r_idx_max, radar_grid,
            proj, dem_interp_block, getDemCoords, flag_direction_line,
            flag_save_vectors, flag_compute_min_max, az_time_correction,
            slant_range_correction, &a_right, &r_right, &dem_right);

    // load radar grid data
    int offset_x = 0, offset_y = 0;
    int xbound = radar_grid.width() - 1;
    int ybound = radar_grid.length() - 1;

    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> outputData;
    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> outputDataBlock;

    // isce3::core::Matrix<float> rtc_area_block, rtc_area_sigma_block;
    // isce3::core::Matrix<uint8_t> input_layover_shadow_mask_block;
    std::vector<std::unique_ptr<isce3::core::Matrix<T_out>>> inputDataBlock;
    if (!is_radar_grid_single_block) {

        int margin_pixels = 25;

        // azimuth block boundary
        offset_y = std::min(std::max(static_cast<int>(std::floor(a_idx_min) -
                                                      margin_pixels),
                                    0),
                static_cast<int>(input_raster.length() - 1));
        ybound = std::min(
                std::max(static_cast<int>(std::ceil(a_idx_max) + margin_pixels),
                        0),
                static_cast<int>(input_raster.length() - 1));

        int grid_size_y = ybound - offset_y + 1;

        // range block boundary
        offset_x = std::min(std::max(static_cast<int>(std::floor(r_idx_min) -
                                                      margin_pixels),
                                    0),
                static_cast<int>((input_raster.width() - 1)));

        xbound = std::min(std::max(static_cast<int>(std::floor(r_idx_max) +
                                                    margin_pixels),
                                  0),
                static_cast<int>((input_raster.width() - 1)));

        int grid_size_x = xbound - offset_x + 1;

        if (grid_size_y <= 0 || grid_size_x <= 0) {

            /*
            _fillGcovBlocksWithNans<T_out>(block_x, block_size_x, block_y,
                block_size_y, this_block_size_x, this_block_size_y,
                &output_raster);
            */

            return;
        }

        isce3::product::RadarGridParameters radar_grid_block =
                radar_grid.offsetAndResize(offset_y, offset_x, grid_size_y,
                                           grid_size_x);

        _getInputDataBlock<T, T_out>(inputDataBlock, input_raster,
                dem_x1, dem_y1, this_block_size_x, this_block_size_y,
                geocode_memory_mode,
                min_block_size, max_block_size, info);


        outputDataBlock.reserve(nbands);
        for (int band = 0; band < nbands; ++band) {
            outputDataBlock.emplace_back(std::make_unique<isce3::core::Matrix<T_out>>(
                    radar_grid_block.length(), radar_grid_block.width()));
            outputDataBlock[band]->fill(0);
        }
    } else {
        outputData.reserve(nbands);
        for (int band = 0; band < nbands; ++band) {
            outputData.emplace_back(std::make_unique<isce3::core::Matrix<T_out>>(
                    radar_grid.length(), radar_grid.width()));
            outputData[band]->fill(0);
        }
    }

    // nan_t_out *= std::numeric_limits<T_out_real>::quiet_NaN();
 


    /*

         r_last[j], a_last[j]                   r_last[j+1], a_last[j+1]
       -----------|----------------------------------------|
         r01, a01 | r00, a00                      r01, a01 |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                 (i, j)                 |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                                        |
                  |                                        |
         r11, a11 | r10, a10                      r11, a11 |
       -----------|----------------------------------------|

       Notice that only the r11 and a11 position that need to be calculated.
       As execution moves to the right. The new r10 and a10 will update their
       values from the previous r11, a11 and so on. The values of the upper
       vertices are obtained from the r_last and a_last vectors.

    */

    for (int i = 0; i < this_block_size_y; ++i) {

        // initiating lower right vertex
        const int ii = block_y * block_size_y + i;

        if (i < this_block_size_y - 1) {
            a11 = a_left[i];
            r11 = r_left[i];
            dem11 = dem_left[i];
        } else {
            a11 = a_bottom[0];
            r11 = r_bottom[0];
            dem11 = dem_bottom[0];
        }

        // initiating lower edge geogrid lat/northing position
        dem_y1 = _geoGridStartY +
                 _geoGridSpacingY * (1.0 + ii);

        for (int j = 0; j < this_block_size_x; ++j) {

            const int jj = block_x * block_size_x + j;

            _Pragma("omp atomic") numdone++;
            if (numdone % progress_block == 0)
                _Pragma("omp critical")
                {
                    printf("\rslant-range project progress: %d%%",
                            static_cast<int>(numdone / progress_block)),
                            fflush(stdout);
                }

            // bottom left (copy from previous bottom right)
            const double a10 = a11;
            const double r10 = r11;
            const Vec3 dem10 = dem11;

            // top left (copy from a_last, r_last, and dem_last)
            const double a00 = a_last[j];
            const double r00 = r_last[j];
            const Vec3 dem00 = dem_last[j];

            // top right (copy from a_last, r_last, and dem_last)
            const double a01 = a_last[j + 1];
            const double r01 = r_last[j + 1];
            const Vec3 dem01 = dem_last[j + 1];

            // update "last" vectors (from lower left vertex)
            a_last[j] = a10;
            r_last[j] = r10;
            dem_last[j] = dem10;

            if (i < this_block_size_y - 1 &&
                j < this_block_size_x - 1) {
                // pre-calculate new bottom right
                if (!std::isnan(a10) && !std::isnan(a00) && !std::isnan(a01)) {
                    a11 = a01 + a10 - a00;
                    r11 = r01 + r10 - r00;
                } else if (std::isnan(a11) && !std::isnan(a01)) {
                    a11 = a01;
                    r11 = r01;
                } else if (std::isnan(a11) && !std::isnan(a00)) {
                    a11 = a00;
                    r11 = r00;
                }

                const double dem_x1 =
                        _geoGridStartX +
                        _geoGridSpacingX * (1.0 + jj);

                // Convert DEM coordinates (`dem_x` and `dem_y`) from _epsgOut
                // to DEM EPSG coordinates x and y, interpolate height (z), and
                // return: dem11 = {x, y, z}
                dem11 = getDemCoords(dem_x1, dem_y1, dem_interp_block, proj);

                int converged = _geo2rdrWrapper(
                        dem_interp_block.proj()->inverse(dem11), _ellipsoid,
                        _orbit, _doppler, a11, r11, radar_grid.wavelength(),
                        radar_grid.lookSide(), az_time_correction,
                        slant_range_correction, _threshold, _numiter, 1.0e-8);

                if (!converged) {
                    a11 = std::numeric_limits<double>::quiet_NaN();
                    r11 = std::numeric_limits<double>::quiet_NaN();
                }

            } else if (i >= this_block_size_y - 1 &&
                       !std::isnan(a_bottom[j + 1]) &&
                       !std::isnan(r_bottom[j + 1])) {
                a11 = a_bottom[j + 1];
                r11 = r_bottom[j + 1];
                dem11 = dem_bottom[j + 1];
            } else if (j >= this_block_size_x - 1 &&
                       !std::isnan(a_right[i]) && !std::isnan(r_right[i])) {
                a11 = a_right[i];
                r11 = r_right[i];
                dem11 = dem_right[i];
            } else {
                a11 = std::numeric_limits<double>::quiet_NaN();
                r11 = std::numeric_limits<double>::quiet_NaN();
            }

            // if last column, also update top-right "last" arrays (from lower
            //   right vertex)
            if (j == this_block_size_x - 1) {
                a_last[j + 1] = a11;
                r_last[j + 1] = r11;
                dem_last[j + 1] = dem11;
            }

            if (std::isnan(a00) || std::isnan(a10) || std::isnan(a10) ||
                    std::isnan(a11)) {
                continue;
            }

            double y00 = (a00 - start) / pixazm;
            double y10 = (a10 - start) / pixazm;
            double y01 = (a01 - start) / pixazm;
            double y11 = (a11 - start) / pixazm;

            double x00 = (r00 - r0) / dr;
            double x10 = (r10 - r0) / dr;
            double x01 = (r01 - r0) / dr;
            double x11 = (r11 - r0) / dr;

            int margin = isce3::core::AREA_PROJECTION_RADAR_GRID_MARGIN;

            // define slant-range window
            const int y_min = std::floor((std::min(std::min(y00, y01),
                                      std::min(y10, y11)))) -
                              1;
            if (y_min < -margin ||
                y_min > ybound + 1)
                continue;
            const int x_min = std::floor((std::min(std::min(x00, x01),
                                      std::min(x10, x11)))) -
                              1;
            if (x_min < -margin ||
                x_min > xbound + 1)
                continue;
            const int y_max = std::ceil((std::max(std::max(y00, y01),
                                      std::max(y10, y11)))) +
                              1;
            if (y_max > ybound + 1 + margin || y_max < -1 || y_max < y_min)
                continue;
            const int x_max = std::ceil((std::max(std::max(x00, x01),
                                      std::max(x10, x11)))) +
                              1;
            if (x_max > xbound + 1 + margin || x_max < -1 || x_max < x_min)
                continue;

            // Crop indexes around (x_min, y_min) and (x_max, y_max)
            // New indexes vary from 0 to (size_x, size_y)
            double y00_cut = y00 - y_min;
            double y10_cut = y10 - y_min;
            double y01_cut = y01 - y_min;
            double y11_cut = y11 - y_min;
            double x00_cut = x00 - x_min;
            double x10_cut = x10 - x_min;
            double x01_cut = x01 - x_min;
            double x11_cut = x11 - x_min;
            const int size_x = x_max - x_min + 1;
            const int size_y = y_max - y_min + 1;

            isce3::core::Matrix<double> w_arr(size_y, size_x);
            w_arr.fill(0);
            double w_total = 0;
            int plane_orientation;
            if (radar_grid.lookSide() == isce3::core::LookSide::Left)
                plane_orientation = -1;
            else
                plane_orientation = 1;

            isce3::geometry::areaProjIntegrateSegment(y00_cut, y01_cut, x00_cut,
                    x01_cut, size_y, size_x, w_arr, w_total, plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(y01_cut, y11_cut, x01_cut,
                    x11_cut, size_y, size_x, w_arr, w_total, plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(y11_cut, y10_cut, x11_cut,
                    x10_cut, size_y, size_x, w_arr, w_total, plane_orientation);
            isce3::geometry::areaProjIntegrateSegment(y10_cut, y00_cut, x10_cut,
                    x00_cut, size_y, size_x, w_arr, w_total, plane_orientation);

            bool flag_self_intersecting_area_element = false;

            // test for self-intersection
            for (int yy = 0; yy < size_y; ++yy) {
                for (int xx = 0; xx < size_x; ++xx) {
                    double w = w_arr(yy, xx);
                    if (w * w_total < 0 && abs(w) >  0.00001) {
                        flag_self_intersecting_area_element = true;
                        break;
                    }
                }
                if (flag_self_intersecting_area_element) {
                    break;
                }
            }

            if (flag_self_intersecting_area_element) {
                /*
                If self-intersecting, divide area element (geogrid pixel) into
                two triangles and integrate them separately.
                */
                isce3::core::Matrix<double> w_arr_1(size_y, size_x);
                w_arr_1.fill(0);
                double w_total_1 = 0;
                isce3::geometry::areaProjIntegrateSegment(y00_cut, y01_cut, x00_cut,
                        x01_cut, size_y, size_x, w_arr_1, w_total_1, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y01_cut, y11_cut, x01_cut,
                        x11_cut, size_y, size_x, w_arr_1, w_total_1, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y11_cut, y00_cut, x11_cut,
                        x00_cut, size_y, size_x, w_arr_1, w_total_1, plane_orientation);

                isce3::core::Matrix<double> w_arr_2(size_y, size_x);
                w_arr_2.fill(0);
                double w_total_2 = 0;
                isce3::geometry::areaProjIntegrateSegment(y00_cut, y11_cut, x00_cut,
                        x11_cut, size_y, size_x, w_arr_2, w_total_2, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y11_cut, y10_cut, x11_cut,
                        x10_cut, size_y, size_x, w_arr_2, w_total_2, plane_orientation);
                isce3::geometry::areaProjIntegrateSegment(y10_cut, y00_cut, x10_cut,
                        x00_cut, size_y, size_x, w_arr_2, w_total_2, plane_orientation);

                w_total = 0;
                /*
                The new weight array `w_arr` is the sum of the absolute values of both
                triangles weighted arrays `w_arr_1` and `w_arr_2`. The integrated
                total `w_total` is updated accordingly.
                */
                for (int yy = 0; yy < size_y; ++yy) {
                    for (int xx = 0; xx < size_x; ++xx) {
                        w_arr(yy, xx) = std::min(
                            abs(w_arr_1(yy, xx)) + abs(w_arr_2(yy, xx)), 1.0);
                        w_total += w_arr(yy, xx);
                    }
                }
            }

            double nlooks = 0;
            // std::vector<T> v1(nbands, 0);

            // TODO fix this
            // x, y positions are binned by integer quotient (floor)
            const int x = static_cast<int>(j);
            const int y = static_cast<int>(i);

            // compute backscatter contribution v and update output arrays

            for (int band = 0; band < nbands; ++band) {
                // T_out v = (static_cast<T_out>(
                //         (cumulative_sum[band]) / nlooks));
                T v1 = inputDataBlock[band]->operator()(y, x);

                if (std::isnan(std::abs(v1))) {
                    continue;
                }

                // add all slant-range elements that contributes to the geogrid
                // pixel
                for (int yy = 0; yy < size_y; ++yy) {
                    for (int xx = 0; xx < size_x; ++xx) {
                        double w = w_arr(yy, xx);
                        int y = yy + y_min;
                        int x = xx + x_min;

                        /* Radar sample does not intersect with projected polygon
                        (geogrid pixel)
                        */
                        if (w == 0)
                            continue;

                        // Radar sample is out of bounds
                        else if (y - offset_y < 0 || x - offset_x < 0 ||
                                y >= ybound || x >= xbound) {
                            continue;
                        }

                        w = std::abs(w);
                        /*
                        if (is_radar_grid_single_block || flag_rtc_raster_is_in_memory) {
                            rtc_value = rtc_area(y, x);
                        } else {
                            rtc_value =
                                    rtc_area_block(y - offset_y, x - offset_x);
                        }
                                    */

                        nlooks += w;

                        // isce3::geocode::_accumulate(cumulative_sum[band], v1, w);

                        // int band_index = 0;
                        // for (int band = 0; band < nbands; ++band) {

                        if (is_radar_grid_single_block) {
                            isce3::geocode::_accumulate(
                                outputData[band]->operator()(
                                    y - offset_y, x - offset_x), v1, w);
                        } else {
                            isce3::geocode::_accumulate(
                                outputDataBlock[band]->operator()(
                                    y - offset_y, x - offset_x), v1, w);
                        }


                        // }
                    }
                    if (std::isnan(nlooks))
                        break;
                }

                
            }
        }

        /*
        for (int band = 0; band < nbands; ++band) {
            if (is_radar_grid_single_block) {
            _Pragma("omp critical")
                {
                    output_raster.setBlock(
                        outputData[band]->data(),
                        raster_offset_x, raster_offset_y, 
                        xbound - raster_offset_x + 1,
                        ybound - raster_offset_y + 1,
                        band + 1);
                }
            }
            else {
                _Pragma("omp critical")
                {
                    output_raster.setBlock(
                        outputDataBlock[band]->data(),
                        raster_offset_x, raster_offset_y, 
                        xbound - raster_offset_x + 1,
                        ybound - raster_offset_y + 1,
                        band + 1);
                }
            }
        }
    }

    outputDataBlock.clear();

}

/** Convert enum output_mode to string */
std::string _get_geocode_memory_mode_str(
        isce3::core::GeocodeMemoryMode geocode_memory_mode) {
    std::string geocode_memory_mode_str;
    switch (geocode_memory_mode) {
    case isce3::core::GeocodeMemoryMode::SingleBlock:
        geocode_memory_mode_str = "single block";
        break;
    case isce3::core::GeocodeMemoryMode::BlocksGeogrid:
        geocode_memory_mode_str = "blocks geogrid";
        break;
    case isce3::core::GeocodeMemoryMode::BlocksGeogridAndRadarGrid:
        geocode_memory_mode_str = "blocks geogrid and radargrid";
        break;
    case isce3::core::GeocodeMemoryMode::Auto:
        geocode_memory_mode_str = "auto";
        break;
    default:
        std::string error_message = "ERROR invalid geocode memory mode";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
        break;
    }
    return geocode_memory_mode_str;
}

template<class T>
void ProjectSlantRange<T>::_print_parameters(pyre::journal::info_t& channel, 
                                  isce3::core::GeocodeMemoryMode& geocode_memory_mode,
                                  const long long min_block_size,
                                  const long long max_block_size) {
    channel << "geocode memory mode: "
            << _get_geocode_memory_mode_str(geocode_memory_mode)
            << pyre::journal::newline
            << "min. block size: " << isce3::core::getNbytesStr(min_block_size)
            << pyre::journal::newline
            << "max. block size: " << isce3::core::getNbytesStr(max_block_size)
            << pyre::journal::newline
            << pyre::journal::endl;
}

template class ProjectSlantRange<float>;
template class ProjectSlantRange<double>;
template class ProjectSlantRange<std::complex<float>>;
template class ProjectSlantRange<std::complex<double>>;


} // namespace geocode
} // namespace isce3
