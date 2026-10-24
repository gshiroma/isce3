#pragma once

#include <functional>
#include <optional>

// pyre
#include <pyre/journal.h>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/blockProcessing.h>

// isce3::geometry
#include <isce3/geometry/DEMInterpolator.h>

// isce3::io
#include <isce3/io/Raster.h>

// isce3::product
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3 { namespace geometry {

template<class T>
class ProjectSlantRange {
public:
    /**
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_raster        Input raster. Can be real-
     * or complex-valued. If the input raster is complex-valued
     * and the output is real-valued, it is assumed that the
     * input raster represents single-look complex (SLC) data.
     * In such cases, the complex data
     * is converted to real-valued backscatter, which is proportional to
     * power or intensity. This conversion is performed by taking the
     * square of the SLC magnitudes, and it is applied before geocoding.
     * @param[out] output_raster       Output raster. Can be real-
     * or complex-valued. This module provides options to perform
     * absolute radiometric calibration, through `abs_cal_factor`
     * and radiometric terrain correction (RTC). To apply these
     * calibrations, it is assumed that complex-valued output rasters
     * represent single-look complex (SLC) data.
     * Both `abs_cal_factor` and RTC normalization values are
     * defined in terms of power or intensity. Therefore, when applied
     * to SLC data, the square roots of these values are used to properly
     * calibrate the magnitude of the complex signal.
     * If the output is a complex interferogram, the user should ensure that
     * RTC correction and absolute radiometric calibration are disabled.
     * This can be done by not providing the `flag_apply_rtc` or
     * `abs_cal_factor parameters`, which default to `false` and `1`,
     * respectively.
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  exponent            Exponent to be applied to the input data.
     * The value 0 indicates that the the exponent is based on the data type of
     * the input raster (1 for real and 2 for complex rasters).
     * @param[in]  az_time_correction     Azimuth additive correction, in
     * seconds, as a function of azimuth and range
     * @param[in]  slant_range_correction  Slant range additive correction,
     * in meters, as a function of azimuth and range
     * @param[in]  geocode_memory_mode Select memory mode
     * @param[in]  min_block_size      Minimum block size (per thread)
     * @param[in]  max_block_size      Maximum block size (per thread)
     * @param[in]  dem_interp_method   DEM interpolation method
     */
    void project(const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
            isce3::io::Raster& dem_raster,
            int exponent = 0,
            const isce3::core::LUT2d<double>& az_time_correction = {},
            const isce3::core::LUT2d<double>& slant_range_correction = {},
            isce3::core::GeocodeMemoryMode geocode_memory_mode =
                    isce3::core::GeocodeMemoryMode::Auto,
            const long long min_block_size =
                    isce3::core::DEFAULT_MIN_BLOCK_SIZE,
            const long long max_block_size =
                    isce3::core::DEFAULT_MAX_BLOCK_SIZE,
            isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD);

    template<class T_out>
    void _project(const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
            isce3::io::Raster& dem_raster,
            const isce3::core::LUT2d<double>& az_time_correction = {},
            const isce3::core::LUT2d<double>& slant_range_correction = {},
            isce3::core::GeocodeMemoryMode geocode_memory_mode =
                    isce3::core::GeocodeMemoryMode::Auto,
            const long long min_block_size =
                    isce3::core::DEFAULT_MIN_BLOCK_SIZE,
            const long long max_block_size =
                    isce3::core::DEFAULT_MAX_BLOCK_SIZE,
            isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD);


    /** Set the output geogrid
     * @param[in]  geoGridStartY       Starting Lat/Northing position
     * @param[in]  geoGridSpacingY     Lat/Northing step size
     * @param[in]  geoGridStartX       Starting Lon/Easting position
     * @param[in]  geoGridSpacingX     Lon/Easting step size
     * @param[in]  geogrid_width       Geographic width (number of pixels) in
     * the Lon/Easting direction
     * @param[in]  geogrid_length      Geographic length (number of pixels) in
     * the Lat/Northing direction
     * @param[in]  epsgcode            Output geographic grid EPSG
     */
    void geoGrid(double geoGridStartX, double geoGridStartY,
                 double geoGridSpacingX, double geoGridSpacingY, int width,
                 int length, int epsgcode);

    /** Update the output geogrid with radar grid and DEM attributes
     * @param[in]  radar_grid          Radar grid
     * @param[in]  dem_raster          Input DEM raster
     */
    void updateGeoGrid(const isce3::product::RadarGridParameters& radar_grid,
                       isce3::io::Raster& dem_raster);

    // Get/set data interpolator
    isce3::core::dataInterpMethod dataInterpolator() const 
    { 
            return _data_interp_method; 
    }

    void doppler(isce3::core::LUT2d<double> doppler) { _doppler = doppler; }

    void nativeDoppler(isce3::core::LUT2d<double> nativeDoppler)
    {
        _nativeDoppler = nativeDoppler;
    }

    void orbit(isce3::core::Orbit& orbit) { _orbit = orbit; }

    void ellipsoid(isce3::core::Ellipsoid& ellipsoid)
    {
        _ellipsoid = ellipsoid;
    }

    void thresholdGeo2rdr(double threshold) { _threshold = threshold; }

    void numiterGeo2rdr(int numiter) { _numiter = numiter; }

    void radarBlockMargin(int radarBlockMargin)
    {
        _radarBlockMargin = radarBlockMargin;
    }

    // start X position for the output geogrid
    double geoGridStartX() const { return _geoGridStartX; }

    // start Y position for the output geogrid
    double geoGridStartY() const { return _geoGridStartY; }

    // X spacing for the output geogrid
    double geoGridSpacingX() const { return _geoGridSpacingX; }

    // Y spacing for the output geogrid
    double geoGridSpacingY() const { return _geoGridSpacingY; }

    // number of pixels in east-west direction (X direction)
    int geoGridWidth() const { return _geoGridWidth; }

    // number of lines in north-south direction (Y direction)
    int geoGridLength() const { return _geoGridLength; }

private:
    /*
    Get radar grid boundaries (offsets and window size) based on
    the Geocode object geogrid attributes.
    */
    void _getRadarGridBoundaries(
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
            isce3::core::ProjectionBase* proj,
            isce3::core::dataInterpMethod dem_interp_method, int* offset_y,
            int* offset_x, int* grid_size_y, int* grid_size_x);

    /*
    Compute radar positions (az, rg, DEM vect.) for a geogrid vector
    (e.g. geogrid border) in X or Y direction (defined by flag_direction_line).
    If flag_compute_min_max is True, the function also return the min/max
    az. and rg. positions
    */
    void _getRadarPositionVect(double dem_y1, const int k_start,
            const int k_end, double* a11,
            double* r11, double* y_min, double* x_min, double* y_max,
            double* x_max,
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::core::ProjectionBase* proj,
            isce3::geometry::DEMInterpolator& dem_interp_block,
            const std::function<Vec3(double, double,
                    const isce3::geometry::DEMInterpolator&,
                    isce3::core::ProjectionBase*)>& getDemCoords,
            bool flag_direction_line, bool flag_save_vectors,
            bool flag_compute_min_max,
            const isce3::core::LUT2d<double>& az_time_correction,
            const isce3::core::LUT2d<double>& slant_range_correction,
            std::vector<double>* a_last = nullptr,
            std::vector<double>* r_last = nullptr,
            std::vector<Vec3>* dem_last = nullptr);
    /*
    Check if a geogrid bounding box (y0, x0, yf, xf) fully
    covers the RSLC (represented by the radar_grid).
    */
    bool _checkLoadEntireRslcCorners(const double y0, const double x0,
            const double yf, const double xf,
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::core::ProjectionBase* proj,
            const std::function<Vec3(double, double,
                    const isce3::geometry::DEMInterpolator&,
                    isce3::core::ProjectionBase*)>& getDemCoords,
            isce3::geometry::DEMInterpolator& dem_interp, int margin_pixels);

    /*
    Get radar grid boundaries, i.e. min and max rg. and az. indexes, using
    the border of a geogrid bounding box.
    */
    void _getRadarPositionBorder(const double dem_y1,
            const double dem_x1, const double dem_yf, const double dem_xf,
            double* a_min, double* r_min, double* a_max, double* r_max,
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::core::ProjectionBase* proj,
            const std::function<Vec3(double, double,
                    const isce3::geometry::DEMInterpolator&,
                    isce3::core::ProjectionBase*)>& getDemCoords,
            isce3::geometry::DEMInterpolator& dem_interp,
            const isce3::core::LUT2d<double>& az_time_correction = {},
            const isce3::core::LUT2d<double>& slant_range_correction = {});

    template<class T_out>
    void _runBlock(const isce3::product::RadarGridParameters& radar_grid,
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
            isce3::core::GeocodeMemoryMode geocode_memory_mode,
            const long long min_block_size, const long long max_block_size,
            pyre::journal::info_t& info);

    std::string _get_nbytes_str(long nbytes);

    int _geo2rdr(const isce3::product::RadarGridParameters& radar_grid,
            double x, double y, double& azimuthTime, double& slantRange,
            isce3::geometry::DEMInterpolator& demInterp,
            isce3::core::ProjectionBase* proj, float& dem_value);

    void _print_parameters(pyre::journal::info_t& channel, 
                           isce3::core::GeocodeMemoryMode& geocode_memory_mode,
                           const long long min_block_size,
                           const long long max_block_size);

    // isce3::core objects
    isce3::core::Orbit _orbit;
    isce3::core::Ellipsoid _ellipsoid;

    // Optimization options

    double _threshold = 1e-8;
    int _numiter = 100;

    // radar grids parameters
    isce3::core::LUT2d<double> _doppler;

    // native Doppler
    isce3::core::LUT2d<double> _nativeDoppler;

    // start X position for the output geogrid
    double _geoGridStartX = std::numeric_limits<double>::quiet_NaN();

    // start Y position for the output geogrid
    double _geoGridStartY = std::numeric_limits<double>::quiet_NaN();

    // X spacing for the output geogrid
    double _geoGridSpacingX = std::numeric_limits<double>::quiet_NaN();

    // Y spacing for the output geogrid
    double _geoGridSpacingY = std::numeric_limits<double>::quiet_NaN();

    // number of pixels in east-west direction (X direction)
    int _geoGridWidth = -32768;

    // number of lines in north-south direction (Y direction)
    int _geoGridLength = -32768;

    // epsg code for the output geogrid
    int _epsgOut = 0;

    // margin around the computed bounding box for radar dara (integer number of
    // lines/pixels)
    int _radarBlockMargin;

    // interpolator
    isce3::core::dataInterpMethod _data_interp_method =
            isce3::core::dataInterpMethod::BIQUINTIC_METHOD;
};

}} // namespace isce3::geocode

