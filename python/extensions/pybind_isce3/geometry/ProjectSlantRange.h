#pragma once

#include <isce3/geometry/ProjectSlantRange.h>
#include <pybind11/pybind11.h>

template<typename T>
void addbinding(pybind11::class_<isce3::geometry::ProjectSlantRange<T>>&);
