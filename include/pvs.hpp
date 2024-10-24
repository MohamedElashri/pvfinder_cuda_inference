#pragma once
#include <vector>
#include <tuple>

namespace pvfinder {

// Find PV locations using updated resolution method
std::vector<float> pv_locations_updated_res(
    const std::vector<float>& targets,
    float threshold,
    float integral_threshold,
    int min_width);

// Find PV locations using standard resolution method
std::vector<float> pv_locations_res(
    const std::vector<float>& targets,
    float threshold,
    float integral_threshold,
    int min_width);

// Filter NaN values from PV locations
std::vector<float> filter_nans_res(
    const std::vector<float>& items,
    const std::vector<float>& mask);

// Get reconstruction resolution
std::vector<float> get_reco_resolution(
    const std::vector<float>& pred_PVs_loc,
    const std::vector<float>& predict,
    float nsig_res,
    int steps_extrapolation,
    float ratio_max,
    bool debug = false);

// Compare reconstruction results
std::tuple<int, int, int> compare_res_reco(
    std::vector<float> target_PVs_loc,
    const std::vector<float>& pred_PVs_loc,
    const std::vector<float>& reco_res,
    bool debug = false);

} // namespace pvfinder
