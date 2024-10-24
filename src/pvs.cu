#include "pvs.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace pvfinder {

std::vector<float> pv_locations_updated_res(
    const std::vector<float>& targets,
    float threshold,
    float integral_threshold,
    int min_width) {
    
    int state = 0;
    float integral = 0.0f;
    float sum_weights_locs = 0.0f;
    
    std::vector<float> items;
    items.reserve(150);  // Same as Python numpy.empty(150)
    
    bool peak_passed = false;
    float local_peak_value = 0.0f;
    int local_peak_index = 0;
    
    for (int i = 0; i < targets.size(); i++) {
        if (targets[i] >= threshold) {
            state += 1;
            integral += targets[i];
            sum_weights_locs += i * targets[i];
            
            if (targets[i] > local_peak_value) {
                local_peak_value = targets[i];
                local_peak_index = i;
            }
            
            if (i > 0 && (targets[i-1] > targets[i] + 0.05) && 
                (targets[i-1] > 1.1 * targets[i])) {
                peak_passed = true;
            }
        }
        
        if ((targets[i] < threshold || i == targets.size() - 1 || 
             (i > 0 && targets[i-1] < targets[i] && peak_passed)) && state > 0) {
            
            if (state >= min_width && integral >= integral_threshold) {
                items.push_back((sum_weights_locs / integral) + 0.5f);
            }
            
            // Reset state
            state = 0;
            integral = 0.0f;
            sum_weights_locs = 0.0f;
            peak_passed = false;
            local_peak_value = 0.0f;
        }
    }
    
    return items;
}

std::vector<float> pv_locations_res(
    const std::vector<float>& targets,
    float threshold,
    float integral_threshold,
    int min_width) {
    
    int state = 0;
    float integral = 0.0f;
    float sum_weights_locs = 0.0f;
    
    std::vector<float> items;
    items.reserve(150);
    
    for (int i = 0; i < targets.size(); i++) {
        if (targets[i] >= threshold) {
            state += 1;
            integral += targets[i];
            sum_weights_locs += i * targets[i];
        }
        
        if ((targets[i] < threshold || i == targets.size() - 1) && state > 0) {
            if (state >= min_width && integral >= integral_threshold) {
                items.push_back((sum_weights_locs / integral) + 0.5f);
            }
            
            state = 0;
            integral = 0.0f;
            sum_weights_locs = 0.0f;
        }
    }
    
    return items;
}

std::vector<float> filter_nans_res(
    const std::vector<float>& items,
    const std::vector<float>& mask) {
    
    std::vector<float> retval;
    retval.reserve(items.size());
    
    for (float item : items) {
        int index = static_cast<int>(std::round(item));
        if (index >= 0 && index < mask.size() && !std::isnan(mask[index])) {
            retval.push_back(item);
        }
    }
    
    return retval;
}

std::vector<float> get_reco_resolution(
    const std::vector<float>& pred_PVs_loc,
    const std::vector<float>& predict,
    float nsig_res,
    int steps_extrapolation,
    float ratio_max,
    bool debug) {
    
    std::vector<float> predict_clean = predict;
    for (auto& val : predict_clean) {
        if (std::isnan(val)) val = 0.0f;
    }
    
    std::vector<float> reco_reso(pred_PVs_loc.size());
    float rms = 1.0f / std::sqrt(12.0f);
    
    if (steps_extrapolation == 0) {
        // [Previous implementation for steps_extrapolation == 0]
    } else {
        if (debug) {
            std::cout << " pred_PVs_loc = " << std::endl;
            for (const auto& pv : pred_PVs_loc) {
                std::cout << pv << " ";
            }
            std::cout << std::endl;
        }

        for (size_t i = 0; i < pred_PVs_loc.size(); i++) {
            int predict_pv_ibin = static_cast<int>(pred_PVs_loc[i]);
            float predict_pv_KDE_max = predict_clean[predict_pv_ibin];
            float FHWM = ratio_max * predict_pv_KDE_max;

            if (debug) {
                std::cout << " ***** " << std::endl;
                std::cout << " step != 0 " << std::endl;
                std::cout << " predict_pv, predict_pv_ibin, predict_pv_KDE_max = "
                         << pred_PVs_loc[i] << ", " << predict_pv_ibin << ", " 
                         << predict_pv_KDE_max << std::endl;
            }

            float ibin_min_extrapol = -1;
            float ibin_max_extrapol = -1;
            bool found_min = false;
            bool found_max = false;

            // Find minimum bin with extrapolation
            for (int ibin = predict_pv_ibin; ibin > std::max(0, predict_pv_ibin-20); --ibin) {
                if (!found_min) {
                    float predict_pv_KDE_val_ibin = predict_clean[ibin];
                    float predict_pv_KDE_val_prev = predict_clean[ibin-1];

                    // Apply linear extrapolation
                    float delta_steps = (predict_pv_KDE_val_prev - predict_pv_KDE_val_ibin) / steps_extrapolation;
                    float extrapolated_val = predict_pv_KDE_val_ibin;

                    for (int sub_bin = 0; sub_bin < steps_extrapolation; sub_bin++) {
                        extrapolated_val -= delta_steps * sub_bin;

                        if (extrapolated_val < FHWM) {
                            ibin_min_extrapol = static_cast<float>(ibin * steps_extrapolation - sub_bin) / steps_extrapolation;
                            found_min = true;
                            break;
                        }
                    }
                }
            }

            // Find maximum bin with extrapolation
            for (int ibin = predict_pv_ibin; 
                 ibin < std::min(static_cast<int>(predict_clean.size()), predict_pv_ibin+20); 
                 ++ibin) {
                if (!found_max) {
                    float predict_pv_KDE_val_ibin = predict_clean[ibin];
                    float predict_pv_KDE_val_next = predict_clean[ibin+1];

                    // Apply linear extrapolation
                    float delta_steps = (predict_pv_KDE_val_ibin - predict_pv_KDE_val_next) / steps_extrapolation;
                    float extrapolated_val = predict_pv_KDE_val_ibin;

                    for (int sub_bin = 0; sub_bin < steps_extrapolation; sub_bin++) {
                        extrapolated_val -= delta_steps * sub_bin;

                        if (extrapolated_val < FHWM) {
                            ibin_max_extrapol = (ibin * steps_extrapolation + sub_bin) / 
                                              static_cast<float>(steps_extrapolation);
                            found_max = true;
                            break;
                        }
                    }
                }

                // Calculate RMS if both bounds are found
                if (found_min && found_max) {
                    float sumsq = 0.0f;
                    float sumContents = 0.0f;

                    for (int index = static_cast<int>(ibin_min_extrapol); 
                         index <= static_cast<int>(ibin_max_extrapol); 
                         index++) {
                        float contents = predict_clean[index];
                        
                        if (debug) {
                            std::cout << "index, contents = " << index << ", " 
                                     << contents << std::endl;
                        }

                        float diff = (index + 0.5f - pred_PVs_loc[i]);
                        sumsq += diff * diff * contents;
                        sumContents += contents;

                        if (debug) {
                            std::cout << "index+0.5, predict_pv, contents, sumsq, sumContents = "
                                     << (index + 0.5f) << ", " << pred_PVs_loc[i] << ", " 
                                     << contents << ", " << sumsq << ", " << sumContents 
                                     << std::endl;
                        }
                    }

                    float local_rms = sumsq / sumContents;
                    
                    if (debug) {
                        std::cout << "rms = " << std::fixed << std::setprecision(2) 
                                 << local_rms << std::endl;
                    }

                    reco_reso[i] = nsig_res * local_rms;
                }
            }

            if (debug && !(found_min && found_max)) {
                std::cout << " not (found_min and found_max) " << std::endl;
            }

            float FHWM_w = ibin_max_extrapol - ibin_min_extrapol;
            if (debug) {
                std::cout << "FHWM_w = " << FHWM_w << std::endl;
            }

            float standard_dev = FHWM_w / 2.335f;
            reco_reso[i] = nsig_res * standard_dev;

            if (found_min && found_max) {
                // Use RMS value if calculated
                float sumsq = 0.0f;
                float sumContents = 0.0f;
                
                for (int index = static_cast<int>(ibin_min_extrapol);
                     index <= static_cast<int>(ibin_max_extrapol);
                     index++) {
                    float contents = predict_clean[index];
                    float diff = (index + 0.5f - pred_PVs_loc[i]);
                    sumsq += diff * diff * contents;
                    sumContents += contents;
                }
                
                float local_rms = sumsq / sumContents;
                reco_reso[i] = nsig_res * local_rms;
            }
        }
    }
    
    return reco_reso;
}

std::tuple<int, int, int> compare_res_reco(
    std::vector<float> target_PVs_loc,  // Pass by value since we modify it
    const std::vector<float>& pred_PVs_loc,
    const std::vector<float>& reco_res,
    bool debug) {
    
    int succeed = 0;
    int missed = 0;
    int false_pos = 0;
    
    int len_pred_PVs_loc = pred_PVs_loc.size();
    int len_target_PVs_loc = target_PVs_loc.size();
    
    // Create modifiable copies
    std::vector<float> working_pred_pvs = pred_PVs_loc;
    std::vector<float> working_reco_res = reco_res;
    
    if (len_pred_PVs_loc >= len_target_PVs_loc) {
        if (debug) {
            std::cout << "In len(pred_PVs_loc) >= len(target_PVs_loc)" << std::endl;
        }
        
        for (int i = 0; i < len_pred_PVs_loc; i++) {
            if (debug) {
                std::cout << "pred_PVs_loc = " << pred_PVs_loc[i] << std::endl;
            }
            
            bool matched = false;
            float min_val = pred_PVs_loc[i] - reco_res[i];
            float max_val = pred_PVs_loc[i] + reco_res[i];
            
            if (debug) {
                std::cout << "resolution = " << (max_val - min_val)/2.0 << std::endl;
                std::cout << "min_val = " << min_val << std::endl;
                std::cout << "max_val = " << max_val << std::endl;
            }
            
            // Loop over remaining target PVs
            for (size_t j = 0; j < target_PVs_loc.size(); j++) {
                if (min_val <= target_PVs_loc[j] && target_PVs_loc[j] <= max_val) {
                    matched = true;
                    succeed++;
                    if (debug) {
                        std::cout << "succeed = " << succeed << std::endl;
                    }
                    target_PVs_loc.erase(target_PVs_loc.begin() + j);
                    break;
                }
            }
            
            if (!matched) {
                false_pos++;
                if (debug) {
                    std::cout << "false_pos = " << false_pos << std::endl;
                }
            }
        }
        
        missed = len_target_PVs_loc - succeed;
        if (debug) {
            std::cout << "missed = " << missed << std::endl;
        }
        
    } else {
        if (debug) {
            std::cout << "In len(pred_PVs_loc) < len(target_PVs_loc)" << std::endl;
        }
        
        for (int i = 0; i < len_target_PVs_loc; i++) {
            if (debug) {
                std::cout << "target_PVs_loc = " << target_PVs_loc[i] << std::endl;
            }
            
            bool matched = false;
            
            // Loop over remaining predicted PVs
            for (size_t j = 0; j < working_pred_pvs.size(); j++) {
                float min_val = working_pred_pvs[j] - working_reco_res[j];
                float max_val = working_pred_pvs[j] + working_reco_res[j];
                
                if (debug) {
                    std::cout << "pred_PVs_loc = " << working_pred_pvs[j] << std::endl;
                    std::cout << "resolution = " << (max_val - min_val)/2.0 << std::endl;
                    std::cout << "min_val = " << min_val << std::endl;
                    std::cout << "max_val = " << max_val << std::endl;
                }
                
                if (min_val <= target_PVs_loc[i] && target_PVs_loc[i] <= max_val) {
                    matched = true;
                    succeed++;
                    if (debug) {
                        std::cout << "succeed = " << succeed << std::endl;
                    }
                    
                    // Remove matched prediction
                    working_pred_pvs.erase(working_pred_pvs.begin() + j);
                    working_reco_res.erase(working_reco_res.begin() + j);
                    break;
                }
            }
            
            if (!matched) {
                missed++;
                if (debug) {
                    std::cout << "missed = " << missed << std::endl;
                }
            }
        }
        
        false_pos = len_pred_PVs_loc - succeed;
        if (debug) {
            std::cout << "false_pos = " << false_pos << std::endl;
        }
    }
    
    return std::make_tuple(succeed, missed, false_pos);
}

} // namespace pvfinder