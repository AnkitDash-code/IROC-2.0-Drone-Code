#pragma once

#include <opencv2/core.hpp>

#include <vector>

struct TrackedPoint {
    float prev_x = 0.0f;
    float prev_y = 0.0f;
    float curr_x = 0.0f;
    float curr_y = 0.0f;
    bool valid = false;
};

class VPIFeatureTracker {
public:
    VPIFeatureTracker();
    ~VPIFeatureTracker();

    VPIFeatureTracker(const VPIFeatureTracker&) = delete;
    VPIFeatureTracker& operator=(const VPIFeatureTracker&) = delete;

    std::vector<TrackedPoint> track(const cv::Mat& gray_or_bgr);
    int num_features() const { return num_features_; }

public:
    struct Impl;
    Impl* impl_ = nullptr;
    int num_features_ = 0;
};
