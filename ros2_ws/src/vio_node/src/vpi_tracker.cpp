#include "vpi_tracker.hpp"

#if 0
#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  vpi_tracker.cpp  —  VPI 3.x  (corrected for actual installed API)
//  Key VPI 3 differences vs VPI 2:
//    • KLT takes VPIPyramid, not VPIImage — use vpiPyramidCreate()
//    • VPIOpticalFlowPyrLKParams: numIterations (not maxIterations), no nLevels
//    • Array type for keypoints: VPI_ARRAY_TYPE_KEYPOINT_F32
//      (some installs call it VPI_ARRAY_TYPE_KPT_F32 — we handle both via alias)
//    • Termination flags: VPI_TERMINATION_CRITERIA_EPSILON / _ITERATIONS
//    • OpenCV wrapper: vpiImageCreateWrapperOpenCVMat() (not ...OpenCVMatWrapper)
//    • Array size: read from vpiArrayGetSize(), not kp_data.buffer.aos.size
// ─────────────────────────────────────────────────────────────────────────────
#include <vpi/VPI.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Pyramid.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cmath>

// ── tunables ────────────────────────────────────────────────────────────────
static constexpr int   MAX_FEATURES   = 200;
static constexpr float HARRIS_K       = 0.04f;
static constexpr int   HARRIS_BLOCK   = 5;
static constexpr int   PYRAMID_LEVELS = 3;
static constexpr int   LK_WINDOW      = 21;
// ────────────────────────────────────────────────────────────────────────────

#define VPI_CHECK(expr) \
    do { \
        VPIStatus _s = (expr); \
        if (_s != VPI_SUCCESS) { \
            char _buf[256]; \
            vpiGetLastStatusMessage(_buf, sizeof(_buf)); \
            throw std::runtime_error(std::string("[VPI] ") + #expr + " → " + _buf); \
        } \
    } while (0)

struct TrackedPoint {
    float prev_x, prev_y;
    float curr_x, curr_y;
    bool  valid;
};
#endif

#include <vpi/VPI.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Pyramid.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
constexpr int kMaxFeatures = 240;
constexpr int kPyramidLevels = 3;
constexpr float kPyramidScale = 0.5f;
constexpr int kRedetectInterval = 20;

#define VPI_CHECK(expr) \
    do { \
        VPIStatus status = (expr); \
        if (status != VPI_SUCCESS) { \
            char msg[256]; \
            vpiGetLastStatusMessage(msg, sizeof(msg)); \
            throw std::runtime_error(std::string("[VPI] ") + #expr + ": " + msg); \
        } \
    } while (0)

cv::Mat to_gray8(const cv::Mat& input) {
    if (input.empty()) return {};
    if (input.type() == CV_8UC1) return input.isContinuous() ? input : input.clone();

    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    } else {
        input.convertTo(gray, CV_8U);
    }
    return gray;
}
}

struct VPIFeatureTracker::Impl {
    VPIStream stream = nullptr;
    VPIPayload harris_payload = nullptr;
    VPIPayload klt_payload = nullptr;
    VPIArray prev_pts = nullptr;
    VPIArray curr_pts = nullptr;
    VPIArray scores = nullptr;
    VPIArray status = nullptr;
    VPIPyramid prev_pyr = nullptr;
    VPIPyramid curr_pyr = nullptr;
    VPIImage prev_img = nullptr;
    VPIImage curr_img = nullptr;
    VPIHarrisCornerDetectorParams harris_params {};
    VPIOpticalFlowPyrLKParams klt_params {};
    std::vector<VPIKeypointF32> prev_host;
    int width = 0;
    int height = 0;
    int redetect_counter = 0;
    bool initialized = false;
};

VPIFeatureTracker::VPIFeatureTracker() : impl_(new Impl()) {
    VPI_CHECK(vpiStreamCreate(VPI_BACKEND_CUDA, &impl_->stream));
    VPI_CHECK(vpiInitHarrisCornerDetectorParams(&impl_->harris_params));
    impl_->harris_params.gradientSize = 5;
    impl_->harris_params.blockSize = 5;
    impl_->harris_params.sensitivity = 0.04f;
    impl_->harris_params.strengthThresh = 18.0f;
    impl_->harris_params.minNMSDistance = 8.0f;

    VPI_CHECK(vpiInitOpticalFlowPyrLKParams(&impl_->klt_params));
    impl_->klt_params.useInitialFlow = 0;
    impl_->klt_params.termination =
        VPI_TERMINATION_CRITERIA_ITERATIONS | VPI_TERMINATION_CRITERIA_EPSILON;
    impl_->klt_params.numIterations = 30;
    impl_->klt_params.windowDimension = 21;
    impl_->klt_params.epsilon = 0.01f;
    impl_->prev_host.resize(kMaxFeatures);
}

VPIFeatureTracker::~VPIFeatureTracker() {
    if (!impl_) return;
    if (impl_->klt_payload) vpiPayloadDestroy(impl_->klt_payload);
    if (impl_->harris_payload) vpiPayloadDestroy(impl_->harris_payload);
    if (impl_->prev_pts) vpiArrayDestroy(impl_->prev_pts);
    if (impl_->curr_pts) vpiArrayDestroy(impl_->curr_pts);
    if (impl_->scores) vpiArrayDestroy(impl_->scores);
    if (impl_->status) vpiArrayDestroy(impl_->status);
    if (impl_->prev_pyr) vpiPyramidDestroy(impl_->prev_pyr);
    if (impl_->curr_pyr) vpiPyramidDestroy(impl_->curr_pyr);
    if (impl_->prev_img) vpiImageDestroy(impl_->prev_img);
    if (impl_->curr_img) vpiImageDestroy(impl_->curr_img);
    if (impl_->stream) vpiStreamDestroy(impl_->stream);
    delete impl_;
}

static void init_buffers(VPIFeatureTracker::Impl& s, int width, int height) {
    s.width = width;
    s.height = height;
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA, &s.prev_img));
    VPI_CHECK(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA, &s.curr_img));
    VPI_CHECK(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, kPyramidLevels,
                               kPyramidScale, VPI_BACKEND_CUDA, &s.prev_pyr));
    VPI_CHECK(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, kPyramidLevels,
                               kPyramidScale, VPI_BACKEND_CUDA, &s.curr_pyr));
    VPI_CHECK(vpiArrayCreate(kMaxFeatures, VPI_ARRAY_TYPE_KEYPOINT_F32,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &s.prev_pts));
    VPI_CHECK(vpiArrayCreate(kMaxFeatures, VPI_ARRAY_TYPE_KEYPOINT_F32,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &s.curr_pts));
    VPI_CHECK(vpiArrayCreate(kMaxFeatures, VPI_ARRAY_TYPE_U32,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &s.scores));
    VPI_CHECK(vpiArrayCreate(kMaxFeatures, VPI_ARRAY_TYPE_U8,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &s.status));
    VPI_CHECK(vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, width, height, &s.harris_payload));
    VPI_CHECK(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA, width, height, VPI_IMAGE_FORMAT_U8,
                                        kPyramidLevels, kPyramidScale, &s.klt_payload));
}

static void submit_gray_to_pyramid(VPIFeatureTracker::Impl& s, const cv::Mat& gray,
                                   VPIImage image, VPIPyramid pyramid) {
    VPIImage wrapper = nullptr;
    VPI_CHECK(vpiImageCreateWrapperOpenCVMat(gray, VPI_IMAGE_FORMAT_U8, 0, &wrapper));
    VPI_CHECK(vpiSubmitConvertImageFormat(s.stream, VPI_BACKEND_CUDA, wrapper, image, nullptr));
    VPI_CHECK(vpiSubmitGaussianPyramidGenerator(
        s.stream, VPI_BACKEND_CUDA, image, pyramid, VPI_BORDER_CLAMP));
    VPI_CHECK(vpiStreamSync(s.stream));
    vpiImageDestroy(wrapper);
}

static int detect_harris(VPIFeatureTracker::Impl& s, VPIImage image) {
    VPI_CHECK(vpiSubmitHarrisCornerDetector(s.stream, VPI_BACKEND_CUDA, s.harris_payload,
                                            image, s.prev_pts, s.scores, &s.harris_params));
    VPI_CHECK(vpiStreamSync(s.stream));

    int32_t n = 0;
    VPI_CHECK(vpiArrayGetSize(s.prev_pts, &n));
    n = std::min<int32_t>(n, kMaxFeatures);

    VPIArrayData data;
    VPI_CHECK(vpiArrayLockData(s.prev_pts, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &data));
    auto* pts = reinterpret_cast<VPIKeypointF32*>(data.buffer.aos.data);
    for (int i = 0; i < n; ++i) {
        s.prev_host[i] = pts[i];
    }
    VPI_CHECK(vpiArrayUnlock(s.prev_pts));
    return n;
}

std::vector<TrackedPoint> VPIFeatureTracker::track(const cv::Mat& image) {
    cv::Mat gray = to_gray8(image);
    if (gray.empty()) {
        num_features_ = 0;
        return {};
    }

    if (!impl_->initialized) {
        init_buffers(*impl_, gray.cols, gray.rows);
        submit_gray_to_pyramid(*impl_, gray, impl_->prev_img, impl_->prev_pyr);
        num_features_ = detect_harris(*impl_, impl_->prev_img);
        impl_->initialized = true;
        std::vector<TrackedPoint> zero;
        zero.reserve(num_features_);
        for (int i = 0; i < num_features_; ++i) {
            zero.push_back({impl_->prev_host[i].x, impl_->prev_host[i].y,
                            impl_->prev_host[i].x, impl_->prev_host[i].y, true});
        }
        return zero;
    }

    if (gray.cols != impl_->width || gray.rows != impl_->height) {
        throw std::runtime_error("VPI tracker input size changed after initialization");
    }

    submit_gray_to_pyramid(*impl_, gray, impl_->curr_img, impl_->curr_pyr);
    VPI_CHECK(vpiSubmitOpticalFlowPyrLK(impl_->stream, VPI_BACKEND_CUDA, impl_->klt_payload,
                                        impl_->prev_pyr, impl_->curr_pyr, impl_->prev_pts,
                                        impl_->curr_pts, impl_->status, &impl_->klt_params));
    VPI_CHECK(vpiStreamSync(impl_->stream));

    int32_t n = 0;
    VPI_CHECK(vpiArrayGetSize(impl_->curr_pts, &n));
    n = std::min<int32_t>(n, num_features_);

    VPIArrayData curr_data;
    VPIArrayData status_data;
    VPI_CHECK(vpiArrayLockData(impl_->curr_pts, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &curr_data));
    VPI_CHECK(vpiArrayLockData(impl_->status, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &status_data));
    auto* curr = reinterpret_cast<VPIKeypointF32*>(curr_data.buffer.aos.data);
    auto* st = reinterpret_cast<uint8_t*>(status_data.buffer.aos.data);

    std::vector<TrackedPoint> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        const float dx = curr[i].x - impl_->prev_host[i].x;
        const float dy = curr[i].y - impl_->prev_host[i].y;
        const bool valid = st[i] == 0 && dx * dx + dy * dy < 80.0f * 80.0f &&
                           curr[i].x >= 0.0f && curr[i].y >= 0.0f &&
                           curr[i].x < static_cast<float>(impl_->width) &&
                           curr[i].y < static_cast<float>(impl_->height);
        result.push_back({impl_->prev_host[i].x, impl_->prev_host[i].y,
                          curr[i].x, curr[i].y, valid});
    }
    for (int i = 0; i < n; ++i) {
        impl_->prev_host[i] = curr[i];
    }
    VPI_CHECK(vpiArrayUnlock(impl_->status));
    VPI_CHECK(vpiArrayUnlock(impl_->curr_pts));

    std::swap(impl_->prev_pts, impl_->curr_pts);
    std::swap(impl_->prev_img, impl_->curr_img);
    std::swap(impl_->prev_pyr, impl_->curr_pyr);
    num_features_ = n;

    if (++impl_->redetect_counter >= kRedetectInterval || num_features_ < kMaxFeatures / 3) {
        num_features_ = detect_harris(*impl_, impl_->prev_img);
        impl_->redetect_counter = 0;
    }

    return result;
}

#if 0
class VPIFeatureTracker {
public:
    VPIFeatureTracker() {
        VPI_CHECK(vpiStreamCreate(VPI_BACKEND_CUDA, &stream_));

        // ── Harris params ─────────────────────────────────────────────────
        std::memset(&harris_params_, 0, sizeof(harris_params_));
        harris_params_.gradientSize   = HARRIS_BLOCK;
        harris_params_.blockSize      = HARRIS_BLOCK;
        harris_params_.sensitivity    = HARRIS_K;
        harris_params_.minNMSDistance = 8;
        harris_params_.strengthThresh = 20;

        // ── KLT params (VPI 3 struct) ─────────────────────────────────────
        // Field names confirmed from your error output:
        //   numIterations  (not maxIterations)
        //   useInitialFlow, windowDimension, epsilon are the same
        //   nLevels removed — pyramid depth set at vpiCreateOpticalFlowPyrLK()
        std::memset(&klt_params_, 0, sizeof(klt_params_));
        klt_params_.useInitialFlow  = 0;
        klt_params_.windowDimension = LK_WINDOW;
        klt_params_.numIterations   = 30;    // ← corrected field name
        klt_params_.epsilon         = 0.01f;

        initialised_      = false;
        num_features_     = 0;
        redetect_counter_ = 0;
    }

    ~VPIFeatureTracker() {
        if (klt_payload_)     vpiPayloadDestroy(klt_payload_);
        if (harris_payload_)  vpiPayloadDestroy(harris_payload_);
        if (keypoints_buf_)   vpiArrayDestroy(keypoints_buf_);
        if (tracking_status_) vpiArrayDestroy(tracking_status_);
        if (scores_buf_)      vpiArrayDestroy(scores_buf_);
        if (prev_pyr_)        vpiPyramidDestroy(prev_pyr_);
        if (curr_pyr_)        vpiPyramidDestroy(curr_pyr_);
        // Staging images (used to upload cv::Mat then copy to pyramid base)
        if (prev_img_)        vpiImageDestroy(prev_img_);
        if (curr_img_)        vpiImageDestroy(curr_img_);
        if (stream_)          vpiStreamDestroy(stream_);
    }

    // Returns tracked (prev→curr) point pairs.
    std::vector<TrackedPoint> track(const cv::Mat& left_bgr) {
        cv::Mat grey;
        cv::cvtColor(left_bgr, grey, cv::COLOR_BGR2GRAY);
        current_gray_ = grey;

        if (!initialised_) {
            _init_buffers(grey.cols, grey.rows);
            _upload_to_pyramid(grey, prev_img_, prev_pyr_);
            _detect_harris(prev_pyr_);
            vpiStreamSync(stream_);
            initialised_ = true;
            return _make_zero_tracks();
        }

        // Upload new frame into curr pyramid
        _upload_to_pyramid(grey, curr_img_, curr_pyr_);

        // ── KLT: track prev_pyr → curr_pyr ───────────────────────────────
        VPI_CHECK(vpiSubmitOpticalFlowPyrLK(
            stream_,
            VPI_BACKEND_CUDA,
            klt_payload_,
            prev_pyr_,          // ← VPIPyramid (prev)
            curr_pyr_,          // ← VPIPyramid (curr)
            keypoints_buf_,     // input: previous keypoint positions
            keypoints_buf_,     // output: updated keypoint positions (in-place)
            tracking_status_,   // per-point status
            &klt_params_
        ));
        vpiStreamSync(stream_);

        // ── Read back updated positions ───────────────────────────────────
        VPIArrayData kp_data;
        VPI_CHECK(vpiArrayLockData(keypoints_buf_,
                                   VPI_LOCK_READ,
                                   VPI_ARRAY_BUFFER_HOST_AOS,
                                   &kp_data));

        // VPI 3: get count via vpiArrayGetSize(), not kp_data.buffer.aos.size
        int32_t n_pts = 0;
        VPI_CHECK(vpiArrayGetSize(keypoints_buf_, &n_pts));
        if (n_pts > num_features_) n_pts = num_features_;

        auto* kps = reinterpret_cast<VPIKeypointF32*>(kp_data.buffer.aos.data);

        std::vector<TrackedPoint> result;
        result.reserve(n_pts);
        for (int i = 0; i < n_pts; ++i) {
            TrackedPoint tp;
            tp.prev_x = prev_keypoints_[i].x;
            tp.prev_y = prev_keypoints_[i].y;
            tp.curr_x = kps[i].x;
            tp.curr_y = kps[i].y;
            float dx  = tp.curr_x - tp.prev_x;
            float dy  = tp.curr_y - tp.prev_y;
            tp.valid  = (dx*dx + dy*dy) < (100.f * 100.f);
            result.push_back(tp);
        }

        // Save curr positions as "prev" for next call
        for (int i = 0; i < n_pts; ++i)
            prev_keypoints_[i] = kps[i];
        num_features_ = n_pts;

        VPI_CHECK(vpiArrayUnlock(keypoints_buf_));

        // Swap pyramid handles — curr becomes prev for next frame
        std::swap(prev_pyr_, curr_pyr_);
        std::swap(prev_img_, curr_img_);

        // Periodic re-detect to replace lost tracks
        if (++redetect_counter_ >= REDETECT_INTERVAL) {
            _detect_harris(prev_pyr_);
            vpiStreamSync(stream_);
            redetect_counter_ = 0;
        }

        return result;
    }

    int num_features() const { return num_features_; }

private:
    static constexpr int REDETECT_INTERVAL = 30;

    VPIStream   stream_        = nullptr;
    VPIPayload  harris_payload_= nullptr;
    VPIPayload  klt_payload_   = nullptr;
    VPIArray    keypoints_buf_ = nullptr;
    VPIArray    tracking_status_ = nullptr;
    VPIArray    scores_buf_    = nullptr;

    // VPI 3: pyramids are separate objects from images
    VPIPyramid  prev_pyr_      = nullptr;
    VPIPyramid  curr_pyr_      = nullptr;
    VPIImage    prev_img_      = nullptr;  // staging: cv::Mat → VPIImage → pyramid base
    VPIImage    curr_img_      = nullptr;

    VPIHarrisCornerDetectorParams harris_params_{};
    VPIOpticalFlowPyrLKParams     klt_params_{};

    std::vector<VPIKeypointF32> prev_keypoints_;
    cv::Mat current_gray_;
    int  num_features_     = 0;
    int  redetect_counter_ = 0;
    bool initialised_      = false;
    int  img_w_            = 0;
    int  img_h_            = 0;

    // ── Initialise all VPI objects (called once on first frame) ───────────
    void _init_buffers(int w, int h) {
        img_w_ = w;
        img_h_ = h;

        // Staging images (GRAY8, CUDA-backed)
        VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U8,
                                 VPI_BACKEND_CUDA, &prev_img_));
        VPI_CHECK(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U8,
                                 VPI_BACKEND_CUDA, &curr_img_));

        // Pyramids (GRAY8, PYRAMID_LEVELS levels)
        VPI_CHECK(vpiPyramidCreate(w, h, VPI_IMAGE_FORMAT_U8,
                                   PYRAMID_LEVELS, 0.5f,
                                   VPI_BACKEND_CUDA, &prev_pyr_));
        VPI_CHECK(vpiPyramidCreate(w, h, VPI_IMAGE_FORMAT_U8,
                                   PYRAMID_LEVELS, 0.5f,
                                   VPI_BACKEND_CUDA, &curr_pyr_));

        // VPI 3 keypoint array type.
        VPI_CHECK(vpiArrayCreate(MAX_FEATURES,
                                 VPI_ARRAY_TYPE_KEYPOINT_F32,
                                 VPI_BACKEND_CUDA | VPI_BACKEND_CPU,
                                 &keypoints_buf_));

        // Scores array (U32 for Harris response)
        VPI_CHECK(vpiArrayCreate(MAX_FEATURES,
                                 VPI_ARRAY_TYPE_U32,
                                 VPI_BACKEND_CUDA | VPI_BACKEND_CPU,
                                 &scores_buf_));

        VPI_CHECK(vpiArrayCreate(MAX_FEATURES,
                     VPI_ARRAY_TYPE_U8,
                     VPI_BACKEND_CUDA | VPI_BACKEND_CPU,
                     &tracking_status_));

        // Harris payload
        VPI_CHECK(vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA,
                                                w, h,
                                                &harris_payload_));

        // KLT payload — pyramid depth specified HERE in VPI 3 (not in params)
        VPI_CHECK(vpiCreateOpticalFlowPyrLK(
            VPI_BACKEND_CUDA,
            w, h,
            VPI_IMAGE_FORMAT_U8,
            PYRAMID_LEVELS,
            0.5f,
            &klt_payload_
        ));

        prev_keypoints_.resize(MAX_FEATURES);
    }

    // ── Upload a cv::Mat greyscale into a VPIImage, then build its pyramid ─
    void _upload_to_pyramid(const cv::Mat& grey, VPIImage& img, VPIPyramid& pyr) {
        // Wrap the cv::Mat memory as a VPIImage (host memory, no copy yet)
        VPIImage wrapper = nullptr;
        VPI_CHECK(vpiImageCreateWrapperOpenCVMat(grey, 0, &wrapper));

        // Copy host wrapper → CUDA-backed staging image
        VPI_CHECK(vpiSubmitConvertImageFormat(stream_,
                                              VPI_BACKEND_CUDA,
                                              wrapper,
                                              img,
                                              nullptr));
        vpiStreamSync(stream_);
        vpiImageDestroy(wrapper);  // wrapper doesn't own memory; safe to destroy

        // Build the Gaussian pyramid from the staging image base level
        VPI_CHECK(vpiSubmitGaussianPyramidGenerator(stream_,
                                                    VPI_BACKEND_CUDA,
                                                    img,
                                                    pyr,
                                                    VPI_BORDER_CLAMP));
        vpiStreamSync(stream_);
    }

    // ── Detect Harris corners on the base level of a pyramid ─────────────
    void _detect_harris(VPIPyramid& pyr) {
        (void)pyr;

        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(
            current_gray_,
            corners,
            MAX_FEATURES,
            0.01,
            8.0,
            cv::Mat(),
            3,
            false,
            0.04);

        const int32_t n = std::min<int32_t>(static_cast<int32_t>(corners.size()), MAX_FEATURES);

        // Write seed points into the VPI array.
        VPIArrayData kp_data;
        VPI_CHECK(vpiArrayLockData(keypoints_buf_,
                                   VPI_LOCK_WRITE,
                                   VPI_ARRAY_BUFFER_HOST_AOS,
                                   &kp_data));
        auto* kps = reinterpret_cast<VPIKeypointF32*>(kp_data.buffer.aos.data);
        for (int i = 0; i < n; ++i) {
            kps[i].x = corners[i].x;
            kps[i].y = corners[i].y;
        }
        VPI_CHECK(vpiArrayUnlock(keypoints_buf_));
        VPI_CHECK(vpiArraySetSize(keypoints_buf_, n));

        num_features_ = n;
        for (int i = 0; i < num_features_; ++i) {
            prev_keypoints_[i].x = corners[i].x;
            prev_keypoints_[i].y = corners[i].y;
        }
    }

    std::vector<TrackedPoint> _make_zero_tracks() {
        std::vector<TrackedPoint> result;
        for (int i = 0; i < num_features_; ++i) {
            TrackedPoint tp;
            tp.prev_x = tp.curr_x = prev_keypoints_[i].x;
            tp.prev_y = tp.curr_y = prev_keypoints_[i].y;
            tp.valid  = true;
            result.push_back(tp);
        }
        return result;
    }
};
#endif
