#include "stdafx.h"
#include "common.h"
#include <algorithm>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <random>

using namespace cv;
using namespace std;

bool isInside(const Mat& img, int i, int j) {
    return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
}

// builds a disk structuring element of the given radius
// pixels inside the circle are 0, the rest is background (255)
// this is the shape used by the erosion below

Mat_<uchar> get_disk_strel(int radius) {

    int size = 2 * radius + 1;
    Mat_<uchar> strel(size, size);
    strel.setTo(255);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int dx = i - radius;
            int dy = j - radius;
            if (dx * dx + dy * dy <= radius * radius) {
                strel(i, j) = 0;
            }
        }
    }
    return strel;
}


// a pixel stays foreground only if every foreground cell of the
// structuring element also lands on foreground in the source
// shrinks blobs, here used to thicken the dark edge lines

Mat_<uchar> erosion(Mat_<uchar> src, Mat_<uchar> strel) {

    Mat_<uchar> dst(src.size());
    dst.setTo(255);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src(i, j) == 0) {
                int ok = 1;
                for (int u = 0; u < strel.rows; u++) {
                    for (int v = 0; v < strel.cols; v++) {
                        if (strel(u, v) == 0) {
                            int i2 = i + u - strel.rows / 2;
                            int j2 = j + v - strel.cols / 2;
                            if (!isInside(src, i2, j2) || src(i2, j2) != 0) {
                                ok = 0;
                                break;
                            }
                        }
                    }
                    if (!ok) break;
                }
                if (ok) {
                    dst(i, j) = 0;
                }
            }
        }
    }
    return dst;
}

// gives every separate foreground (0) blob its own integer label
// returns an image where each pixel holds the label of its blob (0 = background)

Mat_<int> component_labeling(Mat_<uchar> img) {
    
    Mat_<int> labels(img.size());
    labels.setTo(0);
    int label = 0;

    const int di8[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
    const int dj8[] = { -1, 0, 1, 1, 1, 0, -1, -1 };

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0 && labels(i, j) == 0) {
                label++;
                queue<pair<int, int>> q;
                labels(i, j) = label;
                q.push({ i, j });

                while (!q.empty()) {
                    auto p = q.front();
                    q.pop();
                    for (int k = 0; k < 8; k++) {
                        int ni = p.first + di8[k];
                        int nj = p.second + dj8[k];
                        if (isInside(img, ni, nj) && img(ni, nj) == 0 && labels(ni, nj) == 0) {
                            labels(ni, nj) = label;
                            q.push({ ni, nj });
                        }
                    }
                }
            }
        }
    }
    return labels;
}

// turns a label image into a color image for visualization
// each label gets a random (but fixed seed) color background stays white
// only used to display the debug labeled components window

Mat_<Vec3b> color_labels(const Mat_<int>& labels) {

    double min, max;
    minMaxLoc(labels, &min, &max);
    int num_labels = (int)max;

    default_random_engine gen(42);
    uniform_int_distribution<int> d(0, 255);

    vector<Vec3b> colors(num_labels + 1);
    colors[0] = Vec3b(255, 255, 255);
    for (int i = 1; i <= num_labels; i++) {
        colors[i] = Vec3b(d(gen), d(gen), d(gen));
    }

    Mat_<Vec3b> color_img(labels.size());
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            color_img(i, j) = colors[labels(i, j)];
        }
    }
    return color_img;
}

// generic 2D convolution: slides the kernel over the image and
// outputs the weighted sum at each pixel (pixels outside are skipped)

template <typename T>

Mat_<float> compute_convolution(const Mat_<T>& src, const Mat_<float>& kernel) {

    Mat_<float> dst(src.size());
    dst.setTo(0);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float sum = 0.0f;
            for (int u = 0; u < kernel.rows; u++) {
                for (int v = 0; v < kernel.cols; v++) {
                    int ni = i + u - kernel.rows / 2;
                    int nj = j + v - kernel.cols / 2;
                    if (isInside(src, ni, nj)) {
                        sum += kernel(u, v) * src(ni, nj);
                    }
                }
            }

            dst(i, j) = sum;
        }
    }
    return dst;
}

// gaussian

Mat_<uchar> apply_gaussian_1d(const Mat_<uchar>& src, int w) {

    float sigma = w / 6.0f;

    Mat_<float> G_row(1, w);
    Mat_<float> G_col(w, 1);
    float sum = 0.0f;

    for (int i = 0; i < w; i++) {
        float dx = i - w / 2;
        float val = exp(-(dx * dx) / (2.0f * sigma * sigma));
        G_row(0, i) = val;
        G_col(i, 0) = val;
        sum += val;
    }

    G_row /= sum;
    G_col /= sum;

    Mat_<float> temp = compute_convolution(src, G_row);
    Mat_<float> res = compute_convolution(temp, G_col);

    Mat_<uchar> dst;
    res.convertTo(dst, CV_8U);
    return dst;
}

// adaptive threshold , compares each pixel to its local (gaussian) mean
// minus a constant C, instead of one global threshold
// pixels darker than their neighborhood become 255 (the edges), rest 0

Mat_<uchar> my_adaptive_threshold(Mat_<uchar> src, int blockSize, int C) {

    // block size must be odd and at least 3
    if (blockSize % 2 == 0) blockSize++;

    if (blockSize < 3) blockSize = 3;

    // the local mean is just a gaussian blur of the image
    Mat_<uchar> local_mean = apply_gaussian_1d(src, blockSize);

    Mat_<uchar> dst(src.size(), (uchar)0);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // threshold = local mean minus a constant
            int T = (int)local_mean(i, j) - C;
            // darker than the neighborhood -> edge (255)
            dst(i, j) = (src(i, j) < T) ? 255 : 0;
        }
    }
    return dst;
}


// struct holding geometry and color statistics for one
// blob that might be a rubiks cube sticker

struct StickerCand {
    int label;                                  // the blobs integer id
    int area;                                   // number of pixels in the blob
    int r_min, r_max, c_min, c_max;             // bounding box
    double r_c, c_c;                            // center of mass
    long long sum_H, sum_S, sum_V;              // sums of HSV from the YUV normalized
    long long sum_raw_H, sum_raw_S, sum_raw_V;  // sums of HSV from the original
};

// used to remove lighting variations (shadows and highlights)
// where every pixel has normalized lightinng

Mat_<Vec3b> apply_yuv_normalization(const Mat_<Vec3b>& src) {

    Mat yuv;
    cvtColor(src, yuv, COLOR_BGR2YUV);
    Mat_<Vec3b> res = yuv.clone();

    for (int i = 0; i < yuv.rows; i++) {
        for (int j = 0; j < yuv.cols; j++) {
            Vec3b p = yuv.at<Vec3b>(i, j);

            double Y = p[0];
            double U = p[1];
            double V = p[2];

            // k scales the color toward/away from gray depending on brightness
            double k = 1.0;
            // bright pixel -> color down
            if (Y > 128.0) {
                k = 128.0 / Y;
            }


            // dark pixel -> boost the color a bit
            else if (Y < 128.0 && Y > 0) {
                k = (256.0 - Y) / 128.0;

            }

            // how far the color is from gray (128)
            double u_dist = abs(U - 128.0);
            double v_dist = abs(V - 128.0);

            // clamp k so no overflow
            double max_dist = max(u_dist, v_dist);
            if (max_dist * k > 127.0) {
                k = 127.0 / max_dist;
            }

            // scale the color channels around gray, keep brightness fixed at 128
            double U_prime = (U - 128.0) * k + 128.0;
            double V_prime = (V - 128.0) * k + 128.0;
            uchar u = (U_prime < 0) ? 0 : (U_prime > 255) ? 255 : static_cast<uchar>(U_prime);
            uchar v = (V_prime < 0) ? 0 : (V_prime > 255) ? 255 : static_cast<uchar>(V_prime);
            res(i, j) = Vec3b(128, u, v);
        }
    }

    Mat dst;
    cvtColor(res, dst, COLOR_YUV2BGR);
    return dst;
}

// converts a BGR image to HSV and splits it into 3 separate channels

void cube_rgb_to_hsv(const Mat_<Vec3b>& src, Mat_<uchar>& H, Mat_<uchar>& S, Mat_<uchar>& V) {

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV_FULL);

    vector<Mat> channels;
    split(hsv, channels);

    H = channels[0];
    S = channels[1];
    V = channels[2];
}

// rule based color classifier for one sticker
// takes the average H/S/V and returns a letter (W/Y/R/O/G/B, ? if unknown)
// low value -> unknown, low saturation -> white, otherwise decide by hue ranges

char classify_cube_color(double h_255, double s_255, double v_255) {

    // too dark to tell -> unknown
    if (v_255 < 25) return '?';

    // barely any color -> white sticker
    if (s_255 < 40) return 'W';

    // convert hue to the usual 0  360 degrees
    double h = h_255 * 360.0 / 255.0;

    // red wraps around both ends of the hue circle
    if (h < 15.0 || h >= 330.0) return 'R';
    if (h < 48.0) return 'O';
    // orange and yellow overlap, use saturation to split them
    if (h >= 48.0 && h < 55.0) {
        if (s_255 > 200) return 'O';
        else return 'Y';
    }
    if (h < 85.0)  return 'Y';
    if (h < 190.0) return 'G';

    // bluish range, but washed out blue looks white
    if (h < 260.0) {
        if (s_255 < 90) return 'W';
        return 'B';
    }

    // anything left over (purple/pink/magenta/etc) falls back to red
    return 'R';
}

// maps a color letter back to an actual BGR color
// used to paint the virtual 3x3 reconstruction of the face

Vec3b cube_color_to_bgr(char c) {

    switch (c) {
        case 'W': return Vec3b(255, 255, 255);
        case 'Y': return Vec3b(0, 255, 255);
        case 'R': return Vec3b(0, 0, 220);
        case 'O': return Vec3b(0, 140, 255);
        case 'G': return Vec3b(0, 200, 0);
        case 'B': return Vec3b(255, 0, 0);
    }

    return Vec3b(80, 80, 80);
}

int app_mode = 1;

// main pipeline for one frame or image:
// normalize lighting -> get HSV -> blur -> adaptive threshold (edges) ->
// thicken edges -> label blobs -> filter to sticker candidates ->
// pick the 9 that best form a 3x3 grid -> classify each color and draw
// both the detection overlay and the virtual reconstruction

void detect_rubiks_face(Mat_<Vec3b> img) {

    // resize to a manageable width so that the
    // chosen kernel sizes / area thresholds remain consistent
    if (img.cols > 640) {
        int new_h = (int)(img.rows * 640.0 / img.cols);     // keep aspect ratio
        resize(img, img, Size(640, new_h));
    }

    // YUV normalization
    Mat_<Vec3b> color_stable_img = apply_yuv_normalization(img);

    // HSV channels for both the normalized and the original
    Mat_<uchar> H, S, V;
    cube_rgb_to_hsv(color_stable_img, H, S, V);

    Mat_<uchar> raw_H, raw_S, raw_V;
    cube_rgb_to_hsv(img, raw_H, raw_S, raw_V);

    // gaussian blur 
    Mat_<uchar> blurred = apply_gaussian_1d(raw_V, 9);

    // adaptive threshold for edge detection
    Mat_<uchar> edges = my_adaptive_threshold(blurred, 15, 3);

    // thicken edge lines so small gaps are closed and each
    // sticker is enclosed by a continuous border
    Mat_<uchar> thick_edges = erosion(edges, get_disk_strel(2));

    // label every blob of the thickened edges
    Mat_<int> labels = component_labeling(thick_edges);

    // find how many labels we ended up with
    double mn, mx;
    minMaxLoc(labels, &mn, &mx);
    int N = (int)mx;

    // have a stats slot per label then accumulate in one
    // pass over the labels image
    // hold statistics of each blob

    vector<StickerCand> stats(N + 1);

    for (int l = 0; l <= N; l++) {
        stats[l].label = l;
        stats[l].area = 0;
        stats[l].r_min = INT_MAX; stats[l].r_max = INT_MIN;
        stats[l].c_min = INT_MAX; stats[l].c_max = INT_MIN;
        stats[l].r_c = 0;         stats[l].c_c = 0;
        stats[l].sum_H = 0; stats[l].sum_S = 0; stats[l].sum_V = 0;
        stats[l].sum_raw_H = 0; stats[l].sum_raw_S = 0; stats[l].sum_raw_V = 0;
    }

    // for each labeled pixel update its component stats
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int l = labels(i, j);
            if (l == 0) continue;
            StickerCand& s = stats[l];
            s.area++;
            s.r_c += i;
            s.c_c += j;

            if (i < s.r_min) s.r_min = i;
            if (i > s.r_max) s.r_max = i;
            if (j < s.c_min) s.c_min = j;
            if (j > s.c_max) s.c_max = j;

            // add the hsv of the pixel and later divide sums by the area
            // to see the avg color of the entire candiate
            s.sum_H += H(i, j);
            s.sum_S += S(i, j);
            s.sum_V += V(i, j);
            s.sum_raw_H += raw_H(i, j);
            s.sum_raw_S += raw_S(i, j);
            s.sum_raw_V += raw_V(i, j);
        }
    }

    // filter blobs by area and shape to get sticker candidates

    int total_pixels = img.rows * img.cols;
    int min_area = total_pixels / 1000;         // too small blobs are noise
    int max_area = total_pixels / 10;           // too large blobs are background

    vector<StickerCand> cands;

    for (int l = 1; l <= N; l++) {

        StickerCand& s = stats[l];

        if (s.area < min_area || s.area > max_area) continue;   // size filter

        // aspect ratio filter - stickers are ~ square so width/height ~ 1

        int w = s.c_max - s.c_min + 1;
        int h = s.r_max - s.r_min + 1;

        double ar = (double)w / h;
        if (ar < 0.6 || ar > 1.6) continue;

        // a sticker fills most of its bounding box

        double fill = (double)s.area / (w * h);
        if (fill < 0.5) continue;

        // centre of mass
        s.r_c /= s.area;
        s.c_c /= s.area;


        cands.push_back(s); // survived all filters
    }

    // choose the best 9 of the candidates that form a tight cluster
    // try every canditate and ask what are the 9 nearest blobs including iteself
    // how spread out are they and how uniform are their areas
    // the tuple with the lowest combined score wins

    vector<StickerCand> good;

    if (cands.size() >= 9) {

        double best_score = DBL_MAX;

        // treat each candidate as the possible center of the face
        for (int i = 0; i < cands.size(); i++) {

            // distance from this candidate to every other candidate
            vector<pair<double, StickerCand>> dists;
            for (int j = 0; j < cands.size(); j++) {
                double dist = hypot(cands[i].r_c - cands[j].r_c, cands[i].c_c - cands[j].c_c);
                dists.push_back({ dist, cands[j] });
            }

            // closest first, so the first 9 are this blob + its 8 nearest
            sort(dists.begin(), dists.end(), [](auto& a, auto& b) { return a.first < b.first; });

            vector<StickerCand> current_9;

            double max_dist = 0;
            double area_sum = 0;

            // bounding box of the 9 chosen blobs
            double min_r = DBL_MAX, max_r = -DBL_MAX;
            double min_c = DBL_MAX, max_c = -DBL_MAX;

            // gather the 9 nearest and their stats

            for (int k = 0; k < 9; k++) {

                current_9.push_back(dists[k].second);
                if (dists[k].first > max_dist) max_dist = dists[k].first;
                area_sum += dists[k].second.area;

                min_r = min(min_r, dists[k].second.r_c);
                max_r = max(max_r, dists[k].second.r_c);
                min_c = min(min_c, dists[k].second.c_c);
                max_c = max(max_c, dists[k].second.c_c);
            }

            // the 9 blobs should roughly form a square -> reject if not
            double grid_w = max_c - min_c;
            double grid_h = max_r - min_r;

            if (grid_h <= 0) continue;

            double grid_aspect_ratio = grid_w / grid_h;

            if (grid_aspect_ratio < 0.6 || grid_aspect_ratio > 1.6) continue;

            // reject if the blobs are spread too far apart for their size
            double mean_area = area_sum / 9.0;
            double expected_sticker_width = sqrt(mean_area);

            if (max_dist > expected_sticker_width * 6.0) continue;

            // total area difference from the mean (stickers should be similar)
            double area_var = 0;
            for (int k = 0; k < 9; k++) {
                area_var += abs(dists[k].second.area - mean_area);
            }

            // lower score = tighter and more uniform -> better candidate set
            double score = max_dist + (area_var / mean_area) * 100.0;
            if (score < best_score) {
                best_score = score;
                good = current_9;
            }
        }

        // even the best group is too messy -> probably no cube in view
        if (best_score > 600.0) {
            good.clear();
        }
    }

    Mat_<Vec3b> result = img.clone();

    if (good.size() == 9) {

        // order the 9 into reading order: first by row, then within each row by column
        sort(good.begin(), good.end(), [](const StickerCand& a, const StickerCand& b) { return a.r_c < b.r_c; });
        for (int row = 0; row < 3; row++) {
            sort(good.begin() + row * 3, good.begin() + row * 3 + 3, [](const StickerCand& a, const StickerCand& b) { return a.c_c < b.c_c; });
        }

        vector<char> face_colors;
        cout << "\n    Detected Rubik's cube face    \n";

        for (int k = 0; k < good.size(); k++) {

            StickerCand& s = good[k];

            // average HSV over the whole sticker (sums / area)
            double avg_h = (double)s.sum_H / s.area;
            double avg_s = (double)s.sum_S / s.area;
            double avg_v = (double)s.sum_V / s.area;

            // decide the color letter from that average
            char c = classify_cube_color(avg_h, avg_s, avg_v);
            face_colors.push_back(c);

            // draw the yellow box around the sticker
            rectangle(result, Point(s.c_min, s.r_min), Point(s.c_max, s.r_max), Scalar(0, 255, 255), 2);
            Point ctr((int)s.c_c, (int)s.r_c);

            // write the letter twice (thick black then thin white) so it reads on any background
            string txt(1, c);
            putText(result, txt, Point(ctr.x - 10, ctr.y + 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 4);
            putText(result, txt, Point(ctr.x - 10, ctr.y + 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1);
        }

        cout << "\n    Final 3x3 cube face    \n";
        for (int i = 0; i < 9; i++) {
            cout << face_colors[i] << "  ";
            if (i % 3 == 2) cout << "\n";
        }
        cout << "\n\n";

        // box that wraps all 9 stickers = the whole face
        Point pt_min(INT_MAX, INT_MAX), pt_max(INT_MIN, INT_MIN);
        for (auto& s : good) {
            if (s.c_min < pt_min.x) pt_min.x = s.c_min;
            if (s.r_min < pt_min.y) pt_min.y = s.r_min;
            if (s.c_max > pt_max.x) pt_max.x = s.c_max;
            if (s.r_max > pt_max.y) pt_max.y = s.r_max;
        }
        // green outline around the full face
        rectangle(result, Point(pt_min.x - 10, pt_min.y - 10), Point(pt_max.x + 10, pt_max.y + 10), Scalar(0, 255, 0), 3);

        // build the virtual 3x3 face, each cell is an 80px square
        int sq = 80;
        Mat_<Vec3b> face(sq * 3, sq * 3, Vec3b(30, 30, 30));
        for (int k = 0; k < 9; k++) {
            int row = k / 3, col = k % 3;
            // fill the cell with the detected color (small inset for the gaps)
            Vec3b bgr = cube_color_to_bgr(face_colors[k]);
            for (int y = row * sq + 4; y < (row + 1) * sq - 4; y++)
                for (int x = col * sq + 4; x < (col + 1) * sq - 4; x++)
                    face(y, x) = bgr;
        }
        imshow("Virtual reconstruction", face);
    }
    else {
        try { destroyWindow("Virtual reconstruction"); }
        catch (...) {}
    }

    if (app_mode == 2) {
        imshow("0 - Original image", img);
        imshow("1 - YUV normalized color", color_stable_img);
        imshow("3 - Gaussian filter", blurred);
        imshow("4 - Adaptive thresholding", edges);
        imshow("5 - Thickened edges", thick_edges);
        imshow("6 - Labeled components", color_labels(labels));
    }

    imshow("7 - Final detection", result);
}


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

    cout << "   Rubik's cube detector\n";
    cout << "\n";
    cout << "1. Use webcam\n";
    cout << "2. Choose an image\n";
    cout << "Enter your choice (1 or 2): ";

    cin >> app_mode;

    if (app_mode == 1) {
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            return -1;
        }

        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);

        cout << "\nRubiks cube face detector\n";
        cout << "ESC or q to quit\n\n";

        Mat frame;
        int fps_frames = 0;
        double fps_t0 = (double)getTickCount();
        double fps = 0.0;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            Mat_<Vec3b> img = frame;

            detect_rubiks_face(img);

            fps_frames++;
            double now = (double)getTickCount();
            double elapsed = (now - fps_t0) / getTickFrequency();
            if (elapsed >= 1.0) {
                fps = fps_frames / elapsed;
                fps_frames = 0;
                fps_t0 = now;
            }

            Mat_<Vec3b> hud = frame.clone();
            char fps_buf[64];
            sprintf(fps_buf, "FPS: %.1f", fps);
            putText(hud, fps_buf, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 4);
            putText(hud, fps_buf, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);
            imshow("Webcam", hud);

            int key = waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') break;
        }

        cap.release();
        destroyAllWindows();
    }
    else if (app_mode == 2) {
        char fname[MAX_PATH];

        while (openFileDlg(fname)) {
            Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);
            if (src.empty()) continue;

            detect_rubiks_face(src);

            waitKey(0);
            destroyAllWindows();
        }
    }
    else {

    }

    return 0;
}