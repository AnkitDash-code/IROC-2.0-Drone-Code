#include <iostream>
#include <mutex>
#include <map>
#include <atomic>

#ifdef WIN32
#include <io.h>
#include <windows.h>
#include <imm.h>
#include <direct.h>
#include "dbghelp.h"
#pragma comment(lib, "dbghelp.lib")
#else
#include <thread>
#include <unistd.h>
#include <sys/stat.h>
#endif
#include <string.h>
#include <chrono>
#include "SYSDKInterface.h"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "SYDataDefine.h"
#include <string>
#include <iostream>

#if WIN32
// #define USE_FRAME_OBSERVER
#endif

#ifdef WIN32
std::atomic_bool g_is_start(false);
std::atomic_bool g_bRefreshFPS(false);
#else
std::atomic_bool g_is_start(false);
std::atomic_bool g_bBreak(false);
std::atomic_bool g_bRefreshFPS(false);
#endif
std::map<unsigned int, int> g_mapFPS;
std::map<unsigned int, int> g_mapFrameCount;
std::map<unsigned int, Synexens::SYStreamType> g_mapStreamType;

std::map<unsigned int, bool> g_mapSavePCL;         // 存点云
std::map<unsigned int, bool> g_mapSaveDataAll;     // 有啥存啥
std::map<unsigned int, bool> g_mapShowConterValue; // 显示中心点位置

double g_last_time = 0;
std::thread fpsThread;
// unsigned int nT1 = 0;
// unsigned int nT2 = 0;;

#ifndef WIN32
void sprintf_s(char *const str, size_t const leng, char const *const _Format, unsigned int id)
{
    sprintf(str, _Format, id);
}
#endif

void calculate_framerate()
{
    while (g_is_start)
    {
        double cur_time = cv::getTickCount() / cv::getTickFrequency() * 1000;

        if (cur_time - g_last_time >= 1000)
        {
            // printf("===============> cur_time:%lf \n", cur_time);
            g_bRefreshFPS = true;
            g_last_time = cur_time;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
// 鼠标事件
static cv::Point cur_mouse_point = {-1, -1};
void on_mouse(int event, int x, int y, int flags, void *ustc)
{

    if (event == cv::EVENT_LBUTTONDOWN || (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)))
    {

        printf("x:%d y:%d\n", x, y);
        cur_mouse_point = cv::Point(x, y);
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        cur_mouse_point = cv::Point(-1, -1);
    }
}

// 检测目录，创建目录
std::string createCatalogue(std::string filePathIn)
{
#ifdef WIN32
    char *pFilePath = _getcwd(NULL, 0);
#else
    char *pFilePath = getcwd(NULL, 0);
#endif
    std::string strFileName = "";
    if (pFilePath != nullptr)
    {
        strFileName = pFilePath;
    }

#ifdef WIN32
    strFileName += "\\" + filePathIn + "\\";
    int ret = _mkdir(strFileName.c_str());
#else
    strFileName += "/" + filePathIn + "/";
    int ret = mkdir(strFileName.c_str(), S_IRWXU);
#endif
    return strFileName;
}

// 保存raw
void DumpRaw(unsigned int nDeviceID, uint8_t *saveData, Synexens::SYFrameType streamType, int w, int h)
{
    std::string sTyepName = "";
    if (streamType == Synexens::SYFRAMETYPE_RAW)
        sTyepName = "raw";
    else if (streamType == Synexens::SYFRAMETYPE_DEPTH)
        sTyepName = "depth";
    else if (streamType == Synexens::SYFRAMETYPE_IR)
        sTyepName = "ir";

    std::string savePath = createCatalogue(sTyepName);

    std::string pre = std::to_string(nDeviceID + 1) + "_" + std::to_string(w) + "x" + std::to_string(h) + "-" + sTyepName + "_" + std::to_string(time(0)) + ".raw";
    std::string filePath = savePath + pre;

    FILE *fp = fopen(filePath.c_str(), "wb+");

    if (fp != NULL)
    {
        fwrite(saveData, sizeof(uint16_t), w * h, fp);
        fflush(fp);
        fclose(fp);
    }
}

// 判断cv窗口
void IsImgWindow(char *cWindowName, cv::Mat imgData)
{
#ifdef WIN32
    HWND *hwnd = (HWND *)(cvGetWindowHandle(cWindowName));
    if (hwnd != nullptr)
    {
        cv::imshow(cWindowName, imgData);
    }
#else
    cv::imshow(cWindowName, imgData);
#endif
}

void ProcessFrameData(unsigned int nDeviceID, Synexens::SYFrameData *pFrameData = nullptr)
{
    auto itStreamFind = g_mapStreamType.find(nDeviceID);
    if (itStreamFind == g_mapStreamType.end())
    {
        return;
    }
    if (itStreamFind->second == Synexens::SYSTREAMTYPE_RGBD)
    {
        std::map<Synexens::SYFrameType, int> mapIndex;
        std::map<Synexens::SYFrameType, int> mapPos;
        int nPos = 0;
        // Set mapping direction to true DepthtoRGB, false RGBtoDepth
        bool bTOFtoRGB = true;
        for (int nFrameIndex = 0; nFrameIndex < pFrameData->m_nFrameCount; nFrameIndex++)
        {
            mapIndex.insert(std::pair<Synexens::SYFrameType, int>(pFrameData->m_pFrameInfo[nFrameIndex].m_frameType, nFrameIndex));
            mapPos.insert(std::pair<Synexens::SYFrameType, int>(pFrameData->m_pFrameInfo[nFrameIndex].m_frameType, nPos));
            nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
        }
        auto itDepthIndex = mapIndex.find(Synexens::SYFRAMETYPE_DEPTH);
        auto itRGBIndex = mapIndex.find(Synexens::SYFRAMETYPE_RGB);

        int nRGBDWidth = 0;
        int nRGBDHeight = 0;
        if(bTOFtoRGB)
        {
            nRGBDWidth = pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameWidth;
            nRGBDHeight = pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameHeight;
        }
        else
        {
            nRGBDWidth = pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameWidth;
            nRGBDHeight = pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameHeight;
        }
        unsigned short *pRGBDDepth = new unsigned short[nRGBDWidth * nRGBDHeight];
        memset(pRGBDDepth, 0, sizeof(unsigned short) * nRGBDWidth * nRGBDHeight);
        unsigned char *pRGBDRGB = new unsigned char[nRGBDWidth * nRGBDHeight * 3];
        memset(pRGBDRGB, 0, sizeof(unsigned char) * nRGBDWidth * nRGBDHeight * 3);
        if (itDepthIndex != mapIndex.end() && itRGBIndex != mapIndex.end())
        {
            if (Synexens::GetRGBD(nDeviceID, pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameWidth, pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameHeight, (unsigned short *)pFrameData->m_pData + mapPos[Synexens::SYFRAMETYPE_DEPTH],
                                  pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameWidth, pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameHeight, (unsigned char *)pFrameData->m_pData + mapPos[Synexens::SYFRAMETYPE_RGB],
                                  nRGBDWidth, nRGBDHeight, pRGBDDepth, pRGBDRGB,bTOFtoRGB) == Synexens::SYERRORCODE_SUCCESS)
            {
                g_mapFrameCount[nDeviceID]++;

                // Depth
                cv::Mat gray16(nRGBDHeight, nRGBDWidth, CV_16UC1, pRGBDDepth);
                int nCount = nRGBDHeight * nRGBDWidth;
                unsigned char *pColor = new unsigned char[nCount * 3];
                cv::Mat rgbimg = cv::Mat(nRGBDHeight, nRGBDWidth, CV_8UC3);
                if (Synexens::GetDepthColor(nDeviceID, nCount, pRGBDDepth, pColor) == Synexens::SYERRORCODE_SUCCESS)
                {
                    memcpy(rgbimg.data, pColor, nCount * 3);
                }
                else
                {
                    cv::Mat tmp;
                    cv::Mat gray8 = cv::Mat(gray16.size(), CV_8U);
                    cv::normalize(gray16, tmp, 0, 255, cv::NORM_MINMAX);
                    cv::convertScaleAbs(tmp, gray8);
                    cv::cvtColor(gray8, rgbimg, cv::COLOR_GRAY2RGB);
                }
                delete[] pColor;

                cv::Mat image_rgbd;
                cvtColor(rgbimg, image_rgbd, cv::COLOR_BGR2RGB);

                std::string msg = std::to_string(nRGBDHeight) + "x" + std::to_string(nRGBDWidth) + " fps:" + std::to_string(g_mapFPS[nDeviceID]);
                int font_face = cv::FONT_HERSHEY_COMPLEX;
                double font_scale = 1;
                int thickness = 2;
                int baseline;
                cv::Size text_size = cv::getTextSize(msg, font_face, font_scale, thickness, &baseline);

                cv::Point origin;
                origin.x = rgbimg.cols / 2 - text_size.width / 2;
                origin.y = 0 + text_size.height;
                cv::putText(image_rgbd, msg, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);

                char cTemp[16];
                sprintf_s(cTemp, 16, "RGBD_depth_%d", nDeviceID);
                IsImgWindow(cTemp, image_rgbd);

                // RGB
                cv::Mat bgrImgRGB(nRGBDHeight, nRGBDWidth, CV_8UC3, pRGBDRGB);

                cv::Mat image_rgb;
                cvtColor(bgrImgRGB, image_rgb, cv::COLOR_BGR2RGB);
                sprintf_s(cTemp, 16, "RGBD_RGB_%d", nDeviceID);
                IsImgWindow(cTemp, image_rgb);
            }
        }
        delete[] pRGBDDepth;
        delete[] pRGBDRGB;
    }
    else if (itStreamFind->second == Synexens::SYSTREAMTYPE_RGBDAI)
    {
        std::map<Synexens::SYFrameType, int> mapIndex;
        std::map<Synexens::SYFrameType, int> mapPos;
        int nPos = 0;
        for (int nFrameIndex = 0; nFrameIndex < pFrameData->m_nFrameCount; nFrameIndex++)
        {
            if (pFrameData->m_pFrameInfo[nFrameIndex].m_frameType > 10 || pFrameData->m_pFrameInfo[nFrameIndex].m_frameType < 0)
            {
                std::cout << "Type:" << pFrameData->m_pFrameInfo[nFrameIndex].m_frameType << std::endl;
                return;
            }
            mapIndex.insert(std::pair<Synexens::SYFrameType, int>(pFrameData->m_pFrameInfo[nFrameIndex].m_frameType, nFrameIndex));

            switch (pFrameData->m_pFrameInfo[nFrameIndex].m_frameType)
            {
            case Synexens::SYFRAMETYPE_DEPTH:
            {
                mapPos.insert(std::pair<Synexens::SYFrameType, int>(Synexens::SYFRAMETYPE_DEPTH, nPos));
                nPos += (pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short));
            }
            break;
            case Synexens::SYFRAMETYPE_RGB:
            {
                mapPos.insert(std::pair<Synexens::SYFrameType, int>(Synexens::SYFRAMETYPE_RGB, nPos));
                nPos += (pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short));
            }
            break;
            case Synexens::SYFRAMETYPE_AI:
            {
                mapPos.insert(std::pair<Synexens::SYFrameType, int>(Synexens::SYFRAMETYPE_AI, nPos));
                nPos += sizeof(Synexens::detect_result_group_t);
            }
            break;
            }
        }

        auto itDepthIndex = mapIndex.find(Synexens::SYFRAMETYPE_DEPTH);
        auto itRGBIndex = mapIndex.find(Synexens::SYFRAMETYPE_RGB);
        int nRGBDWidth = pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameWidth;
        int nRGBDHeight = pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameHeight;
        unsigned short *pRGBDDepth = new unsigned short[nRGBDWidth * nRGBDHeight];
        memset(pRGBDDepth, 0, sizeof(unsigned short) * nRGBDWidth * nRGBDHeight);
        unsigned char *pRGBDRGB = new unsigned char[nRGBDWidth * nRGBDHeight * 3];
        memset(pRGBDRGB, 0, sizeof(unsigned char) * nRGBDWidth * nRGBDHeight * 3);
        if (itDepthIndex != mapIndex.end() && itRGBIndex != mapIndex.end())
        {
            if (Synexens::GetRGBD(nDeviceID, pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameWidth, pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameHeight, (unsigned short *)((unsigned char *)pFrameData->m_pData + mapPos[Synexens::SYFRAMETYPE_DEPTH]),
                                  pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameWidth, pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameHeight, (unsigned char *)pFrameData->m_pData + mapPos[Synexens::SYFRAMETYPE_RGB],
                                  nRGBDWidth, nRGBDHeight, pRGBDDepth, pRGBDRGB) == Synexens::SYERRORCODE_SUCCESS)
            {
                g_mapFrameCount[nDeviceID]++;

                // Depth
                cv::Mat gray16(nRGBDHeight, nRGBDWidth, CV_16UC1, pRGBDDepth);
                int nCount = nRGBDHeight * nRGBDWidth;
                unsigned char *pColor = new unsigned char[nCount * 3];
                cv::Mat rgbimg = cv::Mat(nRGBDHeight, nRGBDWidth, CV_8UC3);
                if (Synexens::GetDepthColor(nDeviceID, nCount, pRGBDDepth, pColor) == Synexens::SYERRORCODE_SUCCESS)
                {
                    memcpy(rgbimg.data, pColor, nCount * 3);
                }
                else
                {
                    cv::Mat tmp;
                    cv::Mat gray8 = cv::Mat(gray16.size(), CV_8U);
                    cv::normalize(gray16, tmp, 0, 255, cv::NORM_MINMAX);
                    cv::convertScaleAbs(tmp, gray8);
                    cv::cvtColor(gray8, rgbimg, cv::COLOR_GRAY2RGB);
                }
                delete[] pColor;

                std::string msg = std::to_string(nRGBDWidth) + "x" + std::to_string(nRGBDHeight) + " fps:" + std::to_string(g_mapFPS[nDeviceID]);
                int font_face = cv::FONT_HERSHEY_COMPLEX;
                double font_scale = 1;
                int thickness = 2;
                int baseline;
                cv::Size text_size = cv::getTextSize(msg, font_face, font_scale, thickness, &baseline);

                cv::Point origin;
                origin.x = rgbimg.cols / 2 - text_size.width / 2;
                origin.y = 0 + text_size.height;
                cv::putText(rgbimg, msg, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);

                cv::Mat image_rgbd;
                cvtColor(rgbimg, image_rgbd, cv::COLOR_BGR2RGB);

                char cTemp[16];
                sprintf_s(cTemp, 16, "RGBDAI_depth_%d", nDeviceID);
                IsImgWindow(cTemp, image_rgbd);

                // RGB
                cv::Mat bgrImgRGB(nRGBDHeight, nRGBDWidth, CV_8UC3, pRGBDRGB);

                cv::Mat image_rgb;
                cvtColor(bgrImgRGB, image_rgb, cv::COLOR_BGR2RGB);
                // std::cout << "AI POS: " << std::to_string(mapPos[Synexens::SYFRAMETYPE_AI]);

                Synexens::detect_result_group_t *detect_result_group = reinterpret_cast<Synexens::detect_result_group_t *>((unsigned char *)pFrameData->m_pData + mapPos[Synexens::SYFRAMETYPE_AI]);
                for (int i = 0; i < detect_result_group->count; i++)
                {
                    Synexens::detect_result_t *det_result = &(detect_result_group->results[i]);
                    // printf("%s @ (%d %d %d %d) %f\n",
                    //     det_result->name,
                    //     det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                    //     det_result->prop);
                    int x1 = det_result->box.left;
                    int y1 = det_result->box.top;
                    int x2 = det_result->box.right;
                    int y2 = det_result->box.bottom;

                    int centerX = (x2 + x1) / 2;
                    int centerY = (y2 + y1) / 2;

                    int depth = pRGBDDepth[centerY * nRGBDWidth + centerX];
                    rectangle(image_rgb, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
                    putText(image_rgb, det_result->name, cv::Point(x1, y1 + 12), 1, 2, cv::Scalar(0, 255, 0, 255));

                    std::string text = "Avg Depth: " + std::to_string(depth);
                    putText(image_rgb, text, cv::Point(x1, y1 + 37), 1, 2, cv::Scalar(0, 255, 0, 255));
                }
                sprintf_s(cTemp, 16, "RGBDAI_RGB_%d", nDeviceID);
                IsImgWindow(cTemp, image_rgb);
            }
        }
        delete[] pRGBDDepth;
        delete[] pRGBDRGB;
    }
    else
    {
        int nPos = 0;
        for (int nFrameIndex = 0; nFrameIndex < pFrameData->m_nFrameCount; nFrameIndex++)
        {
            switch (pFrameData->m_pFrameInfo[nFrameIndex].m_frameType)
            {
            case Synexens::SYFRAMETYPE_RAW:
            {
                g_mapFrameCount[nDeviceID]++;
                cv::Mat gray16(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_16UC1, (unsigned char *)pFrameData->m_pData + nPos);
                cv::Mat tmp;
                cv::Mat gray8 = cv::Mat(gray16.size(), CV_8U);
                cv::normalize(gray16, tmp, 0, 255, cv::NORM_MINMAX);
                cv::convertScaleAbs(tmp, gray8);
                cv::Mat rgbimg = cv::Mat(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_8UC3);
                cv::cvtColor(gray8, rgbimg, cv::COLOR_GRAY2RGB);

                std::string msg = std::to_string(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth) + "x" + std::to_string(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight) + " fps:" + std::to_string(g_mapFPS[nDeviceID]);
                int font_face = cv::FONT_HERSHEY_COMPLEX;
                double font_scale = 1;
                int thickness = 2;
                int baseline;
                cv::Size text_size = cv::getTextSize(msg, font_face, font_scale, thickness, &baseline);

                cv::Point origin;
                origin.x = rgbimg.cols / 2 - text_size.width / 2;
                origin.y = 0 + text_size.height;
                cv::putText(rgbimg, msg, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);

                char cTemp[16];
                sprintf_s(cTemp, 16, "raw_%d", nDeviceID);
                IsImgWindow(cTemp, rgbimg);

                //==== dump data start ====//
                auto itSaveDataAll = g_mapSaveDataAll.find(nDeviceID);
                if (itSaveDataAll != g_mapSaveDataAll.end())
                {
                    if (itSaveDataAll->second)
                    {
                        DumpRaw(nDeviceID, (unsigned char *)pFrameData->m_pData + nPos, Synexens::SYFRAMETYPE_RAW, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight);
                    }
                }
                //==== dump data end ====//

                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
                break;
            }
            case Synexens::SYFRAMETYPE_DEPTH:
            {
                g_mapFrameCount[nDeviceID]++;
                cv::Mat gray16(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_16UC1, (unsigned char *)pFrameData->m_pData + nPos);
                int nCount = pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth;
                unsigned char *pColor = new unsigned char[nCount * 3];
                cv::Mat rgbimg = cv::Mat(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_8UC3);
                cv::Mat bgrimg = cv::Mat(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_8UC3);
                if (Synexens::GetDepthColor(nDeviceID, nCount, (unsigned short *)((unsigned char *)pFrameData->m_pData + nPos), pColor) == Synexens::SYERRORCODE_SUCCESS)
                {
                    memcpy(rgbimg.data, pColor, nCount * 3);
                    cv::cvtColor(rgbimg, bgrimg, cv::ColorConversionCodes::COLOR_RGB2BGR);

                    auto itSavePCL = g_mapSavePCL.find(nDeviceID);
                    if (itSavePCL != g_mapSavePCL.end())
                    {
                        if (itSavePCL->second)
                        {
                            Synexens::SYPointCloudData *pPCLData = new Synexens::SYPointCloudData[nCount];

                            if (Synexens::GetDepthPointCloud(nDeviceID, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, (unsigned short *)((unsigned char *)pFrameData->m_pData + nPos), pPCLData) == Synexens::SYERRORCODE_SUCCESS)
                            {
                                std::string sTyepName = "pcldata";
                                std::string savePath = createCatalogue(sTyepName);

                                std::string filename = savePath + std::to_string(nDeviceID) + "PointCloudData-" + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".pcd";
                                std::cout << sTyepName << "==" << savePath << "==" << filename << std::endl;

                                FILE *fp = nullptr;
#ifdef WIN32
                                fopen_s(&fp, filename.c_str(), "wb");
#else
                                fp = fopen(filename.c_str(), "wb");
#endif

                                fprintf(fp, "# .PCD v0.7 - Point Cloud Data file format\n");
                                fprintf(fp, "VERSION 0.7\n");
                                fprintf(fp, "FIELDS x y z rgb\n");
                                fprintf(fp, "SIZE 4 4 4 4\n");
                                fprintf(fp, "TYPE F F F U\n");
                                fprintf(fp, "COUNT 1 1 1 1\n");
                                fprintf(fp, "WIDTH  %d\n", nCount);
                                fprintf(fp, "HEIGHT 1\n");
                                fprintf(fp, "VIEWPOINT 0 0 0 1 0 0 0\n");
                                fprintf(fp, "POINTS %d\n", nCount);
                                fprintf(fp, "DATA ascii\n");

                                for (int n = 0; n < nCount; n++)
                                {
                                    unsigned char cTemp[4] = {};
                                    cTemp[1] = pColor[n * 3];
                                    cTemp[2] = pColor[n * 3 + 1];
                                    cTemp[3] = pColor[n * 3 + 2];
                                    unsigned int nTemp;
                                    memcpy(&nTemp, cTemp, 4);
                                    fprintf(fp, "%f %f %f %d\n", pPCLData[n].m_fltX, pPCLData[n].m_fltY, pPCLData[n].m_fltZ, nTemp);
                                }
                                fclose(fp);
                            }
                            delete[] pPCLData;
                        }
                    }
                }
                else
                {
                    cv::Mat tmp;
                    cv::Mat gray8 = cv::Mat(gray16.size(), CV_8U);
                    cv::normalize(gray16, tmp, 0, 255, cv::NORM_MINMAX);
                    cv::convertScaleAbs(tmp, gray8);
                    cv::cvtColor(gray8, rgbimg, cv::COLOR_GRAY2RGB);
                }
                delete[] pColor;

                // 显示中心点位置信息
                auto itShowConter = g_mapShowConterValue.find(nDeviceID);
                if (itShowConter != g_mapShowConterValue.end())
                {
                    if (itShowConter->second)
                    {
                        int centerIndex = (pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * (pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight) / 2 - 1) + (pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * 2 - 1);

                        uint16_t centerValue = 0;
                        memcpy(&centerValue, (unsigned char *)pFrameData->m_pData + centerIndex, sizeof(uint16_t));
                        cv::circle(bgrimg, cv::Point(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth / 2, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight / 2), 5, cv::Scalar(0, 255, 255), 2);

                        // std::string msg = "";
                        cv::putText(bgrimg, std::to_string(centerValue), cv::Point((pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth / 2) + 5, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight / 2), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2, 2, 0);
                    }
                }

                std::string msg = std::to_string(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth) + "x" + std::to_string(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight) + " fps:" + std::to_string(g_mapFPS[nDeviceID]);
                int font_face = cv::FONT_HERSHEY_COMPLEX;
                double font_scale = 1;
                int thickness = 2;
                int baseline;
                cv::Size text_size = cv::getTextSize(msg, font_face, font_scale, thickness, &baseline);

                cv::Point origin;
                origin.x = rgbimg.cols / 2 - text_size.width / 2;
                origin.y = 0 + text_size.height;
                cv::putText(bgrimg, msg, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 2, 0);

                if (cur_mouse_point.x > 0 && cur_mouse_point.y > 0 && cur_mouse_point.x < pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth && cur_mouse_point.y < pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight)
                {
                    char temp[48];
                    int width = pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth;
                    int height = pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight;
                    auto depth = (unsigned char *)pFrameData->m_pData + nPos;
                    sprintf(temp, " (x:%d,y:%d,d:%d)", cur_mouse_point.x, cur_mouse_point.y, ((unsigned short *)depth)[cur_mouse_point.y * width + cur_mouse_point.x]);

                    int local_x = cur_mouse_point.x;
                    if (local_x > height - 10)
                        local_x = height - 10;
                    cv::putText(bgrimg, temp, cv::Point(local_x, cur_mouse_point.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 1, 1);

                    circle(bgrimg, cv::Point(cur_mouse_point.x, cur_mouse_point.y), 3, cv::Scalar(255, 0, 0), 0);
                }

                char cTemp[16];
                sprintf_s(cTemp, 16, "depth_%d", nDeviceID);
                IsImgWindow(cTemp, bgrimg);

                //==== dump data start ====//
                auto itSaveDataAll = g_mapSaveDataAll.find(nDeviceID);
                if (itSaveDataAll != g_mapSaveDataAll.end())
                {
                    if (itSaveDataAll->second)
                    {
                        DumpRaw(nDeviceID, (unsigned char *)pFrameData->m_pData + nPos, Synexens::SYFRAMETYPE_DEPTH, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight);
                    }
                }
                //==== dump data end ====//

                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
                break;
            }
            case Synexens::SYFRAMETYPE_IR:
            {
                cv::Mat gray16(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_16UC1, (unsigned char *)pFrameData->m_pData + nPos);
                cv::Size nSize = gray16.size();
                cv::Mat gray8 = cv::Mat(gray16.size(), CV_8U);

                for (int i = 0; i < pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth; i++)
                {
                    int nTemp = gray16.at<uint16_t>(i / pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, i % pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth) / 2;
                    if (nTemp > 255)
                        nTemp = 255;

                    gray8.at<uint8_t>(i / pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, i % pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth) = nTemp;
                }

                char cTemp[16];
                sprintf_s(cTemp, 16, "ir_%d", nDeviceID);
                IsImgWindow(cTemp, gray8);

                //==== dump data start ====//
                auto itSaveDataAll = g_mapSaveDataAll.find(nDeviceID);
                if (itSaveDataAll != g_mapSaveDataAll.end())
                {
                    if (itSaveDataAll->second)
                    {
                        DumpRaw(nDeviceID, (unsigned char *)pFrameData->m_pData + nPos, Synexens::SYFRAMETYPE_IR, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight);
                    }
                }
                //==== dump data end ====//

                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
                break;
            }
            case Synexens::SYFRAMETYPE_RGB:
            {
                cv::Mat yuvImg(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_8UC2, (unsigned char *)pFrameData->m_pData + nPos);
                cv::Mat rgbImg;
                cv::Mat bgrImg;
                rgbImg = cv::Mat(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_8UC3);
                bgrImg = cv::Mat(pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight, pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth, CV_8UC3);
                cv::cvtColor(yuvImg, rgbImg, cv::ColorConversionCodes::COLOR_YUV2RGB_YUYV);
                cv::cvtColor(rgbImg, bgrImg, cv::ColorConversionCodes::COLOR_RGB2BGR);

                char cTemp[16];
                sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
                IsImgWindow(cTemp, bgrImg);
                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * 3 / 2;
                break;
            }
            }
        }
    }
}

void CreateOpencvWindow(unsigned int nDeviceID, Synexens::SYStreamType streamType, bool bDestoryOld = false)
{
    char cTemp[16];
    switch (streamType)
    {
    case Synexens::SYSTREAMTYPE_NULL:
        break;
    case Synexens::SYSTREAMTYPE_RAW:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "RGBDAI_depth_%d", nDeviceID);
            cv::destroyWindow(cTemp);
            sprintf_s(cTemp, 16, "RGBDAI_RGB_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "raw_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;
    case Synexens::SYSTREAMTYPE_DEPTH:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "raw_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "depth_%d", nDeviceID);
        cv::namedWindow(cTemp);
        cv::setMouseCallback(cTemp, on_mouse, 0);
        break;
    case Synexens::SYSTREAMTYPE_RGB:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "depth_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;
    case Synexens::SYSTREAMTYPE_DEPTHIR:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "depth_%d", nDeviceID);
        cv::namedWindow(cTemp);
        cv::setMouseCallback(cTemp, on_mouse, 0);
        sprintf_s(cTemp, 16, "ir_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;
    case Synexens::SYSTREAMTYPE_DEPTHRGB:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "ir_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "depth_%d", nDeviceID);
        cv::namedWindow(cTemp);
        sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;
    case Synexens::SYSTREAMTYPE_DEPTHIRRGB:
        sprintf_s(cTemp, 16, "depth_%d", nDeviceID);
        cv::namedWindow(cTemp);
        sprintf_s(cTemp, 16, "ir_%d", nDeviceID);
        cv::namedWindow(cTemp);
        sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;
    case Synexens::SYSTREAMTYPE_RGBD:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "depth_%d", nDeviceID);
            cv::destroyWindow(cTemp);
            sprintf_s(cTemp, 16, "ir_%d", nDeviceID);
            cv::destroyWindow(cTemp);
            sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "RGBD_depth_%d", nDeviceID);
        cv::namedWindow(cTemp);
        sprintf_s(cTemp, 16, "RGBD_RGB_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;
    case Synexens::SYSTREAMTYPE_RAWRGB:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "RGBD_depth_%d", nDeviceID);
            cv::destroyWindow(cTemp);
            sprintf_s(cTemp, 16, "RGBD_RGB_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "raw_%d", nDeviceID);
        cv::namedWindow(cTemp);
        sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;
    case Synexens::SYSTREAMTYPE_RGBDAI:
        if (bDestoryOld)
        {
            sprintf_s(cTemp, 16, "raw_%d", nDeviceID);
            cv::destroyWindow(cTemp);
            sprintf_s(cTemp, 16, "RGB_%d", nDeviceID);
            cv::destroyWindow(cTemp);
        }
        sprintf_s(cTemp, 16, "RGBDAI_depth_%d", nDeviceID);
        cv::namedWindow(cTemp);
        sprintf_s(cTemp, 16, "RGBDAI_RGB_%d", nDeviceID);
        cv::namedWindow(cTemp);
        break;

    default:
        break;
    }
}

class FrameObserver : public Synexens::ISYFrameObserver
{
public:
    virtual void OnFrameNotify(unsigned int nDeviceID, Synexens::SYFrameData *pFrameData = nullptr)
    {
        // int nT3 = GetTickCount();
        ProcessFrameData(nDeviceID, pFrameData);
        // int nT4 = GetTickCount();
        // printf("T3 delta %d\n", nT4 - nT3);
    }
};

FrameObserver g_FrameObserver;

void PrintErrorCode(std::string strFunc, Synexens::SYErrorCode errorCode)
{
    printf("%s errorcode:%d\n", strFunc.c_str(), errorCode);
}

int main()
{
    printf("SynexensSDK Yaml Test Demo\n");
    int nSDKVersionLength = 0;
    Synexens::SYErrorCode errorCodeGetSDKVersion = Synexens::GetSDKVersion(nSDKVersionLength, nullptr);
    if (errorCodeGetSDKVersion == Synexens::SYERRORCODE_SUCCESS)
    {
        if (nSDKVersionLength > 0)
        {
            char *pStringSDKVersion = new char[nSDKVersionLength];
            errorCodeGetSDKVersion = Synexens::GetSDKVersion(nSDKVersionLength, pStringSDKVersion);
            if (errorCodeGetSDKVersion == Synexens::SYERRORCODE_SUCCESS)
            {
                printf("SDKVersion:%s\n", pStringSDKVersion);
            }
            else
            {
                PrintErrorCode("GetSDKVersion2", errorCodeGetSDKVersion);
            }
            delete[] pStringSDKVersion;
        }
    }
    else
    {
        PrintErrorCode("GetSDKVersion", errorCodeGetSDKVersion);
    }
    Synexens::SYErrorCode errorCodeInitSDK = Synexens::InitSDK();
    if (errorCodeInitSDK != Synexens::SYERRORCODE_SUCCESS)
    {
        PrintErrorCode("InitSDK", errorCodeInitSDK);
    }

#ifdef USE_FRAME_OBSERVER
    Synexens::SYErrorCode errorCodeRegisterFrameObserver = Synexens::RegisterFrameObserver(&g_FrameObserver);
    if (errorCodeRegisterFrameObserver != Synexens::SYERRORCODE_SUCCESS)
    {
        PrintErrorCode("RegisterFrameObserver", errorCodeInitSDK);
    }
#endif // USE_FRAME_OBSERVER

    int nCount = 0;
    Synexens::SYErrorCode errorCode = Synexens::FindDevice(nCount);
    if (errorCode == Synexens::SYERRORCODE_SUCCESS && nCount > 0)
    {
        Synexens::SYDeviceInfo *pDeviceInfo = new Synexens::SYDeviceInfo[nCount];
        errorCode = Synexens::FindDevice(nCount, pDeviceInfo);
        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
        {
            bool *pOpen = new bool[nCount];
            memset(pOpen, 0, sizeof(bool) * nCount);
            int *pIntegralTimeMin = new int[nCount];
            int *pIntegralTimeMax = new int[nCount];
            int *pIntegralTime = new int[nCount];

            for (int i = 0; i < nCount; i++)
            {
                g_mapSavePCL.insert(std::pair<unsigned int, bool>(pDeviceInfo[i].m_nDeviceID, false));
                g_mapSaveDataAll.insert(std::pair<unsigned int, bool>(pDeviceInfo[i].m_nDeviceID, false));
                g_mapShowConterValue.insert(std::pair<unsigned int, int>(pDeviceInfo[i].m_nDeviceID, false));

                g_mapFrameCount.insert(std::pair<unsigned int, int>(pDeviceInfo[i].m_nDeviceID, 0));
                g_mapFPS.insert(std::pair<unsigned int, int>(pDeviceInfo[i].m_nDeviceID, 0));

                // g_mapSaveCount.insert(std::pair<unsigned int, unsigned int>(pDeviceInfo[i].m_nDeviceID, 0));
                Synexens::SYErrorCode errorCodeOpenDevice = Synexens::OpenDevice(pDeviceInfo[i]);
                if (errorCodeOpenDevice == Synexens::SYERRORCODE_SUCCESS)
                {
                    int nStringLength = 0;
                    Synexens::SYErrorCode errorCodeGetSN = Synexens::GetDeviceSN(pDeviceInfo[i].m_nDeviceID, nStringLength, nullptr);
                    if (errorCodeGetSN == Synexens::SYERRORCODE_SUCCESS)
                    {
                        if (nStringLength > 0)
                        {
                            char *pStringSN = new char[nStringLength];
                            errorCodeGetSN = Synexens::GetDeviceSN(pDeviceInfo[i].m_nDeviceID, nStringLength, pStringSN);
                            if (errorCodeGetSN == Synexens::SYERRORCODE_SUCCESS)
                            {
                                printf("SN%d:%s\n", i, pStringSN);
                            }
                            else
                            {
                                PrintErrorCode("GetDeviceSN", errorCodeGetSN);
                            }
                            delete[] pStringSN;
                        }
                    }
                    else
                    {
                        PrintErrorCode("GetDeviceSN", errorCodeGetSN);
                    }

                    nStringLength = 0;
                    Synexens::SYErrorCode errorCodeGetHWVersion = Synexens::GetDeviceHWVersion(pDeviceInfo[i].m_nDeviceID, nStringLength, nullptr);
                    if (errorCodeGetHWVersion == Synexens::SYERRORCODE_SUCCESS)
                    {
                        if (nStringLength > 0)
                        {
                            char *pStringFWVersion = new char[nStringLength];
                            errorCodeGetHWVersion = Synexens::GetDeviceHWVersion(pDeviceInfo[i].m_nDeviceID, nStringLength, pStringFWVersion);
                            if (errorCodeGetHWVersion == Synexens::SYERRORCODE_SUCCESS)
                            {
                                printf("HWVersion%d:%s\n", i, pStringFWVersion);
                            }
                            else
                            {
                                PrintErrorCode("GetDeviceHWVersion2", errorCodeGetHWVersion);
                            }
                            delete[] pStringFWVersion;
                        }
                    }
                    else
                    {
                        PrintErrorCode("GetDeviceHWVersion", errorCodeGetHWVersion);
                    }
                    int nSupportTypeCount = 0;
                    Synexens::SYErrorCode errorCodeQueryFrameType = Synexens::QueryDeviceSupportFrameType(pDeviceInfo[i].m_nDeviceID, nSupportTypeCount);
                    if (errorCodeQueryFrameType == Synexens::SYERRORCODE_SUCCESS && nSupportTypeCount > 0)
                    {
                        Synexens::SYSupportType *pSupportType = new Synexens::SYSupportType[nSupportTypeCount];
                        errorCodeQueryFrameType = Synexens::QueryDeviceSupportFrameType(pDeviceInfo[i].m_nDeviceID, nSupportTypeCount, pSupportType);
                        if (errorCodeQueryFrameType == Synexens::SYERRORCODE_SUCCESS && nSupportTypeCount > 0)
                        {
                            for (int j = 0; j < nSupportTypeCount; j++)
                            {
                                printf("FrameType%d:%d\n", j, pSupportType[j]);
                                int nResolutionCount = 0;
                                Synexens::SYErrorCode errorCodeQueryResolution = Synexens::QueryDeviceSupportResolution(pDeviceInfo[i].m_nDeviceID, pSupportType[j], nResolutionCount);
                                if (errorCodeQueryResolution == Synexens::SYERRORCODE_SUCCESS && nResolutionCount > 0)
                                {
                                    Synexens::SYResolution *pResolution = new Synexens::SYResolution[nResolutionCount];
                                    errorCodeQueryResolution = Synexens::QueryDeviceSupportResolution(pDeviceInfo[i].m_nDeviceID, pSupportType[j], nResolutionCount, pResolution);
                                    if (errorCodeQueryResolution == Synexens::SYERRORCODE_SUCCESS && nResolutionCount > 0)
                                    {
                                        for (int k = 0; k < nResolutionCount; k++)
                                        {
                                            printf("FrameType%d:%d,Resolution%d:%d\n", j, pSupportType[j], k, pResolution[k]);
                                        }
                                    }
                                    else
                                    {
                                        PrintErrorCode("QueryDeviceSupportResolution2", errorCodeQueryResolution);
                                    }
                                    delete[] pResolution;
                                }
                                else
                                {
                                    PrintErrorCode("QueryDeviceSupportResolution", errorCodeQueryResolution);
                                }
                            }
                        }
                        else
                        {
                            PrintErrorCode("QueryDeviceSupportFrameType2", errorCodeQueryFrameType);
                        }
                        delete[] pSupportType;
                    }
                    else
                    {
                        PrintErrorCode("QueryDeviceSupportFrameType", errorCodeQueryFrameType);
                    }
                }
                else
                {
                    PrintErrorCode("OpenDevice", errorCodeOpenDevice);
                }

                // Import Yaml configuration and open stream
#ifdef WIN32
                char *pYamlConfigBasePath = _getcwd(NULL, 0);
                std::string sYamlConfigName = "\\YamlConfig\\default.yaml";
#else
                char *pYamlConfigBasePath = getcwd(NULL, 0);
                std::string sYamlConfigName = "/YamlConfig/default.yaml";
#endif

                if (pYamlConfigBasePath != nullptr)
                {
                    /* ======
                    Different configurations can be set here based on different SN or device IDs.
                    Demo only shows loading default Yaml configuration.
                    ====== */
                    std::string sYamlConfigBasePath = pYamlConfigBasePath;
                    std::string sYamlConfigPath = sYamlConfigBasePath + sYamlConfigName;
                    Synexens::SYConfiguration configuration;
                    if (Synexens::ParseConfiguration(sYamlConfigPath.length(), sYamlConfigPath.c_str(), configuration) == Synexens::SYERRORCODE_SUCCESS)
                    {
                        Synexens::SYErrorCode errorcodeImport = Synexens::StartStreamingWithConfiguration(pDeviceInfo[i].m_nDeviceID, configuration);

                        if (errorcodeImport == Synexens::SYERRORCODE_SUCCESS)
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[i].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                itStreamFind->second = configuration.m_streamType;
                            }
                            else
                            {
                                g_mapStreamType.insert(std::pair<unsigned int, Synexens::SYStreamType>(pDeviceInfo[i].m_nDeviceID, configuration.m_streamType));
                            }

                            pOpen[i] = true;
                            CreateOpencvWindow(pDeviceInfo[i].m_nDeviceID, configuration.m_streamType);

                            printf("Import configuration success DeviceID:%d \n",pDeviceInfo[i].m_nDeviceID);
                        }
                        else
                        {
                            PrintErrorCode("Import configuration", errorcodeImport);
                        }
                    }
                    else
                    {
                        PrintErrorCode("get default config ERROR", Synexens::SYERRORCODE_FAILED);
                    }
                }
                else
                {
                    PrintErrorCode("Get Yaml Config Path Error", Synexens::SYERRORCODE_FAILED);
                }
            }

// #define SET_INTEGRAL_TIME 1
#if SET_INTEGRAL_TIME
            for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
            {
                int nIntegralTime = 2000;
                Synexens::SYErrorCode errorCodeIntegralTime = Synexens::SetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, nIntegralTime);
                if (errorCodeIntegralTime == Synexens::SYERRORCODE_SUCCESS)
                {
                    printf("SetIntegralTime success, nIntegralTime = %d\n", nIntegralTime);
                }
            }
#endif
            // ================== while开始监控键盘按键 ================== //
            g_is_start = true;
            fpsThread = std::thread(calculate_framerate);
            while (true)
            {
                bool bBreak = false;
                if (g_bRefreshFPS)
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        g_mapFPS[pDeviceInfo[nDeviceIndex].m_nDeviceID] = g_mapFrameCount[pDeviceInfo[nDeviceIndex].m_nDeviceID];
                        g_mapFrameCount[pDeviceInfo[nDeviceIndex].m_nDeviceID] = 0;
                    }
                    g_bRefreshFPS = false;
                }
#ifndef USE_FRAME_OBSERVER
                for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                {
                    if (pOpen[nDeviceIndex])
                    {
                        // nT1 = GetTickCount();
                        // printf("T1 delta %d\n", nT1-nT2);
                        Synexens::SYFrameData *pLastFrameData = nullptr;
                        Synexens::SYErrorCode errorCodeLastFrame = Synexens::GetLastFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, pLastFrameData);
                        // std::shared_ptr<Synexens::SYFrameData> pLastFrameData = nullptr;
                        //                  Synexens::SYErrorCode errorCodeLastFrame = Synexens::GetLastFrameDataSafety(pDeviceInfo[nDeviceIndex].m_nDeviceID, pLastFrameData);
                        // nT2 = GetTickCount();
                        // printf("T2 delta %d\n", nT2 - nT1);
                        if (errorCodeLastFrame == Synexens::SYERRORCODE_SUCCESS)
                        {
                            // int nT3 = GetTickCount();
                            ProcessFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, pLastFrameData);
                            // int nT4 = GetTickCount();
                            // printf("T3 delta %d\n", nT4 - nT3);
                            // ProcessFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, pLastFrameData.get());
                        }
                        else
                        {
                            // PrintErrorCode("GetLastFrameData", errorCode);
                        }
                    }
                }
#endif
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

                switch (cv::waitKey(5))
                {
                case 27: // ESC
                {
                    bBreak = true;
                    break;
                }
                case 's': // S 切换数据流类型
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        switch (pDeviceInfo[nDeviceIndex].m_deviceType)
                        {
                        case Synexens::SYDEVICETYPE_CS30_SINGLE:
                        case Synexens::SYDEVICETYPE_CS30_DUAL:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                bool bSuccess = true;
                                switch (itStreamFind->second)
                                {
                                case Synexens::SYSTREAMTYPE_RGBD:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_RAW;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_RAW, Synexens::SYRESOLUTION_640_480);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                default:
                                {
                                    itStreamFind->second = (Synexens::SYStreamType)(itStreamFind->second + 1);
                                    if (itStreamFind->second == Synexens::SYSTREAMTYPE_RGBD)
                                    {
                                        Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                                        {
                                            errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_RGB, Synexens::SYRESOLUTION_1920_1080);
                                            if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                            {
                                                bSuccess = false;
                                                PrintErrorCode("SetFrameResolution RGB", errorCode);
                                            }
                                        }
                                        else
                                        {
                                            bSuccess = false;
                                            PrintErrorCode("SetFrameResolution Depth", errorCode);
                                        }
                                    }
                                    break;
                                }
                                }
                                if (bSuccess)
                                {
                                    Synexens::SYErrorCode errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                    if (errorCodeChangeResolution == Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        CreateOpencvWindow(nDeviceIndex, itStreamFind->second, true);
                                    }
                                    else
                                    {
                                        PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                    }
                                }
                            }
                            break;
                        }
                        case Synexens::SYDEVICETYPE_CS20_DUAL:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                bool bSuccess = true;
                                switch (itStreamFind->second)
                                {
                                case Synexens::SYSTREAMTYPE_DEPTHIR:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_RAW;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_RAW, Synexens::SYRESOLUTION_640_480);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                case Synexens::SYSTREAMTYPE_RAW:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_DEPTH;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                case Synexens::SYSTREAMTYPE_DEPTH:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_DEPTHIR;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                }
                                if (bSuccess)
                                {
                                    Synexens::SYErrorCode errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                    if (errorCodeChangeResolution == Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        CreateOpencvWindow(nDeviceIndex, itStreamFind->second, true);
                                    }
                                    else
                                    {
                                        PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                    }
                                }
                            }
                            break;
                        }
                        case Synexens::SYDEVICETYPE_CS40:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                bool bSuccess = true;
                                switch (itStreamFind->second)
                                {
                                case Synexens::SYSTREAMTYPE_DEPTHIR:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_DEPTH;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                case Synexens::SYSTREAMTYPE_DEPTH:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_DEPTHIR;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                }
                                if (bSuccess)
                                {
                                    Synexens::SYErrorCode errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                    if (errorCodeChangeResolution == Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        CreateOpencvWindow(nDeviceIndex, itStreamFind->second, true);
                                    }
                                    else
                                    {
                                        PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                    }
                                }
                            }
                            break;
                        }
                        case Synexens::SYDEVICETYPE_CS20_SINGLE:
                        case Synexens::SYDEVICETYPE_CS20_P:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                bool bSuccess = true;
                                switch (itStreamFind->second)
                                {
                                case Synexens::SYSTREAMTYPE_DEPTHIR:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_RAW;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_RAW, Synexens::SYRESOLUTION_320_240);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                case Synexens::SYSTREAMTYPE_RAW:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_DEPTH;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                case Synexens::SYSTREAMTYPE_DEPTH:
                                {
                                    itStreamFind->second = Synexens::SYSTREAMTYPE_DEPTHIR;
                                    Synexens::SYErrorCode errorCode = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);
                                    if (errorCode != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        bSuccess = false;
                                        PrintErrorCode("SetFrameResolution RAW", errorCode);
                                    }
                                    break;
                                }
                                }
                                if (bSuccess)
                                {
                                    Synexens::SYErrorCode errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                    if (errorCodeChangeResolution == Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        CreateOpencvWindow(nDeviceIndex, itStreamFind->second, true);
                                    }
                                    else
                                    {
                                        PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                    }
                                }
                            }
                            break;
                        }
                        }
                    }
                    break;
                }
                case 'l': // L 切换低分辨率
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        switch (pDeviceInfo[nDeviceIndex].m_deviceType)
                        {
                        case Synexens::SYDEVICETYPE_CS30_SINGLE:
                        case Synexens::SYDEVICETYPE_CS30_DUAL:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                if (itStreamFind->second != Synexens::SYSTREAMTYPE_RGBD)
                                {
                                    Synexens::SYErrorCode errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);
                                    if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        PrintErrorCode("ChangeFrameResolution Depth Low", errorCodeChangeResolution);
                                    }
                                    errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_RGB, Synexens::SYRESOLUTION_640_480);
                                    if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        PrintErrorCode("ChangeFrameResolution RGB Low", errorCodeChangeResolution);
                                    }
                                    errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                    if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                    }
                                }
                            }
                            break;
                        }
                        case Synexens::SYDEVICETYPE_CS20_DUAL:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                Synexens::SYErrorCode errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution Depth Low", errorCodeChangeResolution);
                                }
                                errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                }
                                break;
                            }
                        }
                        case Synexens::SYDEVICETYPE_CS40:
                        case Synexens::SYDEVICETYPE_CS40PRO:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                Synexens::SYErrorCode errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution Depth Low", errorCodeChangeResolution);
                                }
                                errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                }
                                break;
                            }
                        }
                        }
                    }
                    break;
                }
                case 'h': // H 切换高分辨率
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        switch (pDeviceInfo[nDeviceIndex].m_deviceType)
                        {
                        case Synexens::SYDEVICETYPE_CS30_SINGLE:
                        case Synexens::SYDEVICETYPE_CS30_DUAL:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                if (itStreamFind->second != Synexens::SYSTREAMTYPE_RGBD)
                                {
                                    Synexens::SYErrorCode errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                    if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        PrintErrorCode("ChangeFrameResolution Depth Low", errorCodeChangeResolution);
                                    }
                                    errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_RGB, Synexens::SYRESOLUTION_1920_1080);
                                    if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        PrintErrorCode("ChangeFrameResolution RGB Low", errorCodeChangeResolution);
                                    }
                                    errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                    if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                    {
                                        PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                    }
                                }
                            }
                            break;
                        }
                        case Synexens::SYDEVICETYPE_CS20_DUAL:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                Synexens::SYErrorCode errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution Depth Low", errorCodeChangeResolution);
                                }
                                errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                }
                                break;
                            }
                        }
                        case Synexens::SYDEVICETYPE_CS40:
                        case Synexens::SYDEVICETYPE_CS40PRO:
                        {
                            auto itStreamFind = g_mapStreamType.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                            if (itStreamFind != g_mapStreamType.end())
                            {
                                Synexens::SYErrorCode errorCodeChangeResolution = Synexens::SetFrameResolution(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution Depth High", errorCodeChangeResolution);
                                }
                                errorCodeChangeResolution = Synexens::ChangeStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID, itStreamFind->second);
                                if (errorCodeChangeResolution != Synexens::SYERRORCODE_SUCCESS)
                                {
                                    PrintErrorCode("ChangeFrameResolution ChangeStreaming", errorCodeChangeResolution);
                                }
                                break;
                            }
                        }
                        }
                    }
                    break;
                }
                case 'u': // U 提高积分时间
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        Synexens::SYErrorCode errorCodeIntegralTime = Synexens::GetIntegralTimeRange(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYRESOLUTION_640_480, pIntegralTimeMin[nDeviceIndex], pIntegralTimeMax[nDeviceIndex]);
                        if (errorCodeIntegralTime == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetIntegralTimeRange success, Min = %d, Max = %d\n", pIntegralTimeMin[nDeviceIndex], pIntegralTimeMax[nDeviceIndex]);
                        }
                        else
                        {
                            PrintErrorCode("GetIntegralTimeRange", errorCodeIntegralTime);
                        }
                        errorCodeIntegralTime = Synexens::GetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex]);
                        if (errorCodeIntegralTime == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetIntegralTime success, nIntegralTime = %d\n", pIntegralTime[nDeviceIndex]);
                        }
                        else
                        {
                            PrintErrorCode("GetIntegralTime", errorCodeIntegralTime);
                        }
                        if (pIntegralTime[nDeviceIndex] >= pIntegralTimeMax[nDeviceIndex])
                        {
                            printf("IntegralTime already be Max\n");
                        }
                        else
                        {
                            pIntegralTime[nDeviceIndex] += 100;
                            if (pIntegralTime[nDeviceIndex] >= pIntegralTimeMax[nDeviceIndex])
                                pIntegralTime[nDeviceIndex] = pIntegralTimeMax[nDeviceIndex];
                            Synexens::SYErrorCode errorCodeIntegralTime = Synexens::SetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex]);
                            if (errorCodeIntegralTime == Synexens::SYERRORCODE_SUCCESS)
                            {
                                printf("SetIntegralTime success, nIntegralTime = %d\n", pIntegralTime[nDeviceIndex]);
                            }
                            else
                            {
                                PrintErrorCode("SetIntegralTime", errorCodeIntegralTime);
                            }
                        }
                    }
                    break;
                }
                case 'd': // D 降低积分时间
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        Synexens::SYErrorCode errorCodeIntegralTime = Synexens::GetIntegralTimeRange(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYRESOLUTION_640_480, pIntegralTimeMin[nDeviceIndex], pIntegralTimeMax[nDeviceIndex]);
                        if (errorCodeIntegralTime == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetIntegralTimeRange success, Min = %d, Max = %d\n", pIntegralTimeMin[nDeviceIndex], pIntegralTimeMax[nDeviceIndex]);
                        }
                        else
                        {
                            PrintErrorCode("GetIntegralTimeRange", errorCodeIntegralTime);
                        }
                        errorCodeIntegralTime = Synexens::GetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex]);
                        if (errorCodeIntegralTime == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetIntegralTime success, nIntegralTime = %d\n", pIntegralTime[nDeviceIndex]);
                        }
                        else
                        {
                            PrintErrorCode("GetIntegralTime", errorCodeIntegralTime);
                        }
                        if (pIntegralTime[nDeviceIndex] <= pIntegralTimeMin[nDeviceIndex])
                        {
                            printf("IntegralTime already be Min\n");
                        }
                        else
                        {
                            pIntegralTime[nDeviceIndex] -= 100;
                            if (pIntegralTime[nDeviceIndex] <= pIntegralTimeMin[nDeviceIndex])
                                pIntegralTime[nDeviceIndex] = pIntegralTimeMin[nDeviceIndex];
                            Synexens::SYErrorCode errorCodeIntegralTime = Synexens::SetIntegralTime(pDeviceInfo[nDeviceIndex].m_nDeviceID, pIntegralTime[nDeviceIndex]);
                            if (errorCodeIntegralTime == Synexens::SYERRORCODE_SUCCESS)
                            {
                                printf("SetIntegralTime success, nIntegralTime = %d\n", pIntegralTime[nDeviceIndex]);
                            }
                            else
                            {
                                PrintErrorCode("SetIntegralTime", errorCodeIntegralTime);
                            }
                        }
                    }
                    break;
                }
                case 'f': // F Filter滤波
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        bool bFilter = false;
                        Synexens::SYErrorCode errorCodeFilter = Synexens::GetFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter);
                        if (errorCodeFilter == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetFilter success, bFilter = %d\n", bFilter);
                        }
                        else
                        {
                            PrintErrorCode("GetFilter", errorCodeFilter);
                        }
                        bFilter = !bFilter;
                        errorCodeFilter = Synexens::SetFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter);
                        if (errorCodeFilter == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("SetFilter success, bFilter = %d\n", bFilter);
                        }
                        else
                        {
                            PrintErrorCode("SetFilter", errorCodeFilter);
                        }
                    }
                    break;
                }
                case 'm': // M Mirror水平镜像
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        bool bMirror = false;
                        Synexens::SYErrorCode errorCodeMirror = Synexens::GetMirror(pDeviceInfo[nDeviceIndex].m_nDeviceID, bMirror);
                        if (errorCodeMirror == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetMirror success, bMirror = %d\n", bMirror);
                        }
                        else
                        {
                            PrintErrorCode("GetMirror", errorCodeMirror);
                        }
                        bMirror = !bMirror;
                        errorCodeMirror = Synexens::SetMirror(pDeviceInfo[nDeviceIndex].m_nDeviceID, bMirror);
                        if (errorCodeMirror == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("SetMirror success, bMirror = %d\n", bMirror);
                        }
                        else
                        {
                            PrintErrorCode("SetMirror", errorCodeMirror);
                        }
                    }
                    break;
                }
                case 'i': // i Flip垂直翻转
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        bool bFlip = false;
                        Synexens::SYErrorCode errorCodeFlip = Synexens::GetFlip(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFlip);
                        if (errorCodeFlip == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetFlip success, bFlip = %d\n", bFlip);
                        }
                        else
                        {
                            PrintErrorCode("GetFlip", errorCodeFlip);
                        }
                        bFlip = !bFlip;
                        errorCodeFlip = Synexens::SetFlip(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFlip);
                        if (errorCodeFlip == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("SetFlip success, bFlip = %d\n", bFlip);
                        }
                        else
                        {
                            PrintErrorCode("SetFlip", errorCodeFlip);
                        }
                    }
                    break;
                }
                case 'p': // p 存点云数据
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        auto itSavePCL = g_mapSavePCL.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                        if (itSavePCL != g_mapSavePCL.end())
                        {
                            itSavePCL->second = !itSavePCL->second;
                        }
                    }
                    break;
                }
                case 'v': // 存数据
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        auto itSaveAll = g_mapSaveDataAll.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                        if (itSaveAll != g_mapSaveDataAll.end())
                        {
                            itSaveAll->second = !itSaveAll->second;
                        }
                    }
                    break;
                }
                case 'a': // 增加滤波
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        Synexens::SYErrorCode errorCodeAddFilter = Synexens::AddFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFILTERTYPE_SPECKLE);
                        if (errorCodeAddFilter == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("AddFilter success\n");
                        }
                        else
                        {
                            PrintErrorCode("AddFilter", errorCodeAddFilter);
                        }
                    }
                    break;
                }
                case 't': // T TrailFilter去拖影滤波
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        bool bFilter = false;
                        Synexens::SYErrorCode errorCodeFilter = Synexens::GetTrailFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter);
                        if (errorCodeFilter == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetTrailFilter success, bFilter = %d\n", bFilter);
                        }
                        else
                        {
                            PrintErrorCode("GetTrailFilter", errorCodeFilter);
                        }
                        bFilter = !bFilter;
                        errorCodeFilter = Synexens::SetTrailFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, bFilter);
                        if (errorCodeFilter == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("SetTrailFilter success, bFilter = %d\n", bFilter);
                        }
                        else
                        {
                            PrintErrorCode("SetTrailFilter", errorCodeFilter);
                        }
                    }
                    break;
                }
                case 'q': // depth 显示中心点距离
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        auto itShowConter = g_mapShowConterValue.find(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                        if (itShowConter != g_mapShowConterValue.end())
                        {
                            itShowConter->second = !itShowConter->second;
                        }
                    }
                    break;
                }
                case 'c': // 关流不退出循环
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        errorCode = Synexens::StopStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                    }
                    break;
                }
                case 'e': // 重新开流
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        // errorCode = Synexens::StartStreaming(pDeviceInfo[nDeviceIndex].m_nDeviceID);
                    }
                    break;
                }
                case 'b': // 设置幅值滤波
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        float fFilterParams[10];
                        int num = 1;
                        Synexens::SYErrorCode errorFilterParam;

                        fFilterParams[0] = 0;
                        errorFilterParam = Synexens::SetFilterParam(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFILTERTYPE_AMPLITUDE, num, fFilterParams);
                        if (errorFilterParam == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("SetFilterParam SUCCESS");
                        }
                        else
                        {
                            PrintErrorCode("SetFilterParam", errorFilterParam);
                        }
                    }
                    break;
                }
                case 'r': // 读取温度
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        Synexens::SYErrorCode errorReadTemperature = Synexens::SYERRORCODE_FAILED;

                        float fltTemperature;
                        errorReadTemperature = Synexens::GetTemperature(pDeviceInfo[nDeviceIndex].m_nDeviceID, fltTemperature);
                        if (errorReadTemperature == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("ReadTemperature SUCCESS,%f", fltTemperature);
                        }
                        else
                        {
                            PrintErrorCode("ReadTemperature", errorReadTemperature);
                        }
                    }
                    break;
                }
                case 'g': // 设置IR滤波，并读取IR滤波
                {
                    for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                    {
                        Synexens::SYErrorCode errorCodeAddFilter = Synexens::AddFilter(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFILTERTYPE_IR);
                        if (errorCodeAddFilter == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("AddFilter IR success\n");
                        }
                        else
                        {
                            PrintErrorCode("AddFilter IR", errorCodeAddFilter);
                        }

                        float fFilterParams[10];
                        int num = 1;
                        fFilterParams[0] = 120;
                        Synexens::SYErrorCode errorSetFilterParams = Synexens::SetFilterParam(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFILTERTYPE_IR, num, fFilterParams);
                        if (errorSetFilterParams == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("SetFilterParam IR success\n");
                        }
                        else
                        {
                            PrintErrorCode("SetFilterParam IR", errorSetFilterParams);
                        }

                        float fGetFilterParams[10];
                        Synexens::SYErrorCode errorGetFilterParams = Synexens::GetFilterParam(pDeviceInfo[nDeviceIndex].m_nDeviceID, Synexens::SYFILTERTYPE_IR, num, fGetFilterParams);
                        if (errorGetFilterParams == Synexens::SYERRORCODE_SUCCESS)
                        {
                            printf("GetFilterParam IR success :%f \n", fGetFilterParams[0]);
                        }
                        else
                        {
                            PrintErrorCode("GetFilterParam IR", errorGetFilterParams);
                        }
                    }
                    break;
                }
                case 'n': // 导入配置
                {
                    std::string strFile = "E:\\HV0130315L0358_sdkconfig.yaml";
                    Synexens::SYConfiguration configuration;
                    if (Synexens::ParseConfiguration(strFile.length(), strFile.c_str(), configuration) == Synexens::SYERRORCODE_SUCCESS)
                    {
                        for (int nDeviceIndex = 0; nDeviceIndex < nCount; nDeviceIndex++)
                        {
                            Synexens::SYErrorCode errorcodeImport = Synexens::StartStreamingWithConfiguration(pDeviceInfo[nDeviceIndex].m_nDeviceID, configuration);
                            if (errorcodeImport == Synexens::SYERRORCODE_SUCCESS)
                            {
                                printf("Import configuration success\n");
                            }
                            else
                            {
                                PrintErrorCode("Import configuration", errorcodeImport);
                            }
                        }
                    }
                    break;
                }
                }

                if (bBreak)
                    break;
            }
            for (int i = 0; i < nCount; i++)
            {
                if (pOpen[i])
                    errorCode = Synexens::StopStreaming(pDeviceInfo[i].m_nDeviceID);
                if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                {
                    printf("StopStreaming Success\n");
                }
                else
                {
                    PrintErrorCode("StopStreaming", errorCode);
                }
            }

            delete[] pOpen;
            delete[] pIntegralTimeMin;
            delete[] pIntegralTimeMax;
            delete[] pIntegralTime;
        }
        else
        {
            PrintErrorCode("FindDevice2", errorCode);
        }

        delete[] pDeviceInfo;
    }
    else
    {
        PrintErrorCode("FindDevice", errorCode);
    }
    errorCodeInitSDK = Synexens::UnInitSDK();
    if (errorCodeInitSDK != Synexens::SYERRORCODE_SUCCESS)
    {
        PrintErrorCode("InitSDK", errorCodeInitSDK);
    }

    g_is_start = false;
    if (fpsThread.joinable())
        fpsThread.join();

    while (cv::waitKey(30) != 27)
        continue;

    return 0;
}
