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
#include "SYDataDefine.h"
#include <string>
#include <iostream>

#if WIN32
#define USE_FRAME_OBSERVER
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

std::chrono::milliseconds g_last_time;
std::thread fpsThread;

void calculate_framerate()
{
    while (g_is_start)
    {
        std::chrono::milliseconds cur_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());

        if (cur_time.count() - g_last_time.count() >= 1000)
        {
            // printf("===============> cur_time:%lf \n", cur_time);
            g_bRefreshFPS = true;
            g_last_time = cur_time;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

#ifndef WIN32
void sprintf_s(char *const str, size_t const leng, char const *const _Format, unsigned int id)
{
    sprintf(str, _Format, id);
}
#endif

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
        for (int nFrameIndex = 0; nFrameIndex < pFrameData->m_nFrameCount; nFrameIndex++)
        {
            mapIndex.insert(std::pair<Synexens::SYFrameType, int>(pFrameData->m_pFrameInfo[nFrameIndex].m_frameType, nFrameIndex));
            mapPos.insert(std::pair<Synexens::SYFrameType, int>(pFrameData->m_pFrameInfo[nFrameIndex].m_frameType, nPos));
            nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
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
            if (Synexens::GetRGBD(nDeviceID, pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameWidth, pFrameData->m_pFrameInfo[itDepthIndex->second].m_nFrameHeight, (unsigned short *)((unsigned char*)pFrameData->m_pData + mapPos[Synexens::SYFRAMETYPE_DEPTH]),
                                  pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameWidth, pFrameData->m_pFrameInfo[itRGBIndex->second].m_nFrameHeight, (unsigned char *)pFrameData->m_pData + mapPos[Synexens::SYFRAMETYPE_RGB],
                                  nRGBDWidth, nRGBDHeight, pRGBDDepth, pRGBDRGB) == Synexens::SYERRORCODE_SUCCESS)
            {
                g_mapFrameCount[nDeviceID]++;

                printf("Get RGBD Success FPS:%d \n", g_mapFPS[nDeviceID]);
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

                printf("Get RAW Success FPS:%d \n", g_mapFPS[nDeviceID]);
                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
                break;
            }
            case Synexens::SYFRAMETYPE_DEPTH:
            {
                g_mapFrameCount[nDeviceID]++;

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
                printf("Get Depth Success FPS:%d \n", g_mapFPS[nDeviceID]);

                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
                break;
            }
            case Synexens::SYFRAMETYPE_IR:
            {
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
                printf("Get IR Success \n");
                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * sizeof(short);
                break;
            }
            case Synexens::SYFRAMETYPE_RGB:
            {
                printf("Get RGB Success \n");
                nPos += pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameHeight * pFrameData->m_pFrameInfo[nFrameIndex].m_nFrameWidth * 3 / 2;
                break;
            }
            }
        }
    }
}

class FrameObserver : public Synexens::ISYFrameObserver
{
public:
    virtual void OnFrameNotify(unsigned int nDeviceID, Synexens::SYFrameData *pFrameData = nullptr)
    {
        printf("Observer Success \n");
        ProcessFrameData(nDeviceID, pFrameData);
    }
};

FrameObserver g_FrameObserver;

void PrintErrorCode(std::string strFunc, Synexens::SYErrorCode errorCode)
{
    printf("%s errorcode:%d\n", strFunc.c_str(), errorCode);
}

int main()
{
    printf("SynexensSDK Test Demo\n");
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
                // 测试打印端口信息
                if (pDeviceInfo[i].m_deviceType == Synexens::SYDEVICETYPE_CS30_SINGLE || pDeviceInfo[i].m_deviceType == Synexens::SYDEVICETYPE_CS30_DUAL)
                {
                    printf("print CS30 Bus:%d deviceAddress:%d \n", pDeviceInfo[i].m_nUsbBus, pDeviceInfo[i].m_nUsbDeviceAddress);
                    int num = pDeviceInfo[i].m_nUsbPortsNumber;
                    for (int j = 0; j < num; j++)
                    {
                        if (j < num)
                            printf("print CS30 port:%d \n", pDeviceInfo[i].m_nUsbPorts[j]);
                    }
                }
                    
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
                    switch (pDeviceInfo[i].m_deviceType)
                    {
                    case Synexens::SYDEVICETYPE_CS30_SINGLE:
                    case Synexens::SYDEVICETYPE_CS30_DUAL:
                    {
                        errorCode = Synexens::SetFrameResolution(pDeviceInfo[i].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                        {
                            errorCode = Synexens::SetFrameResolution(pDeviceInfo[i].m_nDeviceID, Synexens::SYFRAMETYPE_RGB, Synexens::SYRESOLUTION_1920_1080);
                            if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                            {
                                Synexens::SYStreamType streamType = Synexens::SYSTREAMTYPE_DEPTH;
                                errorCode = Synexens::StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType);
                                if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                                {
                                    auto itStreamFind = g_mapStreamType.find(pDeviceInfo[i].m_nDeviceID);
                                    if (itStreamFind != g_mapStreamType.end())
                                    {
                                        itStreamFind->second = streamType;
                                    }
                                    else
                                    {
                                        g_mapStreamType.insert(std::pair<unsigned int, Synexens::SYStreamType>(pDeviceInfo[i].m_nDeviceID, streamType));
                                    }

                                    pOpen[i] = true;
                                }
                                else
                                {
                                    PrintErrorCode("StartStreaming", errorCode);
                                }
                            }
                            else
                            {
                                PrintErrorCode("SetFrameResolution RGB", errorCode);
                            }
                        }
                        else
                        {
                            PrintErrorCode("SetFrameResolution Depth", errorCode);
                        }

                        break;
                    }
                    case Synexens::SYDEVICETYPE_CS20_SINGLE:
                    {
                        errorCode = Synexens::SetFrameResolution(pDeviceInfo[i].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);
                        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                        {
                            Synexens::SYStreamType streamType = Synexens::SYSTREAMTYPE_DEPTHIR;
                            errorCode = Synexens::StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType);
                            if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                            {
                                auto itStreamFind = g_mapStreamType.find(pDeviceInfo[i].m_nDeviceID);
                                if (itStreamFind != g_mapStreamType.end())
                                {
                                    itStreamFind->second = streamType;
                                }
                                else
                                {
                                    g_mapStreamType.insert(std::pair<unsigned int, Synexens::SYStreamType>(pDeviceInfo[i].m_nDeviceID, streamType));
                                }
                                pOpen[i] = true;
                            }
                            else
                            {
                                PrintErrorCode("StartStreaming", errorCode);
                            }
                        }
                        else
                        {
                            PrintErrorCode("SetFrameResolution Depth", errorCode);
                        }

                        break;
                    }
                    case Synexens::SYDEVICETYPE_CS20_DUAL:
                    {

                        errorCode = Synexens::SetFrameResolution(pDeviceInfo[i].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);

                        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                        {
                            Synexens::SYStreamType streamType = Synexens::SYSTREAMTYPE_DEPTHIR;
                            errorCode = Synexens::StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType);
                            if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                            {
                                auto itStreamFind = g_mapStreamType.find(pDeviceInfo[i].m_nDeviceID);
                                if (itStreamFind != g_mapStreamType.end())
                                {
                                    itStreamFind->second = streamType;
                                }
                                else
                                {
                                    g_mapStreamType.insert(std::pair<unsigned int, Synexens::SYStreamType>(pDeviceInfo[i].m_nDeviceID, streamType));
                                }

                                pOpen[i] = true;
                            }
                            else
                            {
                                PrintErrorCode("StartStreaming", errorCode);
                            }
                        }
                        else
                        {
                            PrintErrorCode("SetFrameResolution Depth", errorCode);
                        }

                        break;
                    }
                    case Synexens::SYDEVICETYPE_CS20_P:
                    {
                        errorCode = Synexens::SetFrameResolution(pDeviceInfo[i].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);
                        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                        {
                            Synexens::SYStreamType streamType = Synexens::SYSTREAMTYPE_DEPTH;
                            errorCode = Synexens::StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType);
                            if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                            {
                                auto itStreamFind = g_mapStreamType.find(pDeviceInfo[i].m_nDeviceID);
                                if (itStreamFind != g_mapStreamType.end())
                                {
                                    itStreamFind->second = streamType;
                                }
                                else
                                {
                                    g_mapStreamType.insert(std::pair<unsigned int, Synexens::SYStreamType>(pDeviceInfo[i].m_nDeviceID, streamType));
                                }

                                pOpen[i] = true;
                            }
                            else
                            {
                                PrintErrorCode("StartStreaming", errorCode);
                            }
                        }
                        else
                        {
                            PrintErrorCode("SetFrameResolution Depth", errorCode);
                        }

                        break;
                    }
                    case Synexens::SYDEVICETYPE_CS40:
                    {
                        errorCode = Synexens::SetFrameResolution(pDeviceInfo[i].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                        {
                            Synexens::SYStreamType streamType = Synexens::SYSTREAMTYPE_DEPTH;

                            errorCode = Synexens::StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType);

                            if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                            {
                                auto itStreamFind = g_mapStreamType.find(pDeviceInfo[i].m_nDeviceID);
                                if (itStreamFind != g_mapStreamType.end())
                                {
                                    itStreamFind->second = streamType;
                                }
                                else
                                {
                                    g_mapStreamType.insert(std::pair<unsigned int, Synexens::SYStreamType>(pDeviceInfo[i].m_nDeviceID, streamType));
                                }

                                pOpen[i] = true;
                            }
                            else
                            {
                                PrintErrorCode("StartStreaming", errorCode);
                            }
                        }
                        else
                        {
                            PrintErrorCode("SetFrameResolution Depth", errorCode);
                        }

                        break;
                    }
                    case Synexens::SYDEVICETYPE_CS40PRO:
                    {
                        errorCode = Synexens::SetFrameResolution(pDeviceInfo[i].m_nDeviceID, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_640_480);
                        if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                        {
                            Synexens::SYStreamType streamType = Synexens::SYSTREAMTYPE_DEPTH;

                            errorCode = Synexens::StartStreaming(pDeviceInfo[i].m_nDeviceID, streamType);

                            if (errorCode == Synexens::SYERRORCODE_SUCCESS)
                            {
                                auto itStreamFind = g_mapStreamType.find(pDeviceInfo[i].m_nDeviceID);
                                if (itStreamFind != g_mapStreamType.end())
                                {
                                    itStreamFind->second = streamType;
                                }
                                else
                                {
                                    g_mapStreamType.insert(std::pair<unsigned int, Synexens::SYStreamType>(pDeviceInfo[i].m_nDeviceID, streamType));
                                }

                                pOpen[i] = true;
                            }
                            else
                            {
                                PrintErrorCode("StartStreaming", errorCode);
                            }
                        }
                        else
                        {
                            PrintErrorCode("SetFrameResolution Depth", errorCode);
                        }

                        break;
                    }
                    }
                }
                else
                {
                    PrintErrorCode("OpenDevice", errorCodeOpenDevice);
                }
            }
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
                        Synexens::SYFrameData *pLastFrameData = nullptr;
                        Synexens::SYErrorCode errorCodeLastFrame = Synexens::GetLastFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, pLastFrameData);
                        if (errorCodeLastFrame == Synexens::SYERRORCODE_SUCCESS)
                        {
                            ProcessFrameData(pDeviceInfo[nDeviceIndex].m_nDeviceID, pLastFrameData);
                        }
                        else
                        {
                            // PrintErrorCode("GetLastFrameData", errorCode);
                        }
                    }
                }
#endif
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

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
    return 0;
}
