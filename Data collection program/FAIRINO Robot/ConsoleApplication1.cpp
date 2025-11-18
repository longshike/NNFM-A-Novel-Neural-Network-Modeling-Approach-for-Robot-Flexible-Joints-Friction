#include <robot.h>
#include <robot_types.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cstring>
#include <unistd.h>
#include <Windows.h>
#include <math.h>
#include <vector>
#include <chrono>

#define PI 3.1415926
using namespace std;

// 数据结构存储采集的数据
struct RobotData {
    double timestamp;
    double joint_pos[6];      // 关节角度
    double joint_vel[6];      // 关节角速度
    double joint_acc[6];      // 关节角加速度
    double joint_torque[6];   // 关节扭矩
};

class DataCollector {
private:
    ofstream data_file;
    chrono::steady_clock::time_point start_time;

public:
    bool init(const string& filename) {
        data_file.open(filename, ios::out | ios::trunc);
        if (!data_file.is_open()) {
            cout << "无法打开数据文件!" << endl;
            return false;
        }

        // 写入CSV表头
        data_file << "Timestamp";
        for (int i = 0; i < 6; i++) {
            data_file << ",J" << i + 1 << "_Pos,J" << i + 1 << "_Vel,J" << i + 1 << "_Acc,J" << i + 1 << "_Torque";
        }
        data_file << endl;

        start_time = chrono::steady_clock::now();
        return true;
    }

    bool collectData(FRRobot& robot) {
        RobotData data;

        // 获取时间戳
        auto current_time = chrono::steady_clock::now();
        data.timestamp = chrono::duration<double>(current_time - start_time).count();

        // 获取关节位置（度）
        JointPos j_pos;
        errno_t ret = robot.GetActualJointPosDegree(0, &j_pos);
        if (ret != 0) {
            cout << "获取关节位置失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_pos[i] = j_pos.jPos[i];
        }

        // 获取关节速度（度/秒）
        float j_vel[6];
        ret = robot.GetActualJointSpeedsDegree(0, j_vel);
        if (ret != 0) {
            cout << "获取关节速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_vel[i] = j_vel[i];
        }

        // 获取关节加速度（度/秒²）
        float j_acc[6];
        ret = robot.GetActualJointAccDegree(0, j_acc);
        if (ret != 0) {
            cout << "获取关节加速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_acc[i] = j_acc[i];
        }

        // 获取关节扭矩（Nm）
        float torques[6];
        ret = robot.GetJointTorques(0, torques);

        if (ret != 0) {
            cout << "获取关节扭矩失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_torque[i] = torques[i];
        }

        // 写入文件
        writeToFile(data);
        return true;
    }

private:
    void writeToFile(const RobotData& data) {
        data_file << data.timestamp;
        for (int i = 0; i < 6; i++) {
            data_file << "," << data.joint_pos[i]
                << "," << data.joint_vel[i]
                << "," << data.joint_acc[i]
                << "," << data.joint_torque[i];
        }
        data_file << endl;

        // 确保数据立即写入磁盘
        data_file.flush();
    }

public:
    ~DataCollector() {
        if (data_file.is_open()) {
            data_file.close();
            cout << "数据文件已关闭" << endl;
        }
    }
};

int main(void)
{
    FRRobot robot; // 实例化机器人
    DataCollector collector;

    // 连接机器人
    errno_t ret = robot.RPC("192.168.57.2");
    if (ret != 0) {
        printf("机器人连接失败，错误代码: %d\n", ret);
        return -1;
    }
    printf("机器人连接成功!\n");

    // 初始化数据采集器
    if (!collector.init("robot_trajectory_data.csv")) {
        return -1;
    }
    printf("数据文件初始化成功!\n");

    // 设置机器人参数
    robot.Mode(0);
    //robot.SetSpeed(5);//低速
    //robot.SetSpeed(20);//中速
    robot.SetSpeed(60);//高速

    // 加载并运行示教器程序
    char robot_programname[64] = "/fruser/longshikeluoxuan.lua";
    ret = robot.ProgramLoad(robot_programname);
    if (ret != 0) {
        printf("程序加载失败，错误代码: %d\n", ret);
        return -1;
    }

    ret = robot.ProgramRun();
    if (ret != 0) {
        printf("程序运行失败，错误代码: %d\n", ret);
        return -1;
    }
    printf("开始运行示教器程序...\n");

    // 等待程序启动
    Sleep(1000);

    // 开始数据采集
    int sample_count = 0;
    int failed_samples = 0;
    const int SAMPLE_RATE_MS = 10; // 采样率10ms (100Hz)
    const int MAX_FAILED_SAMPLES = 50; // 最大失败采样次数
    const int COLLECTION_DURATION_MS = 180000; // 采集持续时间：1分钟（60000毫秒）

    printf("开始数据采集...\n");
    printf("采样频率: %dHz\n", 1000 / SAMPLE_RATE_MS);
    printf("采集时长: %d秒\n", COLLECTION_DURATION_MS / 1000);

    // 记录采集开始时间
    auto collection_start_time = chrono::steady_clock::now();

    // 基于时间的数据采集循环
    while (true) {
        // 检查是否达到采集时长
        auto current_time = chrono::steady_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(current_time - collection_start_time).count();

        if (elapsed_time >= COLLECTION_DURATION_MS) {
            printf("达到预设采集时长，停止数据采集\n");
            break;
        }

        // 采集数据
        if (collector.collectData(robot)) {
            sample_count++;
        }
        else {
            failed_samples++;
            if (failed_samples >= MAX_FAILED_SAMPLES) {
                printf("数据采集失败次数过多，停止采集\n");
                break;
            }
        }

        // 每采集100次数据打印一次状态
        if (sample_count % 100 == 0) {
            printf("已采集 %d 组数据，已运行 %.1f 秒...\n",
                sample_count, elapsed_time / 1000.0);

            // 获取当前关节位置显示
            JointPos current_pos;
            if (robot.GetActualJointPosDegree(0, &current_pos) == 0) {
                printf("当前关节位置: J1=%.2f, J2=%.2f, J3=%.2f, J4=%.2f, J5=%.2f, J6=%.2f\n",
                    current_pos.jPos[0], current_pos.jPos[1], current_pos.jPos[2],
                    current_pos.jPos[3], current_pos.jPos[4], current_pos.jPos[5]);
            }
        }

        // 等待下一个采样周期
        Sleep(SAMPLE_RATE_MS);
    }

    printf("数据采集完成: 成功采集 %d 组数据, 失败 %d 次\n", sample_count, failed_samples);

    // 停止机器人
    robot.Mode(1);
    robot.ProgramStop();

    // 断开连接
    robot.CloseRPC();

    printf("数据已保存到 robot_trajectory_data.csv\n");
    printf("程序结束\n");

    return 0;
}