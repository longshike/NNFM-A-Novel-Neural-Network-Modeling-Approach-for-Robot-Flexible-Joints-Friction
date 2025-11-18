#include <FARO_SDK.h>
#include <faro_robot_types.h>
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

    bool collectData(FARO_Robot& robot) {
        RobotData data;

        // 获取时间戳
        auto current_time = chrono::steady_clock::now();
        data.timestamp = chrono::duration<double>(current_time - start_time).count();

        // 获取关节位置（度）- 法奥机器人API
        FARO_JointPosition j_pos;
        int ret = robot.getJointPosition(&j_pos);
        if (ret != FARO_SUCCESS) {
            cout << "获取关节位置失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_pos[i] = j_pos.position[i];
        }

        // 获取关节速度（度/秒）- 法奥机器人API
        FARO_JointVelocity j_vel;
        ret = robot.getJointVelocity(&j_vel);
        if (ret != FARO_SUCCESS) {
            cout << "获取关节速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_vel[i] = j_vel.velocity[i];
        }

        // 获取关节加速度（度/秒²）- 法奥机器人API
        FARO_JointAcceleration j_acc;
        ret = robot.getJointAcceleration(&j_acc);
        if (ret != FARO_SUCCESS) {
            cout << "获取关节加速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_acc[i] = j_acc.acceleration[i];
        }

        // 获取关节扭矩（Nm）- 法奥机器人API
        FARO_JointTorque j_torque;
        ret = robot.getJointTorque(&j_torque);
        if (ret != FARO_SUCCESS) {
            cout << "获取关节扭矩失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_torque[i] = j_torque.torque[i];
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
    FARO_Robot robot; // 实例化法奥机器人
    DataCollector collector;

    // 连接法奥机器人
    int ret = robot.connect("192.168.57.2", 8080); // 假设法奥机器人使用IP和端口连接
    if (ret != FARO_SUCCESS) {
        printf("法奥机器人连接失败，错误代码: %d\n", ret);
        return -1;
    }
    printf("法奥机器人连接成功!\n");

    // 初始化数据采集器
    if (!collector.init("faro_robot_trajectory_data.csv")) {
        return -1;
    }
    printf("数据文件初始化成功!\n");

    // 设置机器人参数
    robot.setOperationMode(FARO_AUTO_MODE); // 自动模式
    //robot.setSpeed(5);  // 低速
    //robot.setSpeed(20); // 中速
    robot.setSpeed(60); // 高速

    // 加载并运行动作程序 - 法奥机器人API
    const char* program_path = "/programs/longshikeluoxuan.fp"; // 法奥机器人程序文件扩展名
    ret = robot.loadProgram(program_path);
    if (ret != FARO_SUCCESS) {
        printf("程序加载失败，错误代码: %d\n", ret);
        return -1;
    }

    ret = robot.startProgram();
    if (ret != FARO_SUCCESS) {
        printf("程序运行失败，错误代码: %d\n", ret);
        return -1;
    }
    printf("开始运行动作程序...\n");

    // 等待程序启动
    Sleep(1000);

    // 开始数据采集
    int sample_count = 0;
    int failed_samples = 0;
    const int SAMPLE_RATE_MS = 10; // 采样率10ms (100Hz)
    const int MAX_FAILED_SAMPLES = 50; // 最大失败采样次数
    const int COLLECTION_DURATION_MS = 180000; // 采集持续时间：3分钟（180000毫秒）

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
            FARO_JointPosition current_pos;
            if (robot.getJointPosition(&current_pos) == FARO_SUCCESS) {
                printf("当前关节位置: J1=%.2f, J2=%.2f, J3=%.2f, J4=%.2f, J5=%.2f, J6=%.2f\n",
                    current_pos.position[0], current_pos.position[1], current_pos.position[2],
                    current_pos.position[3], current_pos.position[4], current_pos.position[5]);
            }
        }

        // 等待下一个采样周期
        Sleep(SAMPLE_RATE_MS);
    }

    printf("数据采集完成: 成功采集 %d 组数据, 失败 %d 次\n", sample_count, failed_samples);

    // 停止机器人
    robot.stopProgram();
    robot.setOperationMode(FARO_MANUAL_MODE); // 切换到手动模式

    // 断开连接
    robot.disconnect();

    printf("数据已保存到 faro_robot_trajectory_data.csv\n");
    printf("程序结束\n");

    return 0;
}