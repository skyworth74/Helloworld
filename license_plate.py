#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国大陆、香港、澳门车牌识别程序
支持：
1. Ubuntu 摄像头实时识别
2. JPG/PNG 图片文件识别
3. TCP/IP 网络接收图片识别
"""

import cv2
import numpy as np
import easyocr
import socket
import struct
import os
import sys
from datetime import datetime


class LicensePlateRecognizer:
    """车牌识别器类"""
    
    def __init__(self, langs=['ch_sim', 'en']):
        """
        初始化识别器
        :param langs: 识别语言列表，默认简体中文和英文
        """
        print("正在加载 OCR 模型，请稍候...")
        self.reader = easyocr.Reader(langs, gpu=False)
        print("OCR 模型加载完成")
        
        # 车牌正则模式（简化版）
        # 中国大陆：省份简称 + 字母 + 5 位字母数字
        # 香港：2 字母 + 2-4 数字 或 2 数字 + 2 字母
        # 澳门：M 字母 + 5 数字 或 M 字母 + 4 数字
        
    def preprocess_image(self, image):
        """
        图像预处理
        :param image: 输入图像
        :return: 预处理后的图像
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 调整图像大小
        height, width = image.shape[:2]
        if width < 800:
            scale = 800 / width
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # 增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab[:, :, 0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_and_recognize(self, image):
        """
        检测并识别车牌
        :param image: 输入图像
        :return: 识别结果列表 [(车牌文本，置信度，边界框)]
        """
        results = []
        
        # 预处理
        processed = self.preprocess_image(image)
        
        # 使用 EasyOCR 进行文字检测
        ocr_results = self.reader.readtext(processed, 
                                           detail=1,
                                           paragraph=False,
                                           min_size=20,
                                           contrast_ths=0.3,
                                           adjust_contrast=0.5,
                                           text_threshold=0.6,
                                           low_text=0.4)
        
        for bbox, text, confidence in ocr_results:
            # 清理文本
            text = text.strip().upper()
            
            # 过滤可能的车牌
            if self.is_possible_plate(text):
                results.append((text, confidence, bbox))
        
        # 按置信度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def is_possible_plate(self, text):
        """
        判断文本是否可能是车牌号
        :param text: 待检测文本
        :return: True/False
        """
        import re
        
        # 移除空格和特殊字符
        text = re.sub(r'[\s·•]', '', text)
        
        # 中国大陆车牌模式
        # 新能源：省份 + 字母 + D/F + 5 位
        cn_new_energy = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][DF][A-Z0-9]{5}$'
        # 普通：省份 + 字母 + 5 位
        cn_normal = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5}$'
        # 武警：WJ + 省份 + 5 位
        cn_police = r'^WJ[A-Z0-9]{5,6}$'
        # 军车：军 + 字母 + 5 位
        cn_military = r'^军[A-Z][A-Z0-9]{4,5}$'
        
        # 香港车牌模式
        # 普通：2 字母 + 1-4 数字
        hk_normal = r'^[A-Z]{1,2}[0-9]{1,4}$'
        # 定制：2 字母 + 空格 + 1-4 数字
        hk_custom = r'^[A-Z]{1,2}\\s[0-9]{1,4}$'
        
        # 澳门车牌模式
        # 普通：M + 1 字母 + 4-5 数字
        mo_normal = r'^M[A-Z][0-9]{4,5}$'
        # 临时：M + 1 字母 + 4-5 数字 + T
        mo_temp = r'^M[A-Z][0-9]{4,5}T$'
        # 政府：M + 1 字母 + 4-5 数字 + G
        mo_gov = r'^M[A-Z][0-9]{4,5}G$'
        
        patterns = [
            cn_new_energy, cn_normal, cn_police, cn_military,
            hk_normal, hk_custom,
            mo_normal, mo_temp, mo_gov
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        # 长度检查（4-8 个字符）
        if 4 <= len(text) <= 8 and any(c.isdigit() for c in text):
            return True
        
        return False
    
    def draw_results(self, image, results):
        """
        在图像上绘制识别结果
        :param image: 原始图像
        :param results: 识别结果
        :return: 绘制后的图像
        """
        output = image.copy()
        
        for i, (text, confidence, bbox) in enumerate(results):
            # 转换边界框为整数
            bbox = [(int(x), int(y)) for x, y in bbox]
            
            # 绘制边界框
            cv2.polylines(output, [np.array(bbox)], True, (0, 255, 0), 2)
            
            # 准备标签文本
            label = f"{text} ({confidence:.2f})"
            
            # 计算标签位置
            top_left = bbox[0]
            bottom_right = bbox[2]
            
            # 获取文本大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制背景矩形
            cv2.rectangle(output, 
                         (top_left[0], top_left[1] - text_height - 10),
                         (top_left[0] + text_width, top_left[1]),
                         (0, 255, 0), -1)
            
            # 绘制文本
            cv2.putText(output, label, 
                       (top_left[0], top_left[1] - 5),
                       font, font_scale, (0, 0, 0), thickness)
        
        return output


class CameraCapture:
    """摄像头捕获类"""
    
    def __init__(self, camera_id=0):
        """
        初始化摄像头
        :param camera_id: 摄像头设备 ID
        """
        self.camera_id = camera_id
        self.cap = None
    
    def open(self):
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise IOError(f"无法打开摄像头 {self.camera_id}")
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"摄像头 {self.camera_id} 已打开")
    
    def read_frame(self):
        """读取一帧"""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        """释放摄像头"""
        if self.cap is not None:
            self.cap.release()
            print("摄像头已关闭")


class ImageFileHandler:
    """图片文件处理类"""
    
    @staticmethod
    def load_image(file_path):
        """
        加载图片文件
        :param file_path: 文件路径
        :return: 图像数组
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"无法读取图像：{file_path}")
        
        return image
    
    @staticmethod
    def get_supported_formats():
        """获取支持的图片格式"""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']


class TCPServer:
    """TCP 服务器类，用于接收网络传输的图片"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        """
        初始化 TCP 服务器
        :param host: 监听地址
        :param port: 监听端口
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
    
    def start(self):
        """启动服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"TCP 服务器已启动，监听 {self.host}:{self.port}")
    
    def accept_connection(self, timeout=5):
        """
        接受客户端连接
        :param timeout: 超时时间（秒）
        :return: True/False
        """
        self.server_socket.settimeout(timeout)
        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"客户端已连接：{addr}")
            return True
        except socket.timeout:
            return False
    
    def receive_image(self):
        """
        接收图片数据
        :return: 图像数组或 None
        """
        if self.client_socket is None:
            return None
        
        # 接收图片长度（4 字节）
        length_data = self._recv_all(4)
        if not length_data:
            return None
        
        length = struct.unpack('>I', length_data)[0]
        
        # 接收图片数据
        image_data = self._recv_all(length)
        if not image_data:
            return None
        
        # 解码图像
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    
    def _recv_all(self, size):
        """接收指定长度的数据"""
        data = b''
        while len(data) < size:
            packet = self.client_socket.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def send_response(self, plate_info):
        """
        发送识别结果
        :param plate_info: 车牌信息
        """
        if self.client_socket is None:
            return
        
        response = str(plate_info).encode('utf-8')
        length = len(response)
        
        # 发送长度
        self.client_socket.sendall(struct.pack('>I', length))
        # 发送数据
        self.client_socket.sendall(response)
    
    def close(self):
        """关闭连接"""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        print("TCP 连接已关闭")


def recognize_from_camera(recognizer, camera_id=0):
    """
    从摄像头实时识别车牌
    :param recognizer: 识别器实例
    :param camera_id: 摄像头 ID
    """
    camera = CameraCapture(camera_id)
    
    try:
        camera.open()
        print("\n=== 车牌实时识别模式 ===")
        print("按 'q' 键退出，按 's' 键保存当前帧")
        print("=" * 30)
        
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("无法读取摄像头画面")
                break
            
            # 识别车牌
            results = recognizer.detect_and_recognize(frame)
            
            # 绘制结果
            output = recognizer.draw_results(frame, results)
            
            # 显示信息
            if results:
                info_text = f"检测到 {len(results)} 个车牌"
                cv2.putText(output, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 打印第一个结果
                plate, conf, _ = results[0]
                print(f"\r检测到车牌：{plate} (置信度：{conf:.2f})", end='', flush=True)
            
            # 显示窗口
            cv2.imshow('License Plate Recognition - Press Q to quit', output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, output)
                print(f"\n已保存：{filename}")
    
    except Exception as e:
        print(f"错误：{e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def recognize_from_file(recognizer, file_path):
    """
    从文件识别车牌
    :param recognizer: 识别器实例
    :param file_path: 图片文件路径
    """
    print(f"\n=== 从文件识别：{file_path} ===")
    
    try:
        image = ImageFileHandler.load_image(file_path)
        results = recognizer.detect_and_recognize(image)
        
        if results:
            print(f"\n检测到 {len(results)} 个可能的车牌:")
            for i, (plate, conf, bbox) in enumerate(results, 1):
                print(f"  {i}. {plate} (置信度：{conf:.2f})")
            
            # 绘制并保存结果
            output = recognizer.draw_results(image, results)
            output_path = f"result_{os.path.basename(file_path)}"
            cv2.imwrite(output_path, output)
            print(f"\n结果已保存到：{output_path}")
        else:
            print("未检测到车牌")
        
        return results
    
    except Exception as e:
        print(f"错误：{e}")
        return []


def recognize_from_tcp(recognizer, host='0.0.0.0', port=5000):
    """
    通过 TCP 接收图片并识别
    :param recognizer: 识别器实例
    :param host: 监听地址
    :param port: 监听端口
    """
    server = TCPServer(host, port)
    
    try:
        server.start()
        print("\n=== TCP 网络识别模式 ===")
        print("等待客户端连接...")
        print("按 Ctrl+C 退出")
        print("=" * 30)
        
        while True:
            if server.accept_connection(timeout=5):
                try:
                    image = server.receive_image()
                    
                    if image is not None:
                        print("\n收到图片，正在识别...")
                        results = recognizer.detect_and_recognize(image)
                        
                        if results:
                            print(f"检测到 {len(results)} 个车牌:")
                            for plate, conf, _ in results:
                                print(f"  - {plate} ({conf:.2f})")
                            
                            # 发送结果
                            server.send_response([(p, c) for p, c, _ in results])
                        else:
                            print("未检测到车牌")
                            server.send_response([])
                    else:
                        print("接收图片失败")
                
                except Exception as e:
                    print(f"处理错误：{e}")
                finally:
                    server.close()
    
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误：{e}")
    finally:
        server.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='中国大陆、香港、澳门车牌识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  1. 摄像头实时识别:
     python license_plate.py --camera
  
  2. 识别图片文件:
     python license_plate.py --image car.jpg
  
  3. 批量识别目录:
     python license_plate.py --dir ./images
  
  4. TCP 网络接收:
     python license_plate.py --tcp --port 5000
        """
    )
    
    parser.add_argument('--camera', '-c', action='store_true',
                       help='使用摄像头实时识别')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头设备 ID (默认：0)')
    parser.add_argument('--image', '-i', type=str,
                       help='识别单个图片文件')
    parser.add_argument('--dir', '-d', type=str,
                       help='批量识别目录中的所有图片')
    parser.add_argument('--tcp', '-t', action='store_true',
                       help='启动 TCP 服务器接收图片')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='TCP 服务器监听地址 (默认：0.0.0.0)')
    parser.add_argument('--port', '-p', type=int, default=5000,
                       help='TCP 服务器监听端口 (默认：5000)')
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助
    if not (args.camera or args.image or args.dir or args.tcp):
        parser.print_help()
        return
    
    # 初始化识别器
    recognizer = LicensePlateRecognizer()
    
    try:
        # 摄像头模式
        if args.camera:
            recognize_from_camera(recognizer, args.camera_id)
        
        # 单文件模式
        elif args.image:
            recognize_from_file(recognizer, args.image)
        
        # 批量模式
        elif args.dir:
            if not os.path.isdir(args.dir):
                print(f"错误：目录不存在 {args.dir}")
                return
            
            supported = ImageFileHandler.get_supported_formats()
            files = [f for f in os.listdir(args.dir) 
                    if any(f.lower().endswith(ext) for ext in supported)]
            
            print(f"\n在目录 {args.dir} 中找到 {len(files)} 个图片文件")
            
            for filename in files:
                filepath = os.path.join(args.dir, filename)
                print(f"\n{'='*50}")
                recognize_from_file(recognizer, filepath)
        
        # TCP 服务器模式
        elif args.tcp:
            recognize_from_tcp(recognizer, args.host, args.port)
    
    except KeyboardInterrupt:
        print("\n程序已终止")
    except Exception as e:
        print(f"程序错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
