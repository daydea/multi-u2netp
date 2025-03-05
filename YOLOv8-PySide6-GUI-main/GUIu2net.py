import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import cv2
import torch
from torch.autograd import Variable
import numpy as np
from data_loader import RescaleT, ToTensorLab
from u2net_FPN import U2NETP_FPN
from skimage import io
import json
import tensorflow as tf
from model_efficientNet import efficient_net



class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("背带扣缺陷检测系统")
        self.root.geometry("800x600")

        # 设置窗口的背景颜色
        self.root.configure(bg="white")

        # 创建主框架
        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(expand=True, fill="both")

        # 创建Logo和标题
        self.create_logo_and_title()

        # 创建导航按钮
        self.create_navigation_buttons()

        # 初始化页面
        self.current_page = None
        self.switch_page("home")

    def create_logo_and_title(self):
        # Logo
        logo = Image.open("logo.png")  # 替换为你的logo路径
        logo.thumbnail((800, 350))  # 缩放logo图像大小
        logo_tk = ImageTk.PhotoImage(logo)
        label_logo = tk.Label(self.main_frame, image=logo_tk, bg="white")
        label_logo.image = logo_tk  # 保持引用
        label_logo.grid(row=0, column=0, columnspan=4, sticky="ew", pady=10)  # 自动拉伸宽度

    def create_navigation_buttons(self):
        # 设置按钮样式
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 18), foreground="white", background="#FF4500")  # 橙红色按钮

        # 导航按钮
        button_home = ttk.Button(self.main_frame, text="主页", command=lambda: self.switch_page("home"), style="TButton")
        button_home.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        button_defect_classification = ttk.Button(self.main_frame, text="图像分类", command=lambda: self.switch_page("defect_classification"), style="TButton")
        button_defect_classification.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        button_defect_detection = ttk.Button(self.main_frame, text="缺陷检测", command=lambda: self.switch_page("defect_detection"), style="TButton")
        button_defect_detection.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

        button_video_detection = ttk.Button(self.main_frame, text="视频检测", command=lambda: self.switch_page("video_detection"), style="TButton")
        button_video_detection.grid(row=1, column=3, padx=10, pady=10, sticky="ew")

    def switch_page(self, page_name):
        # 销毁当前页面
        if self.current_page:
            self.current_page.destroy()

        # 创建新页面
        if page_name == "home":
            self.current_page = self.create_home_page()
        elif page_name == "defect_classification":
            self.current_page = self.create_defect_classification_page()
        elif page_name == "defect_detection":
            self.current_page = self.create_defect_detection_page()
        elif page_name == "video_detection":
            self.current_page = self.create_video_detection_page()

        # 显示新页面
        self.current_page.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)

    def create_home_page(self):
        page = tk.Frame(self.main_frame, bg="white")

        # 欢迎信息
        label_home = tk.Label(page, text="欢迎来到主页！", font=("Helvetica", 18), bg="white", fg="#FF4500")
        label_home.pack(pady=20)

        # 创建一个水平的Frame容器，用于放置三张图片
        images_frame = tk.Frame(page, bg="white")
        images_frame.pack()

        # 图像分类图片
        image_classification = Image.open("classification.png")  # 替换为你的图片路径
        image_classification.thumbnail((200, 200))  # 调整图片大小
        image_classification_tk = ImageTk.PhotoImage(image_classification)
        label_classification_image = tk.Label(images_frame, image=image_classification_tk, bg="white")
        label_classification_image.image = image_classification_tk  # 保持引用
        label_classification_image.pack(side="left", padx=50)  # 并排放置，左侧

        # 缺陷检测图片
        image_detection = Image.open("detection.png")  # 替换为你的图片路径
        image_detection.thumbnail((500, 500))  # 调整图片大小
        image_detection_tk = ImageTk.PhotoImage(image_detection)
        label_detection_image = tk.Label(images_frame, image=image_detection_tk, bg="white")
        label_detection_image.image = image_detection_tk  # 保持引用
        label_detection_image.pack(side="left", padx=20)  # 并排放置，中间

        # 视频检测图片
        image_video = Image.open("video.png")  # 替换为你的图片路径
        image_video.thumbnail((200, 200))  # 调整图片大小
        image_video_tk = ImageTk.PhotoImage(image_video)
        label_video_image = tk.Label(images_frame, image=image_video_tk, bg="white")
        label_video_image.image = image_video_tk  # 保持引用
        label_video_image.pack(side="right", padx=0)  # 并排放置，右侧

        return page

    def create_defect_classification_page(self):
        page = tk.Frame(self.main_frame, bg="white")

        # 设置页面内按钮样式
        style = ttk.Style()
        style.configure("PageButton.TButton", font=("Helvetica", 14), foreground="white", background="#FF8000")  # 页面内按钮字体大小为14

        # 上传图像按钮
        button_upload = ttk.Button(page, text="上传工件图像", command=upload_image, style="TButton")
        button_upload.pack(pady=10, fill="x")

        # 识别缺陷按钮
        button_recognize = ttk.Button(page, text="识别工件种类", command=recognize_defects, style="TButton")
        button_recognize.pack(pady=10, fill="x")

        # 显示图像和结果
        global label_image, label_results
        label_image = tk.Label(page, bg="white")
        label_image.pack()
        label_results = tk.Label(page, text="分类结果：待识别", font=("Helvetica", 16), bg="white", fg="#FF4500")
        label_results.pack(pady=10)

        return page

    def create_defect_detection_page(self):
        page = tk.Frame(self.main_frame, bg="white")

        # 设置页面内按钮样式
        style = ttk.Style()
        style.configure("PageButton.TButton", font=("Helvetica", 14), foreground="white", background="#FF8000")  # 页面内按钮字体大小为14

        # 上传图像按钮
        button_upload = ttk.Button(page, text="上传工件图像", command=upload_image, style="TButton")
        button_upload.pack(pady=10, fill="x")

        # 检测缺陷按钮
        button_detect = ttk.Button(page, text="检测缺陷", command=detect_defects, style="TButton")
        button_detect.pack(pady=10, fill="x")

        # 创建一个水平的Frame容器，用于放置原图和检测结果图
        image_frame = tk.Frame(page, bg="white")
        image_frame.pack(pady=10)

        # 显示原图
        global label_image
        label_image = tk.Label(page, bg="white")
        label_image.pack(side="left", padx=10)  # 并排放置，左侧
        
        # 显示检测结果图
        global label_result_image
        label_result_image = tk.Label(page, bg="white")
        label_result_image.pack(side="left", padx=10)  # 并排放置，右侧
        
        # 显示检测结果文字
        global label_defect_results
        label_defect_results = tk.Label(page, text="缺陷检测结果：待检测", font=("Helvetica", 16), bg="white", fg="#FF4500")
        label_defect_results.pack(pady=10)
        
        return page

    def create_video_detection_page(self):
        page = tk.Frame(self.main_frame, bg="white")
        label_video_detection = tk.Label(page, text="视频检测功能尚未实现！", font=("Helvetica", 18), bg="white", fg="#FF4500")
        label_video_detection.pack(expand=True)
        return page


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = Image.open(file_path)
            image.thumbnail((200, 200))  # 缩放图像以适应界面
            image_tk = ImageTk.PhotoImage(image)
            label_image.config(image=image_tk)
            label_image.image = image_tk  # 保持引用
            label_image.file_path = file_path  # 保存文件路径以供后续使用
        except Exception as e:
            print(f"Error loading image: {e}")


def recognize_defects():
    file_path = label_image.file_path
    if file_path:
        result = classify_image(file_path)
        label_results.config(text=f"分类结果：{result}")


def classify_image(file_path):
    # 图像分类逻辑
    im_height = 224
    im_width = 224
    grayscale_mean = 128.0

    # 加载图像
    img = Image.open(file_path).convert("RGB")
    img = img.resize((im_width, im_height))
    img = np.array(img).astype(np.float32)
    img = img - grayscale_mean
    img = np.expand_dims(img, 0)

    # 加载类别
    json_path = './class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 加载模型
    model = efficient_net(width_coefficient=1.0,
                          depth_coefficient=1.1,
                          input_shape=(224, 224, 3),
                          dropout_rate=0.2,
                          drop_connect_rate=0.2,
                          activation="swish",
                          model_name="efficientnet",
                          include_top=True,
                          num_classes=10)
    weights_path = '/home/sherlock/下载/图像分类网络/deep-learning-for-image-processing-master/tensorflow_classification/Test5_resnet/新数据集下模型对比/NEW/背带扣数据集下改进模型对比/其它模型/efficientnet/mydata_effi.ckpt'
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(weights_path).expect_partial()

    # 预测
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)
    return class_indict[str(predict_class)]


def detect_defects():
    file_path = label_image.file_path
    if file_path:
        result = detect_defects_u2net(file_path)
        label_defect_results.config(text=f"缺陷检测结果：{result}")
        # 显示检测结果图片
        show_detection_image(file_path)
        
def show_detection_image(file_path):
    file_name = os.path.basename(file_path)
    result_image_path = os.path.join("test_out", file_name)
    if os.path.exists(result_image_path):
        try:
            result_image = Image.open(result_image_path)
            result_image.thumbnail((200, 200))
            result_image_tk = ImageTk.PhotoImage(result_image)
            label_result_image.config(image=result_image_tk)
            label_result_image.image = result_image_tk
        except Exception as e:
            print(f"Error loading detection image: {e}")
    else:
        print(f"Detection result image not found: {result_image_path}")

def detect_defects_u2net(file_path):
    # 缺陷检测逻辑
    model_dir = '/home/sherlock/下载/图像分类网络/deep-learning-for-image-processing-master/pytorch_segmentation/multi-u2net/multi_U2NET/saved_models/u2netp_fpn/aug_ngNok/1-50(whole)/u2netp_fpn_epoch_49.pth'
    image = io.imread(file_path)

    if not os.path.exists('test_out/'):
        os.makedirs('test_out/')

    label = np.zeros((image.shape[0], image.shape[1], 1))  # 添加一个通道维度
    imidx = np.array([0])

    sample = {'imidx': imidx, 'image': image, 'label': label}
    deal2 = ToTensorLab(flag=0)
    sample = deal2(sample)

    net = U2NETP_FPN(3, 5)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    model = net.eval()
    predict_np = eval_main(model, sample, (image.shape[1], image.shape[0]))

    # 解析预测结果
    defects = np.unique(predict_np)
    defect_classes = {1: "缺胶", 2: "飞边", 3: "流痕", 4: "拉丝", 5: "背景"}
    result = ", ".join([defect_classes[d] for d in defects if d in defect_classes])
    
    # 保存检测结果图片
    save_detection_image(predict_np, file_path)    
    
    return result

def save_detection_image(predict_np, file_path):
    # 获取文件名
    file_name = os.path.basename(file_path)
    result_image_path = os.path.join("test_out", file_name)

    # 将预测结果转换为彩色图片
    cls = dict([(1, (0, 0, 255)),  # 蓝色
                (2, (255, 0, 255)),  # 粉红色
                (3, (0, 255, 0)),  # 绿色
                (4, (255, 0, 0)),  # 红色
                (5, (255, 255, 0))])  # 黄色

    r = predict_np.copy()
    g = predict_np.copy()
    b = predict_np.copy()
    for c in cls:
        r[r == c] = cls[c][0]
        g[g == c] = cls[c][1]
        b[b == c] = cls[c][2]

    rgb = np.zeros((predict_np.shape[0], predict_np.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    # 保存图片
    Image.fromarray(rgb).save(result_image_path)


def eval_main(model, sample, img_size):
    inputs_test = sample['image'].unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)
    d1 = torch.softmax(d1, dim=1)
    predict_np = torch.argmax(d1, dim=1, keepdim=True)
    predict_np = predict_np.cpu().detach().numpy().squeeze()

    predict_np = cv2.resize(predict_np, img_size, interpolation=cv2.INTER_NEAREST)
    return predict_np


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
