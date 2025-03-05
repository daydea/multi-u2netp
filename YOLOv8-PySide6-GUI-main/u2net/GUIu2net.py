import tkinter as tk
from tkinter import filedialog, messagebox, font as tkfont
from PIL import Image, ImageTk

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)
        show_image(file_path)

def show_image(file_path):
    try:
        img = Image.open(file_path)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)  # 使用新的属性
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img  # 避免垃圾回收
    except Exception as e:
        messagebox.showerror("错误", f"无法加载图像: {e}")

def start_process():
    messagebox.showinfo("信息", "开始处理...")
    # 在这里添加处理逻辑

def export_data():
    messagebox.showinfo("信息", "导出数据...")
    # 在这里添加导出逻辑

def select_model():
    model = model_var.get()
    if model == "图像分类":
        messagebox.showinfo("信息", "选择的是图像分类模型")
    elif model == "缺陷分割":
        messagebox.showinfo("信息", "选择的是缺陷分割模型")

# 创建主窗口
root = tk.Tk()
root.title("背带扣缺陷检测系统")
root.geometry("800x600")

# 设置支持中文的字体
custom_font = tkfont.Font(family="Arial Unicode MS", size=16)

# 创建顶部标题栏
top_frame = tk.Frame(root, bg="#4CAF50", height=50)
top_frame.pack(side=tk.TOP, fill=tk.X)

title_label = tk.Label(top_frame, text="背带扣缺陷检测系统", bg="#4CAF50", fg="white", font=custom_font)
title_label.pack(side=tk.LEFT, padx=10)

# 创建左侧功能区
left_frame = tk.Frame(root, bg="#F2F2F2", width=200, height=500, relief=tk.RAISED, borderwidth=2)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

# 文件操作按钮
file_button = tk.Button(left_frame, text="选择图片文件", command=select_image)
file_button.pack(fill=tk.X, pady=5)

file_button = tk.Button(left_frame, text="选择图片文件夹", command=select_image)
file_button.pack(fill=tk.X, pady=5)

file_button = tk.Button(left_frame, text="选择视频文件", command=select_image)
file_button.pack(fill=tk.X, pady=5)

file_button = tk.Button(left_frame, text="打开摄像头", command=select_image)
file_button.pack(fill=tk.X, pady=5)

all_target_button = tk.Button(left_frame, text="所有目标", command=select_image)
all_target_button.pack(fill=tk.X, pady=5)

start_button = tk.Button(left_frame, text="开始运行", command=start_process)
start_button.pack(fill=tk.X, pady=5)

export_button = tk.Button(left_frame, text="导出数据", command=export_data)
export_button.pack(fill=tk.X, pady=5)

# 创建右侧显示区
right_frame = tk.Frame(root, bg="#F2F2F2", width=550, height=500)
right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

# 图片显示区域
image_label = tk.Label(right_frame)
image_label.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

# 模型选择
model_var = tk.StringVar(value="图像分类")
model_label = tk.Label(right_frame, text="选择模型:", bg="#F2F2F2", font=custom_font)
model_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")

model_option1 = tk.Radiobutton(right_frame, text="图像分类", variable=model_var, value="图像分类", bg="#F2F2F2", font=custom_font)
model_option1.grid(row=1, column=0, padx=5, pady=5, sticky="w")
model_option2 = tk.Radiobutton(right_frame, text="缺陷分割", variable=model_var, value="缺陷分割", bg="#F2F2F2", font=custom_font)
model_option2.grid(row=1, column=1, padx=5, pady=5, sticky="w")

file_label = tk.Label(right_frame, text="选择图片文件:", bg="#F2F2F2", font=custom_font)
file_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

file_entry = tk.Entry(right_frame, width=40, bg="#F2F2F2", font=custom_font)
file_entry.grid(row=2, column=1, padx=5, pady=5, sticky="we")

select_model_button = tk.Button(right_frame, text="选择模型", command=select_model, bg="#4CAF50", fg="white", font=custom_font)
select_model_button.grid(row=3, column=0, columnspan=2, pady=10, sticky="we")

# 运行主循环
root.mainloop()
