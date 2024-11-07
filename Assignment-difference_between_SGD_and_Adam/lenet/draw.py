import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 设置字体为 Times New Roman
# 启用 LaTeX 渲染
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为 Times New Roman
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # 加载 amsmath 包

# 设置字体大小
plt.rcParams['axes.labelsize'] = 26  # 设置轴标签字体大小
plt.rcParams['legend.fontsize'] = 20  # 设置图例字体大小
plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 20  # 设置y轴刻度字体大小

# 生成不重复的文件名
def generate_unique_filename(base_dir, base_name, extension):
    i = 1
    filename = f"{base_name}.{extension}"
    while os.path.exists(os.path.join(base_dir, filename)):
        filename = f"{base_name}_{i}.{extension}"
        i += 1
    return os.path.join(base_dir, filename)

def load_tensorboard_data(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    print("Available tags:", event_acc.Tags()['scalars'])
    train_acc = event_acc.Scalars('train_accu')
    train_loss = event_acc.Scalars('train_loss')
    val_acc = event_acc.Scalars('val  _accu')
    val_loss = event_acc.Scalars('val  _loss')  

    steps_acc = [x.step for x in train_acc]
    values_acc = [x.value for x in train_acc]
    steps_loss = [x.step for x in train_loss]
    values_loss = [x.value for x in train_loss]
    steps_val_acc = [x.step for x in val_acc]
    values_val_acc = [x.value for x in val_acc]
    steps_val_loss = [x.step for x in val_loss]
    values_val_loss = [x.value for x in val_loss]

    return (steps_acc, values_acc), (steps_loss, values_loss), (steps_val_acc, values_val_acc), (steps_val_loss, values_val_loss)



# 实验路径
exp_0_dir = "output/exp_15/tensorboard"
exp_1_dir = "output/exp_16/tensorboard"
exp_2_dir = "output/exp_17/tensorboard"

# 加载数据
(exp_0_acc_steps, exp_0_acc_values), (exp_0_loss_steps, exp_0_loss_values), (exp_0_val_acc_steps, exp_0_val_acc_values), (exp_0_val_loss_steps, exp_0_val_loss_values) = load_tensorboard_data(exp_0_dir)
(exp_1_acc_steps, exp_1_acc_values), (exp_1_loss_steps, exp_1_loss_values), (exp_1_val_acc_steps, exp_1_val_acc_values), (exp_1_val_loss_steps, exp_1_val_loss_values) = load_tensorboard_data(exp_1_dir)
(exp_2_acc_steps, exp_2_acc_values), (exp_2_loss_steps, exp_2_loss_values), (exp_2_val_acc_steps, exp_2_val_acc_values), (exp_2_val_loss_steps, exp_2_val_loss_values) = load_tensorboard_data(exp_2_dir)


# 绘制 Train and Validation Accuracy
plt.figure(figsize=(7, 5))

# 绘制训练集的 Accuracy
plt.plot(exp_0_acc_steps, exp_0_acc_values, label=r'$\eta$ = 0.001', color='blue')
plt.plot(exp_1_acc_steps, exp_1_acc_values, label=r'$\eta$ = 0.01', color='green')
plt.plot(exp_2_acc_steps, exp_2_acc_values, label=r'$\eta$ = 0.1', color='red')

# 绘制验证集的 Accuracy
plt.plot(exp_0_val_acc_steps, exp_0_val_acc_values, color='blue', linestyle='--', dashes=(6, 6))
plt.plot(exp_1_val_acc_steps, exp_1_val_acc_values, color='green', linestyle='--', dashes=(6, 6))
plt.plot(exp_2_val_acc_steps, exp_2_val_acc_values, color='red', linestyle='--', dashes=(6, 6))

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.xticks(np.arange(1, 6, step=1))

# 保存图像
output_dir = 'figure'
os.makedirs(output_dir, exist_ok=True)

plt.legend()
plt.grid(True)
filename_png = generate_unique_filename(output_dir, 'accuracy_comparison', 'png')
filename_pdf = generate_unique_filename(output_dir, 'accuracy_comparison', 'pdf')
plt.savefig(filename_png, bbox_inches='tight')
plt.savefig(filename_pdf, bbox_inches='tight')
plt.show()

# 绘制 Train and Validation Loss
plt.figure(figsize=(7, 5))

# 绘制训练集的 Loss
plt.plot(exp_0_loss_steps, exp_0_loss_values, label=r'$\eta$ = 0.001', color='blue')
plt.plot(exp_1_loss_steps, exp_1_loss_values, label=r'$\eta$ = 0.01', color='green')
plt.plot(exp_2_loss_steps, exp_2_loss_values, label=r'$\eta$ = 0.1', color='red')

# 绘制验证集的 Loss
plt.plot(exp_0_val_loss_steps, exp_0_val_loss_values, color='blue', linestyle='--', dashes=(6, 6))
plt.plot(exp_1_val_loss_steps, exp_1_val_loss_values, color='green', linestyle='--', dashes=(6, 6))
plt.plot(exp_2_val_loss_steps, exp_2_val_loss_values, color='red', linestyle='--', dashes=(6, 6))

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.xticks(np.arange(1, 6, step=1))

plt.legend()
plt.grid(True)
filename_png = generate_unique_filename(output_dir, 'loss_comparison', 'png')
filename_pdf = generate_unique_filename(output_dir, 'loss_comparison', 'pdf')
plt.savefig(filename_png, bbox_inches='tight')
plt.savefig(filename_pdf, bbox_inches='tight')
plt.show()
