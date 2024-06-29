import os
linear_dir = os.path.join(os.getcwd(), 'custom/linear_detection/linear/') #不用更改
detection_dir = os.path.join(os.getcwd(), 'custom/linear_detection/detection/') #不用更改
results_dir = os.path.join(os.getcwd(), 'runs/detect/results/') #不用更改

#选择框的左上角点位置
x0_ratio = 1/8 #可以更改
y0_ratio = 3/8 #可以更改
#选择框的右下角点位置
x1_ratio = 7/8 #可以更改
y1_ratio = 8/9 #可以更改
#获取线性公式设置为 linear，获取识别结果设置为 detection
mode = 'detection' # 'linear' or 'detection' 可以更改只能选择这两个之一
#获取线性公式时，需要手动输入浓度值，个数一定严格和比色皿数量比配
con_list = [1, 10, 30, 50, 80, 100, 120] #可以更改
#计算RGB平均值用到的小数点位数，如不提供程序默认16位
rgb_calculate_accuracy = 16 #可以更改
#图片显示RGB和Con.的小数点位数，可以单独控制
rgb_display_accuracy = 2 #可以更改
con_display_accuracy = 2 #可以更改
#选择相应的颜色通道
color_channel = 'B' # 'R' 'G' 'B' 三选一












