import os # 用于文件和目录操作，如路径拼接、创建目录、列出文件等。
import argparse # 用于解析命令行参数，使得用户可以自定义输入输出路径、设备等。

from PIL import Image # Pillow库，用于图像的打开、处理和保存。
import numpy as np # NumPy库，虽然在此脚本中未直接大量使用，但通常是图像处理和PyTorch的底层依赖。

import torch # PyTorch核心库。
from torchvision.transforms.functional import to_tensor, to_pil_image # torchvision提供的图像转换工具，
                                                                    # to_tensor: PIL Image/NumPy array 转 Tensor，并归一化到[0,1]。
                                                                    # to_pil_image: Tensor 转 PIL Image。

from model import Generator # 从同级目录的 model.py 文件中导入生成器网络结构。


# PyTorch CUDA 后端设置，用于保证实验结果的可复现性。
# 在某些情况下，禁用cudnn的自动调优和确定性算法可以使每次运行的结果完全一致，但可能会牺牲一些性能。
torch.backends.cudnn.enabled = False # 禁用cuDNN，如果为True，cuDNN会尝试使用非确定性算法进行优化。
torch.backends.cudnn.benchmark = False # 若为True，cuDNN会自动寻找最适合当前配置的高效算法，可能导致非确定性。
torch.backends.cudnn.deterministic = True # 若为True，强制cuDNN使用确定性算法，有助于复现。

# load_image: 加载图像，转换为RGB，并可选择将其尺寸调整为32的倍数。
def load_image(image_path, x32=False):
    # 参数:
    #   image_path (str): 图像文件的路径。
    #   x32 (bool): 是否将图像的宽和高调整为32的倍数。默认为False。
    #               某些模型（尤其是卷积神经网络）可能对输入的尺寸有要求，例如必须是某个值的倍数，以避免特征图对齐问题。
    img = Image.open(image_path).convert("RGB") # 打开图像文件，并确保其为RGB格式（处理灰度图、RGBA图等）。

    if x32: # 如果x32参数为True，则执行尺寸调整逻辑。
        def to_32s(x):
            # 内部辅助函数，计算使x成为32的倍数的值。
            # 如果x小于256，则直接返回256（设定一个最小尺寸）。
            # 否则，返回 x - (x % 32)，即 x 向下取整到最近的32的倍数。
            return 256 if x < 256 else x - x % 32
        w, h = img.size # 获取原始图像的宽度和高度。
        img = img.resize((to_32s(w), to_32s(h))) # 使用计算出的新宽高调整图像尺寸。

    return img # 返回处理后的PIL Image对象。

# test: 执行图像的动漫风格转换。
def test(args):
    # 参数:
    #   args (argparse.Namespace): 通过命令行参数解析器获得的参数对象。
    #                              包含了如模型路径、输入输出目录、设备等配置。
    device = args.device # 获取用户指定的推理设备（如 "cuda:0" 或 "cpu"）。
    
    # 1. 加载模型
    net = Generator() # 实例化在 model.py 中定义的生成器网络。
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu")) # 加载预训练的模型权重。
                                                                        # args.checkpoint 是权重文件的路径。
                                                                        # map_location="cpu" 确保权重首先加载到CPU内存中，
                                                                        # 这样即使当前没有指定的GPU设备或GPU显存不足，也能成功加载。
                                                                        # 后续 .to(device) 会将其移至正确设备。
    net.to(device).eval() # 将模型移动到指定的设备 (CPU 或 GPU)。
                          # .eval() 将模型设置为评估（推理）模式。这会影响某些层，如Dropout（会禁用）和BatchNorm（会使用全局统计数据而非批次统计数据）。
    print(f"model loaded: {args.checkpoint}") # 打印模型加载信息。
    
    # 2. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True) # 创建保存结果的目录。
                                                # exist_ok=True 表示如果目录已存在，则不会抛出错误。

    # 3. 遍历输入目录中的图像
    for image_name in sorted(os.listdir(args.input_dir)): # 遍历输入目录下的所有文件和文件夹名，并排序。
        # 文件类型过滤，只处理常见的图像格式。
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue # 如果文件扩展名不是指定的图像格式，则跳过。
            
        image = load_image(os.path.join(args.input_dir, image_name), args.x32) # 调用load_image函数加载并预处理（如果x32为True）图像。

        # 4. 模型推理
        with torch.no_grad(): # 上下文管理器，在其内部禁用梯度计算。
                              # 在推理阶段不需要计算梯度，这样可以减少内存消耗并加速计算。
            # 4.1 图像预处理
            image = to_tensor(image).unsqueeze(0) * 2 - 1 # PIL Image -> Tensor (C,H,W)，像素值[0,1] -> (1,C,H,W)
                                                        # 像素值范围从 [0,1] 转换为 [-1,1]，以匹配模型训练时的输入。
            # 4.2 执行模型前向传播
            out = net(image.to(device), args.upsample_align).cpu() # 将输入张量移到模型所在设备。
                                                               # net(...) 执行模型推理。args.upsample_align 传递给Generator的forward方法。
                                                               # .cpu() 将输出结果从GPU移回CPU，以便后续用PIL处理和保存。
            # 4.3 图像后处理
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5 # 移除批次维度 (1,C,H,W) -> (C,H,W)。
                                                        # .clip(-1, 1) 确保输出值在[-1,1]范围内，模型输出的Tanh激活理论上在此范围，但可能有浮点误差。
                                                        # 像素值范围从 [-1,1] 转换回 [0,1]。
            # 4.4 Tensor 转 PIL Image
            out = to_pil_image(out) # 将处理后的张量转换回PIL图像对象。

        # 5. 保存结果图像
        out.save(os.path.join(args.output_dir, image_name)) # 保存生成的图像到输出目录，文件名与原图一致。
        print(f"image saved: {image_name} -> {os.path.join(args.output_dir, image_name)}") # 打印保存信息。


if __name__ == '__main__':
    # 当脚本作为主程序执行时运行以下代码。

    parser = argparse.ArgumentParser() # 创建一个命令行参数解析器对象。

    # 定义脚本可接受的命令行参数：
    parser.add_argument(
        '--checkpoint', # 参数名
        type=str, # 参数类型为字符串
        default='./weights/paprika.pt', # 如果用户未提供此参数时的默认值（预设的一个权重文件）
        help="Path to the pretrained model checkpoint file." # 参数的帮助说明
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./samples/inputs', # 默认输入目录
        help="Directory containing input images."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples/results', # 默认输出目录
        help="Directory to save processed images."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0', # 推理设备，默认为'cuda:0'，可指定为 'cpu' 或其他cuda设备。
        help="Device to use for inference (e.g., 'cpu', 'cuda:0')."
    )
    parser.add_argument(
        '--upsample_align',
        type=bool, # 布尔类型的参数
        default=False, # 默认值为False
        help="Align corners in decoder upsampling layers (see model.py forward pass)." # 这个参数会传递给 Generator 模型的 forward 方法。
    )
    parser.add_argument(
        '--x32',
        action="store_true", # 当命令行中出现 --x32 时，此参数为True，否则为False。不需要额外的值。
        help="Resize input images to be multiples of 32." # 是否将输入图像尺寸调整为32的倍数。
    )
    args = parser.parse_args() # 解析命令行传入的参数，结果存储在args对象中。
    
    test(args) # 调用test函数，并将解析得到的参数传递给它。
