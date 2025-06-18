import argparse # 用于解析命令行参数。
import numpy as np # NumPy库，用于处理从TensorFlow获取的权重数组。
import os # 用于路径操作等。

import tensorflow as tf # TensorFlow库 (此处特指版本1.x)。
from AnimeGANv2.net import generator as tf_generator # 从原始AnimeGANv2项目导入其TensorFlow生成器网络定义。
                                                  # 这要求 AnimeGANv2 仓库在PYTHONPATH中，或者在同级目录。

import torch # PyTorch核心库。
from model import Generator # 从本项目的 model.py 文件中导入PyTorch版本的Generator定义。

# load_tf_weights: 加载 TensorFlow (v1) AnimeGANv2 生成器模型的 checkpoint 权重。
def load_tf_weights(tf_path):
    # 参数:
    #   tf_path (str): TensorFlow checkpoint文件所在的目录路径。
    # 返回:
    #   dict: 一个字典，键是TensorFlow模型中的变量名 (str)，值是对应的权重 (NumPy ndarray)。

    # 定义TensorFlow图的输入占位符 (TensorFlow v1风格)
    # [1, None, None, 3] 表示批次大小为1，高度和宽度可变，通道数为3。
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')

    # 在 "generator" 变量作用域内构建TensorFlow版本的生成器网络
    # reuse=False 表示创建新的变量 (如果是第一次构建该作用域)。
    # tf_generator.G_net 是从原始AnimeGANv2代码导入的TensorFlow模型构造函数。
    with tf.variable_scope("generator", reuse=False):
        test_generated = tf_generator.G_net(test_real).fake # 构建图，获取生成器的输出节点。

    # 创建一个Saver对象，用于从checkpoint恢复变量。
    saver = tf.train.Saver()

    # 创建一个TensorFlow Session来执行图操作。
    # allow_soft_placement=True: 如果指定设备不可用，允许TF自动选择一个可用设备。
    # device_count = {'GPU': 0}: 强制在CPU上执行，避免GPU相关的麻烦，因为这里只是加载权重，不需要GPU计算。
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})) as sess:
        # 获取checkpoint状态
        ckpt = tf.train.get_checkpoint_state(tf_path)

        # 断言确保checkpoint有效
        assert ckpt is not None and ckpt.model_checkpoint_path is not None, f"Failed to load checkpoint {tf_path}"

        # 从checkpoint恢复模型变量到当前session
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(f"Tensorflow model checkpoint {ckpt.model_checkpoint_path} loaded")

        # 提取所有可训练变量的权重
        tf_weights = {}
        for v in tf.trainable_variables(): # 遍历图中所有可训练的变量
            tf_weights[v.name] = v.eval(session=sess) # v.name是变量名，v.eval()获取其当前的NumPy值。
    
    return tf_weights # 返回包含所有可训练变量名和对应权重的字典。

# convert_keys: 将 TensorFlow 模型的权重名称字符串转换为 PyTorch 模型中对应层的名称字符串，并识别是否为深度卷积。
def convert_keys(k):
    # 参数:
    #   k (str): TensorFlow中的权重名称 (例如 "generator/A/Conv/weights:0")。
    # 返回:
    #   tuple: (torch_key_name (str), is_depthwise_convolution (bool))

    # 1. 预处理：统一相似层的命名，方便后续分割。
    # 例如 "Conv/" 变为 "Conv_0/"，"LayerNorm/" 变为 "LayerNorm_0/"。
    # 这是因为TensorFlow中同一block内若有多个同类层，可能只用数字区分，或直接用层名。
    k = k.replace("Conv/", "Conv_0/").replace("LayerNorm/", "LayerNorm_0/")    
    # 按 "/" 分割，并去掉前缀 "generator/" 和可能的空字符串或顶层作用域名。
    # keys 通常形如 ['A', 'Conv_0', 'weights:0'] 或 ['C', 'r1', 'Conv_0', 'weights:0']
    keys = k.split("/")[2:]
    
    is_dconv = False # 标记是否为深度卷积的权重

    # 2. 特殊处理 Block C (self.block_c) 中的 InvertedResBlock 内部层命名
    # TensorFlow中InvertedResBlock的层级可能比较扁平或有特定命名（如 'r1', 'r2' ...），
    # PyTorch中则通过 nn.Sequential 和 InvertedResBlock 类有更清晰的层级。
    if keys[0] == "C": # 如果是C块的权重
        # C块中，原始TF模型可能将第6个卷积/归一化命名为 Conv_1/LayerNorm_1 (索引从0开始)
        # 在PyTorch模型中，这对应于 self.block_c 的第6个子模块 (ConvNormLReLU(256, 128))
        # 此处将其映射为索引 "5" (例如 self.block_c[5])
        if keys[1] in ["Conv_1", "LayerNorm_1"]: # 这对应 block_c 的最后一个 ConvNormLReLU(256, 128)
            keys[1] = keys[1].replace("1", "5") # PyTorch: block_c[5]
        
        # 处理 InvertedResBlock 内部的层 (例如 C/r1/Conv_0/weights:0)
        if len(keys) == 4: # 格式如 ['C', 'rX', 'LayerName_Y', 'type:0']
            assert "r" in keys[1] # "rX" 代表第X个 InvertedResBlock

            # 特殊处理 InvertedResBlock 内部的深度卷积 (dw) 和逐点卷积 (pw) 的命名
            # 在PyTorch的InvertedResBlock实现中，层结构通常是 [ConvNormLReLU (升维,可选), ConvNormLReLU (深度), Conv2d (降维), GroupNorm]
            if keys[1] == keys[2]: # 例如 C/r1/r1/weights:0 (TF特定命名方式，可能指深度卷积的权重)
                is_dconv = True # 标记为深度卷积
                # PyTorch: InvertedResBlock.layers[?].1 (假设是深度卷积层)
                # self.block_c[X].layers[1] 是深度卷积 ConvNormLReLU, 其内部的 nn.Conv2d 是 .1
                keys[2] = "1.1" # 指向 InvertedResBlock 中第二个 ConvNormLReLU (深度卷积) 内部的 Conv2d
            
            # TensorFlow中 InvertedResBlock (rX) 内部其他层的映射
            block_c_maps = { # TF层名 -> PyTorch InvertedResBlock内部层索引路径
                "1":        "1.2", # rX/1/gamma:0 -> InvertedResBlock.layers[?].2 (深度卷积的GroupNorm)
                "Conv_1":   "2",   # rX/Conv_1/weights:0 -> InvertedResBlock.layers[2] (1x1降维卷积)
                "2":        "3",   # rX/2/gamma:0 -> InvertedResBlock.layers[3] (1x1降维卷积后的GroupNorm)
            }
            if keys[2] in block_c_maps:
                keys[2] = block_c_maps[keys[2]]

            # 组合PyTorch的键名: block_c.INDEX.layers.SUB_INDEX
            # keys[1] (e.g., "r1") 转换为对应 self.block_c 中 InvertedResBlock 的索引，然后拼接层路径。
            # PyTorch: self.block_c[0] is ConvNormLReLU, self.block_c[1-4] are InvertedResBlocks, self.block_c[5] is ConvNormLReLU.
            # TF 'r1' maps to PyTorch block_c[1], 'r2' to block_c[2], etc.
            # So, "rX" becomes "X.layers.SUB_LAYER_PATH"
            keys[1] = keys[1].replace("r", "") + ".layers." + keys[2] # e.g., "1.layers.1.1" for r1's depthwise conv's weight
            keys[2] = keys[3] # 将权重类型 (weights:0 or bias:0 etc.) 前移
            keys.pop(-1)      # 删除最后一个元素 (原来的权重类型部分)
    assert len(keys) == 3 # 确保最终处理为三部分 [block_name, layer_path, type]

    # 3. 处理输出层 (out_layer)
    if "out" in keys[0]: # 例如 "G_out_Conv_0" (TF命名)
        keys[1] = "0"    # PyTorch: self.out_layer[0] (nn.Conv2d)

    # 4. 转换主块名 (A, B, C, D, E)
    if keys[0] in ["A", "B", "C", "D", "E"]:
        keys[0] = "block_" + keys[0].lower() # e.g., "A" -> "block_a"
        
    # 5. 转换层名和类型 (第二部分 layer_idx 和 第三部分 weight/bias)
    # LayerNorm: "LayerNorm_X" -> "X.2" (指向ConvNormLReLU中的GroupNorm, PyTorch模型用GroupNorm(num_groups=1)实现InstanceNorm/LayerNorm效果)
    if "LayerNorm_" in keys[1]:
        keys[1] = keys[1].replace("LayerNorm_", "") + ".2"
    # Conv: "Conv_X" -> "X.1" (指向ConvNormLReLU中的Conv2d)
    if "Conv_" in keys[1]:
        keys[1] = keys[1].replace("Conv_", "") + ".1"
        
    # 6. 权重类型映射 (TF :0 后缀去除, 名称映射到PyTorch通用名称)
    keys[2] = {
        "weights:0": "weight", # 标准卷积权重
        "w:0":       "weight", # 可能用于某些特定层 (如深度卷积的权重在TF中可能叫'w')
        "bias:0":    "bias",   # 偏置项
        "gamma:0":   "weight", # GroupNorm/LayerNorm的可学习缩放参数 gamma 映射为 weight
        "beta:0":    "bias",   # GroupNorm/LayerNorm的可学习平移参数 beta 映射为 bias
    }[keys[2]]
        
    return ".".join(keys), is_dconv # 用 "." 连接成PyTorch风格的键名，并返回是否为深度卷积


# convert_and_save: 执行完整的权重转换流程，从加载TF权重、转换键名和张量维度，到加载进PyTorch模型并保存。
def convert_and_save(tf_checkpoint_path, save_name):
    # 参数:
    #   tf_checkpoint_path (str): TensorFlow checkpoint的路径。
    #   save_name (str): 转换后的PyTorch权重文件的保存名称。

    # 1. 加载TensorFlow权重
    tf_weights = load_tf_weights(tf_checkpoint_path)
    
    # 2. 实例化PyTorch模型并获取其state_dict结构 (用于名称和形状校验)
    torch_net = Generator() # PyTorch模型实例
    torch_weights_structure = torch_net.state_dict() # PyTorch模型的权重字典

    torch_converted_weights = {} # 用于存放转换后的权重
    # 3. 遍历从TensorFlow加载的每一个权重
    for k, v_np in tf_weights.items(): # k: TF权重名, v_np: TF权重值 (NumPy array)
        # 3.1 转换权重名称
        torch_k, is_dconv = convert_keys(k)
        # 断言：确保转换后的PyTorch键名存在于我们的PyTorch模型中
        assert torch_k in torch_weights_structure, f"PyTorch key name mismatch: TF key '{k}' -> PT key '{torch_k}' not found in PyTorch model."

        # 3.2 将NumPy数组转换为PyTorch张量
        converted_weight = torch.from_numpy(v_np)

        # 3.3 处理卷积核权重的维度重排 (Permutation)
        # TensorFlow 卷积核: [KH, KW, Cin, Cout] (高, 宽, 输入通道, 输出通道)
        # PyTorch 普通卷积核: [Cout, Cin, KH, KW]
        # PyTorch 深度卷积核: [Cout_total, 1, KH, KW] (Cout_total = Cin * ChannelMultiplier)
        if len(converted_weight.shape) == 4: # 通常是卷积核 (4D张量)
            if is_dconv: # 如果是深度卷积
                # TF (depthwise): [KH, KW, Cin_tf, ChannelMultiplier=1 typically for depthwise]
                # PT (depthwise): [Cout_pt=Cin_tf, 1, KH, KW]
                # So, [KH,KW,Cin,1] (TF) -> [Cin,1,KH,KW] (PT)
                # permute(2, 3, 0, 1) from [KH,KW,Cin,CMul] -> [Cin,CMul,KH,KW].
                # If CMul is 1 (true depthwise), this is [Cin,1,KH,KW]. This matches PyTorch.
                converted_weight = converted_weight.permute(2, 3, 0, 1)
            else: # 普通卷积
                # TF: [KH, KW, Cin, Cout] -> PT: [Cout, Cin, KH, KW]
                # permute(3, 2, 0, 1) means: Cout -> dim0, Cin -> dim1, KH -> dim2, KW -> dim3
                converted_weight = converted_weight.permute(3, 2, 0, 1)

        # 断言：确保转换后的权重形状与PyTorch模型中对应权重的形状完全一致
        assert torch_weights_structure[torch_k].shape == converted_weight.shape,             f"Shape mismatch for key '{torch_k}': TF key '{k}', TF shape {v_np.shape}, Expected PT shape {torch_weights_structure[torch_k].shape}, Got {converted_weight.shape}"

        torch_converted_weights[torch_k] = converted_weight # 存入转换后的权重

    # 4. 校验是否所有PyTorch模型的权重都已成功转换 (检查键名列表是否完全一致)
    assert sorted(list(torch_converted_weights.keys())) == sorted(list(torch_weights_structure.keys())),         f"Some weights are missing or extra in the converted set. Check key mapping."

    # 5. 将转换后的权重加载到PyTorch模型实例中
    torch_net.load_state_dict(torch_converted_weights)    

    # 6. 保存PyTorch模型的state_dict到文件
    torch.save(torch_net.state_dict(), save_name)
    print(f"PyTorch model saved at {save_name}")
    

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Convert AnimeGANv2 TensorFlow checkpoint to PyTorch state_dict.")
    # TensorFlow checkpoint目录路径参数
    parser.add_argument(
        '--tf_checkpoint_path',
        type=str,
        default='AnimeGANv2/checkpoint/generator_Paprika_weight', # 默认路径 (需要有原始AnimeGANv2仓库在同级目录)
        help="Path to the TensorFlow model checkpoint directory (e.g., 'AnimeGANv2/checkpoint/generator_Paprika_weight')."
    )
    # PyTorch权重保存文件名参数
    parser.add_argument(
        '--save_name', 
        type=str, 
        default='pytorch_generator_Paprika.pt', # 默认保存名称
        help="Name for the converted PyTorch model weights file (e.g., 'pytorch_generator_Paprika.pt')."
    )
    args = parser.parse_args() # 解析命令行参数
    
    # 调用主转换函数
    convert_and_save(args.tf_checkpoint_path, args.save_name)