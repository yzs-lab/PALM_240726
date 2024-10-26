from enum import Enum  # 导入枚举类模块

# 自定义枚举类的基础类，以便在打印和显示时使用名称
class Enum(Enum):
    def __str__(self):
        return self.name  # 返回枚举项的名称
    def __repr__(self):
        return f'{self.name}'  # 返回枚举项的名称

# 定义一些全局常量
ONE_WEEK_MS = 24 * 60 * 60 * 7 * 1000  # 一周的毫秒数
INT_MAX = 2**64 - 1  # 最大的64位整数
# 数据存储单位
TB = 1 / (1024**4)  # TB到字节的转换因子
GB = 1 / (1024**3)  # GB到字节的转换因子
MB = 1 / (1024**2)  # MB到字节的转换因子
KB = 1 / 1024  # KB到字节的转换因子
# 时间单位
Ts = 1 / (1000**4)  # 太秒到秒的转换因子
Gs = 1 / (1000**3)  # 吉秒到秒的转换因子
Ms = 1 / (1000**2)  # 毫秒到秒的转换因子
Ks = 1 / 1000  # 千秒到秒的转换因子
ms = 1000  # 秒到毫秒的转换因子
us = 1000 * 1000  # 秒到微秒的转换因子
h = 1 / 60  # 小时到分钟的转换因子

# 定义操作类型的枚举   包括线性操作、卷积操作、嵌入操作等。
OP = Enum('OP', ('Linear', 'Conv2', 'Embedding', 'Softmax', 'LayerNorm', 'Encoder', 'Pool', 'Concat', 'Sum'))
# 定义通信类型的枚举  包括不同的通信操作。
COMM = Enum('COMM', ('NONE', 'AR', 'AA', 'AG', 'RS'))
# 定义优化器类型的枚举  包括无优化器、SGD和Adam。
optimizer = Enum('optimizer', ('none', 'SGD', 'adam'))
# 定义模式类型的枚举  包括INT8、FP16和FP32。
mode = Enum('mode', ('INT8', 'FP16', 'FP32'))
# 定义状态类型的枚举  包括前向、后向、参数同步和重新计算。
state = Enum('state', ('forward', 'backward', 'param_sync', 'recompute'))
# 定义数据流类型的枚举  包括不同的数据流策略。
dataflow = Enum('dataflow', ('IS', 'WS', 'WeightStream', 'ActStream', 'Stationary'))
# 定义计算模型类型的枚举
comp_model = Enum('comp_model', ('simple', 'scale_sim', 'abrupt_curve'))

# 定义存储类型的枚举  包括缓存、权重、激活、激活权重和无。
store = Enum('store', ('cache', 'weight', 'ACT', 'ACT_weight', 'none'))
# 定义重新计算类型的枚举  包括无、一次、半次和全部重新计算。
recompute = Enum('recompute', ('none', 'one', 'half', 'full'))

# 定义流水线类型的枚举  定义了流水线类型的枚举，包括GPipe、Dreampipe1F1B和Interleaved1F1B。
pipe = Enum('pipe', ('GPipe', 'Dreampipe1F1B', 'Interleaved1F1B'))
# 定义零初始化类型的枚举
zero = Enum('zero', ('none', 's1', 's2', 's3', 'sr'))

# 定义事件类型的枚举  包括激活存储、激活获取、通信、激活前向、梯度获取、梯度存储、权重加载、权重存储、优化器加载、优化器存储、损失加载和损失存储。
event = Enum('event', ('act_store', 'act_fetch', 'comm', 'act_fd', 'grad_fetch', 'grad_store', 'wt_load', 'wt_store', 'opt_load', 'opt_store', 'dloss_load', 'dloss_store'))


def str2enum(str, type='OP'):
    try:
        if type == 'OP':
            enum_value = OP[str]  # 如果类型为'OP'，则从OP枚举中获取对应的枚举值
        elif type == 'recompute':
            enum_value = recompute[str]  # 如果类型为'recompute'，则从recompute枚举中获取对应的枚举值
        elif type == 'pipe':
            enum_value = pipe[str]  # 如果类型为'pipe'，则从pipe枚举中获取对应的枚举值
        elif type == 'zero':
            enum_value = zero[str]  # 如果类型为'zero'，则从zero枚举中获取对应的枚举值
        else:
            raise NotImplementedError  # 如果传入的类型不在上述范围内，则抛出NotImplementedError异常
        return enum_value  # 返回获取到的枚举值
    except KeyError:
        raise NotImplementedError  # 如果枚举中没有找到对应的字符串，抛出NotImplementedError异常
