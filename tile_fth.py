import math
import simpy
import numpy as np
from typing import List, Union
from util import *  # 导入工具函数
from hardware import Hardware  # 导入硬件类
from macro import *  # 导入宏定义
from op import Op  # 导入操作类
from contextlib import nullcontext  # 导入上下文管理器

class Tile():
    def __init__(self, env, hd_config, st_config, sim_config) -> None:
        # hardware configs 硬件配置
        self.name = hd_config["t_name"]  # Tile名称
        self.t_INT8 = hd_config["t_INT8(TOPs)"]  # INT8计算能力（TOPs）
        self.t_FP16 = hd_config["t_FP16(TFLOPs)"]  # FP16计算能力（TFLOPs）
        self.t_FP32 = hd_config["t_FP32(TFLOPs)"]  # FP32计算能力（TFLOPs）
        self.t_sram_cap = hd_config["t_sram_cap(MB)"]  # SRAM容量（MB）
        self.freq = hd_config["clk_freq(GHz)"]  # 时钟频率（GHz）

        # software configs 软件配置
        self.recompute = st_config["recompute"]  # 是否重新计算
        self.zero = st_config["zero"]  # 是否零初始化
        self.pipeline = st_config["pipeline"]  # 是否使用流水线
        self.opti = st_config["optimizer"]  # 优化器（若无则为推理）
        self.mode = st_config["mode"]  # 模式（训练或推理）
        # self.BYTES = {'NONE': 0, 'INT8': 1, 'FP16': 2, 'TF32': 2.375, 'FP32': 4, 'FP64': 5}  # 数据类型字节数
        # Error correction factor 误差修正系数
        self.tile_factor = {
            'gpu': 0.5,
            'wafer': 0.005,
            'default': 0.1,
        }
        
        # env 仿真环境
        self.env = env  # 仿真环境
        self.analytical = sim_config['analytical']  # 是否使用分析模式
        self.cp_trace = []  # 计算过程记录
        
        if not self.analytical:  # 非分析模式
            self.cp_worker = simpy.Resource(env, capacity=1)  # 计算资源
            self.cm_worker = simpy.Resource(env, capacity=1)  # 通信资源

    # 该函数compute_cycles的主要功能是
    # 根据给定的 参数列表param 和 计算模型cp_model 来计算某种操作的成本。
    def compute_cycles(self, param: List[int]):
        assert(len(param) >= 3)  # 确保参数列表长度至少为3
        cost = 0  # 计算成本初始化为0
        coe = 1 if len(param) == 3 else sizeof(param[:-3], 1)  # 根据参数长度计算系数
        [SR, SC, T] = param[-3:]  # 提取最后三个参数：行数、列数和时间
        [R, C] = self.array_shape  # 提取阵列的行列数
        [PR, PC] = self.array_group  # 提取阵列组的行列数
        
        # 根据计算模型计算成本
        if self.cp_model == comp_model.scale_sim:
            sr = math.ceil(SR / PR)
            sc = math.ceil(SC / PC)
            cost = (2 * R + C + T - 2) * math.ceil(sr / R) * math.ceil(sc / C)
        elif self.cp_model == comp_model.simple:
            cost = T * math.ceil(SR / (R * PR)) * math.ceil(SC / (C * SC))
        elif self.cp_model == comp_model.abrupt_curve:
            cost = T * math.ceil(SR / (R * PR)) * math.ceil(SC / (C * SC))
            if SR % (PR * R) == 0 and SC % (PC * C) == 0:
                cost = 1.0 * cost
            elif SR % (PR * R) == 0:
                cost = 1.2 * cost
            elif SR % R == 0 and SC % C == 0:
                cost = 1.8 * cost
            elif SR % R == 0:
                cost = 2.0 * cost
            else:
                cost = 2.5 * cost
            return int(cost)
        else:
            raise NotImplementedError
        
        return cost * coe

    def computation(self, macs_m, compute_power_t):
        exetime = 2 * macs_m * Ks / compute_power_t  # 计算执行时间
        # exetime = self.compute_cycles(macs_m) / self.freq 
        if not self.analytical:
            with self.cp_worker.request() as req:  # 请求计算资源
                yield req  # 等待资源可用
                t_last = self.env.now  # 记录当前时间
                yield self.env.timeout(exetime)  # 执行计算
                self.cp_trace.append(self.env.now - t_last)  # 记录计算时间
                # print(self.env.now - t_last)
        else:
            yield self.env.timeout(exetime)  # 执行计算
            self.cp_trace.append(exetime)  # 记录计算时间


    
    # 定义tile_dataflow函数，用于确定Tile的数据流策略。参数包括所有操作、Tile列表和流水线并行因子。
    def tile_dataflow(self, all_ops, tiles, pp_infs: int = 1):
        tile_num = len(tiles)  # 获取tile的数量
        t_sgy = {}
        t_sgy['dram_cap_req_gb_max'] = 0  # 最大DRAM容量需求，初始化为0
        ops_size = 0  # 操作总大小，初始化为0
        i_size, w_size, o_size, r_size = 0, 0, 0, 0  # 输入、权重、输出和中间结果的大小，初始化为0
        t_sgy['ior_wsg_bytes'] = [1, 1, 1, 1, 1, 1]  # 默认的IO和权重字节数
        sram_weight_bytes = 1  # 默认的SRAM权重字节数

        # 根据模式和优化器类型设置相应的字节数和计算能力
        if self.mode == mode.INT8 or self.opti == optimizer.none:
            t_sgy['ior_wsg_bytes'] = [1, 1, 0, 1, 0, 0]
            sram_weight_bytes = 1
            t_sgy['compute_power_t'] = self.t_INT8
        elif self.mode == mode.FP16 and self.opti == optimizer.SGD:
            t_sgy['ior_wsg_bytes'] = [2, 2, 2, 2, 0, 2]
            sram_weight_bytes = 2
            t_sgy['compute_power_t'] = self.t_FP16
        elif self.mode == mode.FP16 and self.opti == optimizer.adam:
            t_sgy['ior_wsg_bytes'] = [2, 2, 2, 2 + 4, 4 + 4, 2]
            sram_weight_bytes = 2
            t_sgy['compute_power_t'] = self.t_FP16
        elif self.mode == mode.FP32 and self.opti == optimizer.adam:
            t_sgy['ior_wsg_bytes'] = [4, 4, 4, 4, 4 + 4, 4]
            sram_weight_bytes = 4
            t_sgy['compute_power_t'] = self.t_FP32
        elif self.mode == mode.FP32 and self.opti == optimizer.SGD:
            t_sgy['ior_wsg_bytes'] = [4, 4, 4, 4, 0, 4]
            sram_weight_bytes = 4
            t_sgy['compute_power_t'] = self.t_FP32

        t_sgy['sram_weight_bytes'] = sram_weight_bytes  # 设置SRAM权重字节数
        t_sgy['transformer_num'] = 0  # Transformer的数量，初始化为0
        op_cnt = 0  # 操作计数器，初始化为0

        # 遍历所有操作，计算各个部分的大小
        for op in all_ops:
            if op.type == OP.Encoder:
                t_sgy['transformer_num'] += 1  # 如果操作类型是Encoder，Transformer数量加1
            if op_cnt == 0:
                i_size += (op.iwor_size[0]) * t_sgy['ior_wsg_bytes'][0]  # 累加输入大小
            w_size += (op.iwor_size[1]) * t_sgy['ior_wsg_bytes'][3]  # 累加权重大小
            o_size += (op.iwor_size[2]) * t_sgy['ior_wsg_bytes'][1]  # 累加输出大小
            r_size += (op.iwor_size[3]) * t_sgy['ior_wsg_bytes'][2]  # 累加中间结果大小
            ops_size += op.iwor_size[1]  # 累加操作总大小
            op_cnt += 1

        # 设置重新计算系数
        ior_recompute_coe = [1, 1, 1]
        if t_sgy['transformer_num'] > 0:
            if self.recompute == recompute.full:
                ior_recompute_coe = [1 / t_sgy['transformer_num'], 1 / t_sgy['transformer_num'], 1 / t_sgy['transformer_num']]
            elif self.recompute == recompute.one:
                ior_recompute_coe = [1, 1 / t_sgy['transformer_num'], 1 / t_sgy['transformer_num']]
            elif self.recompute == recompute.half:
                half = math.ceil(t_sgy['transformer_num'] / 2) / t_sgy['transformer_num']
                ior_recompute_coe = [half, half, half]

        # 计算DRAM容量需求
        t_sgy['dram_cap_req_gb_max'] = tile_num * (i_size * ior_recompute_coe[0] * pp_infs + w_size + o_size * ior_recompute_coe[1] * pp_infs + r_size * ior_recompute_coe[2] * pp_infs) * GB

        # 计算SRAM容量和其他相关大小
        sram_cap = tile_num * self.t_sram_cap / 1024
        sram_weight_size = ops_size * GB * sram_weight_bytes * tile_num
        act_size = (i_size + o_size + r_size) * GB * tile_num

        # 确定数据流策略
        t_sgy['dataflow'] = None
        if sram_weight_size < sram_cap:
            t_sgy['dataflow'] = dataflow.ActStream
        elif act_size < sram_cap:
            t_sgy['dataflow'] = dataflow.WeightStream
        else:
            t_sgy['dataflow'] = dataflow.Stationary

        t_sgy['dram_cap_req_gb_max'] += (i_size + r_size) * GB

        # 打印和返回Tile策略信息
        tile_info = 'sram_cap={:.3f} GB, sram_weight_size={:.3f} GB, act_size={:.3f} GB, '.format(sram_cap, sram_weight_size, act_size)
        tile_info += 'dram_cap_req={:.3f} GB, '.format(t_sgy['dram_cap_req_gb_max'])
        tile_info += 'ops_size_all_tile ={:.3f} B, '.format(ops_size * Gs * tile_num)
        tile_info += 'dataflow={}, '.format(t_sgy['dataflow'])
        print(t_sgy)
        print(tile_info)
        return t_sgy

    

    def op_events(self,op:Op,hd:Hardware,t_sgy,state=state.forward):
        compute_power_t=t_sgy['compute_power_t']
        sbytes=t_sgy['sram_weight_bytes']
        dbytes=t_sgy['ior_wsg_bytes']
        df=t_sgy['dataflow']
        factor=self.tile_factor[hd.name]
        events=[]
        transformer_num=t_sgy['transformer_num']
        transformer_cnt=0
        iwor=op.iwor_size
        #dbytes[0]+dbytes[1]+dbytes[2]+dbytes[3]+dbytes[4]++dbytes[5]
        while True:
            if state==state.forward:
                macs_m=op.fd_macs*Ms
                write,read=0,0
                if df==dataflow.ActStream:
                    write=(iwor[2]*dbytes[1]+iwor[3]*dbytes[2])*MB
                    read=(iwor[2]*dbytes[0]+iwor[3]*dbytes[2])*MB
                elif df==dataflow.WeightStream:
                    write,read=0,(iwor[1])*dbytes[3]*MB
                else:
                    if op.type!=OP.Encoder:
                        write=iwor[2]*dbytes[1]*MB
                        read=min(math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)*iwor[1]*dbytes[3]*MB,\
                                 math.ceil(iwor[1]*dbytes[3]*MB/self.t_sram_cap)*iwor[0]*dbytes[0]*MB)
                    else:
                        coe=factor*math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)
                        write=coe*(iwor[2]*dbytes[1]+iwor[3]*dbytes[2])*MB
                        read=coe*(iwor[0]*dbytes[0]+iwor[1]*dbytes[3]+iwor[3]*dbytes[2])*MB
                events.append(self.env.process(self.computation(macs_m,compute_power_t)))
                events.append(self.env.process(hd.tile_gd_access(write,read,op.devices,write=1,read=1)))
                events=[simpy.AllOf(self.env, events)]
                if op.d4d_comm['f']!=[]:
                    events.append(self.env.process(self.communication(op.d4d_comm['f'],dbytes[0],hd)))
                    
            elif state==state.backward:
                times_cp=2
                if op.type==OP.Encoder:
                    transformer_cnt+=1
                    if (transformer_cnt % 2 ==1 and self.recompute==recompute.half) or self.recompute==recompute.full:
                        times_cp=3
                macs_m=times_cp*op.fd_macs*Ms
                if df==dataflow.ActStream:
                    write=(iwor[2]*dbytes[0]+iwor[3]*dbytes[2])*MB
                    read=(iwor[2]*dbytes[0]+iwor[3]*dbytes[2])*MB+(iwor[2]*dbytes[1]+iwor[3]*dbytes[2])*MB
                elif df==dataflow.WeightStream:
                    write,read=0,(iwor[1])*dbytes[3]*MB
                else:
                    if op.type!=OP.Encoder:
                        write=iwor[0]*dbytes[0]*MB
                        read=min(math.ceil(iwor[2]*dbytes[1]*MB/self.t_sram_cap)*iwor[1]*dbytes[3]*MB,\
                                 math.ceil(iwor[1]*dbytes[3]*MB/self.t_sram_cap)*iwor[2]*dbytes[1]*MB)
                        read+=min(math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)*iwor[3]*dbytes[2]*MB,\
                                 math.ceil(iwor[3]*dbytes[2]*MB/self.t_sram_cap)*iwor[0]*dbytes[0]*MB)
                    else:
                        coe=factor**math.ceil(iwor[0]*dbytes[0]*MB/self.t_sram_cap)
                        write=coe*(iwor[3]*dbytes[2]+iwor[0]*dbytes[0])*MB
                        read=coe*(iwor[0]*dbytes[0]+iwor[1]*dbytes[3]+2*iwor[3]*dbytes[0])*MB
                events.append(self.env.process(self.computation(macs_m,compute_power_t)))
                events.append(self.env.process(hd.tile_gd_access(write,read,op.devices,write=1,read=1)))
                events=[simpy.AllOf(self.env, events)]
                if op.d4d_comm['b']!=[]:
                    events.append(self.env.process(self.communication(op.d4d_comm['b'],sbytes,hd)))
                        
            elif state==state.param_sync:
                events.append(self.env.process(self.communication(op.d4d_comm['u'],sbytes,hd)))
                events=[simpy.AllOf(self.env, events)]
                write,read=(iwor[1])*(dbytes[3])*MB,(iwor[1])*(dbytes[3]+dbytes[4]+dbytes[4])*MB
                events.append(self.env.process(hd.tile_gd_access(write,read,op.devices,write=1,read=1))) 
            else:#recompute
                pass
            yield simpy.AllOf(self.env, events)
            break

    def ops_events(self,ops,hd:Hardware,t_sgy,state=state.forward):
        while(True):
            for op in ops:
                yield  self.env.process(self.op_events(op,hd,t_sgy,state))
            break

if __name__ == "__main__":
    pass
