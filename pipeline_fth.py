import simpy
import math
import time
from visualize import *  # 导入可视化模块
from util import *  # 导入工具函数模块
from tile import *  # 导入Tile类模块
from hardware import *  # 导入硬件相关模块
from macro import *  # 导入宏定义模块
import numpy as np  # 导入NumPy库

class Stage():
    __stage_id = 0  # 类变量，记录Stage实例的数量

    def __init__(self, env, st_config, hd_config, sim_config, ops, L_devices, C_devices, N_devices) -> None:
        self.ops = ops  # 操作列表
        self.i_size_mb = 0  # 输入数据大小（MB）
        self.o_size_mb = 0  # 输出数据大小（MB）
        self.N_devices = N_devices  # 网络设备
        self.L_devices = L_devices  # 本地设备
        self.C_devices = C_devices  # 计算设备
        self.tile = []  # Tile列表
        
        # 根据C_devices的数量创建Tile实例，并添加到tile列表中
        for i in range(len(self.C_devices)):
            self.tile.append(Tile(env, hd_config, st_config, sim_config))
            if sim_config['tile_aggregate']:  # 如果配置为tile聚合模式，则创建一个Tile即可
                break

        self.stage_info = []  # 阶段信息列表
        self.tile_strategy = None  # Tile策略
        
        # simpy环境变量
        self.env = env  # 仿真环境
        self.qu = simpy.PriorityResource(env, capacity=1)  # 优先级资源
        self.trace = []  # 记录仿真事件的踪迹
        self.fd_cnt = 0  # 记录前向传播的次数
        self.__class__.__stage_id += 1  # 每创建一个Stage实例，stage_id加1

    def tiles_init(self, micro_batch, cur_stage, stage_num, pipe_strategy, train):
        for op in self.ops:  # 初始化每个操作
            op.dims[0] = micro_batch  # 设置操作的微批次大小
            op.analysis()  # 分析操作

        # 设置流水线并行因子
        pp_infs = stage_num - cur_stage if pipe_strategy == pipe.Dreampipe1F1B else stage_num
        pp_infs = pp_infs if train else 1
        
        # 计算Tile的数据流策略
        self.tile_strategy = self.tile[0].tile_dataflow(self.ops, self.C_devices, pp_infs=pp_infs)
        
        # 计算输入和输出数据的大小（MB）
        self.i_size_mb = sizeof(self.ops[0].i_shape, 2) * MB
        self.o_size_mb = sizeof(self.ops[-1].o_shape, 2) * MB

    def up_state(self, hd, c_type=state.forward, wait=1e-15):
        while True:
            with self.qu.request() as req:  # 请求资源
                yield req  # 等待资源可用
                t_last = self.env.now  # 记录当前时间
                
                # 创建并启动Tile操作事件的仿真进程
                event_list = [self.env.process(tile.ops_events(self.ops, hd, self.tile_strategy, c_type)) for tile in self.tile]
                yield self.env.all_of(event_list)  # 等待所有事件完成
                
                # 记录事件的开始和结束时间
                self.trace.append((t_last, self.env.now, c_type))
                
                # 根据事件类型更新前向传播计数
                if c_type == state.forward:
                    self.fd_cnt += 1
                elif c_type == state.backward:
                    self.fd_cnt -= 1

            break  # 退出循环


# 定义一个Pipeline类，用于模拟深度学习的流水线处理。               
class Pipeline():
    def __init__(self, env, stage_devices, stage_ops, hardware, hd_config, st_config, sim_config, pipe_config) -> None:
        # 初始化Pipeline类的构造函数，接收多个配置参数

        #simpy env 
        self.env = env
        # simpy环境变量，用于事件模拟

        self.reg = []
        # 初始化寄存器，用于存储各阶段的数据

        self.cur_fd_times = 0
        self.cur_bd_times = 0
        # 当前前向传播和反向传播的次数

        self.one_epoch_finish = simpy.Store(self.env, capacity=1)
        self.one_fd_finish = simpy.Store(self.env, capacity=1)
        self.one_data_fetch = simpy.Store(self.env, capacity=1)
        # simpy的Store，用于同步各个阶段的事件

        #stage
        self.stage_num = len(stage_ops)
        # 阶段的数量，由stage_ops的长度决定

        self.hd = hardware
        # 硬件配置

        self.mini_batch = pipe_config['mini_batch_size']
        self.micro_batch = pipe_config['micro_batch_size']
        self.micro_batch_num = math.ceil(self.mini_batch / self.micro_batch)
        # mini_batch和micro_batch的大小配置，以及micro_batch的数量

        self.stages = []
        # 初始化阶段列表

        for i in range(self.stage_num):
            self.stages.append(Stage(
                self.env, st_config, hd_config, sim_config, stage_ops[i],
                L_devices=None if i == 0 else stage_devices[i - 1],
                C_devices=stage_devices[i],
                N_devices=stage_devices[i + 1] if i < self.stage_num - 1 else None))
            # 根据阶段的数量初始化每个Stage对象，传入相应的配置和设备

        self.pipe_strategy = st_config['pipeline']
        self.st_config = st_config
        # 管道策略和阶段配置

        #sim config
        self.boost_mode = sim_config['pipe_boost']
        self.train = (st_config['optimizer'] != optimizer.none) and (st_config['mode'] != mode.INT8)
        self.boost_times = 6
        # 是否启用加速模式，训练模式以及加速次数配置

        #dram analysis
        self.strategy = {}
        # 初始化策略字典，用于存储各阶段的DRAM请求

        self.__set_stage()
        # 调用私有方法__set_stage，初始化各阶段

    def __set_stage(self):
        for cur_stage in range(self.stage_num):
            self.reg.append(simpy.PriorityStore(self.env, capacity=self.stage_num - cur_stage))
            # 为每个阶段初始化优先级存储器，容量递减

            self.stages[cur_stage].tiles_init(self.micro_batch, cur_stage, self.stage_num, self.pipe_strategy, self.train)
            # 初始化每个阶段的tiles配置，传入micro_batch和其他配置

            self.strategy['stage_' + str(cur_stage) + '_dram_req'] = self.stages[cur_stage].tile_strategy['dram_cap_req_gb_max']
            # 将当前阶段的DRAM请求最大值记录到strategy字典中

        #print(self.strategy)
        # 打印策略字典（注释掉）

           
    def forward(self, times):
        with self.one_data_fetch.get() as get:
            a = yield get
            # 等待数据获取完成
            
            for i, stg in enumerate(self.stages):
                # 遍历每个阶段
                
                if self.pipe_strategy == pipe.Dreampipe1F1B and self.train:
                    yield self.reg[i].put(1)
                    # 如果策略是Dreampipe1F1B并且在训练，将1放入当前阶段的寄存器
                    
                elif self.pipe_strategy == pipe.GPipe or not self.train:
                    if i == self.stage_num - 1:
                        self.cur_fd_times += 1
                        # 如果策略是GPipe或者不在训练，并且是最后一个阶段，增加前向传播次数
                        
                    if self.cur_fd_times == times:
                        self.one_fd_finish.put(1)
                        # 如果前向传播次数达到指定值，标记前向传播完成
                        
                else:
                    raise NotImplementedError
                    # 如果策略未实现，抛出异常
                    
                yield self.env.process(stg.up_state(self.hd, c_type=state.forward, wait=1e-15))
                # 更新当前阶段状态为前向传播
                
                if stg.N_devices is not None:
                    yield self.env.process(self.hd.stage_data_tranfer(stg.C_devices, stg.N_devices, stg.o_size_mb, ar_flag=True))
                    # 如果有下一阶段设备，进行数据传输
                    
    def backward(self, times): 
        for i in range(self.stage_num - 1, -1, -1):
            # 逆向遍历每个阶段
            
            if self.pipe_strategy == pipe.Dreampipe1F1B:
                with self.reg[i].get() as get:
                    a = yield get
                    # 等待从寄存器中获取数据
                    
                    stg = self.stages[i]
                    yield self.env.process(stg.up_state(self.hd, c_type=state.backward, wait=1e-15))
                    # 更新当前阶段状态为反向传播
                    
                    if stg.L_devices is not None:
                        yield self.env.process(self.hd.stage_data_tranfer(stg.C_devices, stg.L_devices, stg.i_size_mb, ar_flag=True))
                        # 如果有上一阶段设备，进行数据传输
                        
                    if i == 0:
                        self.cur_bd_times += 1
                        # 如果是第一个阶段，增加反向传播次数
                        
                    if self.cur_bd_times == times:
                        self.one_epoch_finish.put(1)
                        # 如果反向传播次数达到指定值，标记一个epoch完成
                        
            elif self.pipe_strategy == pipe.GPipe:
                stg = self.stages[i]
                yield self.env.process(stg.up_state(self.hd, c_type=state.backward, wait=1e-15))
                # 更新当前阶段状态为反向传播
                
                if stg.L_devices is not None:
                    yield self.env.process(self.hd.stage_data_tranfer(stg.C_devices, stg.L_devices, stg.i_size_mb, ar_flag=True))
                    # 如果有上一阶段设备，进行数据传输
                    
                if i == 0:
                    self.cur_bd_times += 1
                    # 如果是第一个阶段，增加反向传播次数
                    
                if self.cur_bd_times == times:
                    self.one_epoch_finish.put(1)
                    # 如果反向传播次数达到指定值，标记一个epoch完成
                    
    def parameter_syn(self):
        while True:
            # parameter_syn 在检测到 one_epoch_finish 事件完成后，立即启动所有阶段的参数同步过程up_state
            with self.one_epoch_finish.get() as get:
                a = yield get
                # 等待一个epoch完成
                
                for stg in self.stages:
                    self.env.process(stg.up_state(self.hd, c_type=state.param_sync, wait=1e-15))
                    # 同步每个阶段的参数
                    
                break
                # 完成后跳出循环
                
    def start(self):
        times = self.boost_times if self.boost_mode else self.micro_batch_num
        # 确定循环次数，根据是否启用加速模式选择boost_times或micro_batch_num
        
        for i in range(times):
            with self.one_data_fetch.put(1) as put:
                yield put
                # 放入数据获取的标记
                
                yield self.env.process(self.hd.tile_gd_access(self.stages[0].i_size_mb, 0, self.stages[0].C_devices, write=0, read=1))
                # 执行数据访问，读取第一阶段的输入数据大小

        
    def register(self): 
        print('----------pipe_info----------')
        print('stage num={}, extute times={}'.format(len(self.stages),self.micro_batch_num))
        print('mini batch={}, micro batch={}'.format(self.mini_batch,self.micro_batch))
        self.boost_times = min(self.boost_times, self.micro_batch_num)
        # 确保 boost_times 不超过 micro_batch_num
        times = self.boost_times if self.boost_mode else self.micro_batch_num
        # 根据是否启用加速模式选择循环次数

        def all_backward(times):
            while True:
                with self.one_fd_finish.get() as get:
                    a = yield get    
                    # 等待前向传播完成
                    
                    for i in range(times):
                        self.env.process(self.backward(times)) 
                        # 启动反向传播进程
                    
                break
                # 完成后跳出循环

        self.env.process(self.start())
        # 启动 start 进程

        for i in range(times):
            self.env.process(self.forward(times))
            # 启动前向传播进程
        
        if self.train:
            # 如果处于训练模式
            if self.pipe_strategy == pipe.GPipe:  
                self.env.process(all_backward(times))
                # 启动所有的反向传播进程
            elif self.pipe_strategy == pipe.Dreampipe1F1B:  
                for i in range(times):
                    self.env.process(self.backward(times))
                    # 启动每个 micro batch 的反向传播进程
            
            self.env.process(self.parameter_syn())
            # 启动参数同步进程

    def simpy_run(self, until_ms=2000):
        self.register()
        # 注册所有进程
        print('----------simpy_run----------')
        sim_start_t = time.time()
        print('start simpy simulation...')
        self.env.run(until=until_ms)
        # 运行 simpy 模拟，直到指定时间
        sim_end_t = time.time()
        print('finish simpy simulation with {:.3f}s\n'.format(sim_end_t - sim_start_t))


    def sim_visualize(self, path='./sim_visualize/pipeline/', draw_pipe=True, write_log=False, clear=True):
        results = ''
        exe_mode = 'training' if self.train else 'inference'
        # 确定执行模式是训练还是推理
        
        tm = time.strftime('_%m_%d_%H_%M_%S', time.localtime())
        # 获取当前时间，格式为 '_月_日_小时_分钟_秒'
        
        name = 'pipeline' + str(tm)
        name_log = name + '.log'
        # 设置文件名和日志文件名
        
        all_trace = []
        utilization = [0] * len(self.stages)
        utilization_tile_cp = [0] * len(self.stages)
        tile_num_list = [0] * len(self.stages)
        pipe_endtime = 0
        title = str(self.pipe_strategy) if self.train else 'Inference'
        # 初始化各种变量
        
        for i, stage in enumerate(self.stages):
            all_trace.append(stage.trace)
            # 收集每个阶段的跟踪信息
            
            if stage.trace[-1][1] > pipe_endtime:
                pipe_endtime = stage.trace[-1][1]
                # 获取流水线结束时间
            
            for item in stage.trace:
                utilization[i] += (item[1] - item[0])
                # 计算每个阶段的利用率
            
            tile_num_list[i] = len(stage.C_devices)
            # 获取每个阶段的设备数量
        
        corr_coe = 0
        if self.boost_mode:
            # 如果启用加速模式
            
            max_unit_time_1F_1B = max_ave_1F_1B_time(all_trace, self.train)
            # 计算 1F 1B 的最大单位时间
            
            add_time = (self.micro_batch_num - self.boost_times) * max_unit_time_1F_1B 
            pipe_endtime = pipe_endtime + add_time
            # 计算增加的时间，并更新流水线结束时间
            
            corr_coe = (self.micro_batch_num - self.boost_times) / self.boost_times
            # 计算校正系数
            
            for i in range(len(utilization)):
                utilization[i] += (utilization[i] * corr_coe)
                utilization_tile_cp[i] += (sum(self.stages[i].tile[0].cp_trace) * (corr_coe + 1))
                utilization_tile_cp[i] /= pipe_endtime
                utilization[i] /= pipe_endtime
                # 校正利用率和计算利用率，并计算每个阶段的利用率
            
        else:
            for i in range(len(utilization)):
                utilization[i] /= pipe_endtime
                utilization_tile_cp[i] += (sum(self.stages[i].tile[0].cp_trace))
                utilization_tile_cp[i] /= pipe_endtime
                # 计算每个阶段的利用率
        
        mini_batch = self.micro_batch_num * self.micro_batch
        endtime_secs = pipe_endtime / 1000
        endtime_days = endtime_secs / 60 / 60 / 24
        # 计算 mini_batch 数量，流水线结束时间（秒和天）
        
        if not os.path.exists(path):
            os.makedirs(path)
        elif clear:
            ls = os.listdir(path)
            for i in ls: 
                f_path = os.path.join(path, i)
                os.remove(f_path)
                # 如果路径不存在则创建，否则清空路径
        
        if write_log:
            with open(path + name_log, 'w') as f:
                for i in range(len(all_trace)):
                    f.write(str(all_trace[i]))
                    f.write('\n')
                # 如果需要写日志，将跟踪信息写入日志文件
        
        if draw_pipe:
            draw_pipeline(all_trace, path=path, title=title, throughout=mini_batch/endtime_secs, name=name)
            # 如果需要绘制流水线图，调用 draw_pipeline 函数
        
        draw_dram_req(self.strategy, path=path, name=title, info=self.st_config)
        # 绘制 DRAM 请求图
        
        results += '{} {} pipeline endtime {:.4f} days [{:.4f}s]\n'.format(title, exe_mode, endtime_days, endtime_secs)
        results += '{} {} pipeline throughout= {:.4f} sample/s\n'.format(title, exe_mode, mini_batch / endtime_secs)
        results += draw_util(utilization, 'utilization', path=path)
        results += draw_util(utilization_tile_cp, 'computational_utilization', path=path)
        # 生成结果字符串，包括流水线结束时间和利用率
        
        return results
        # 返回结果字符串


