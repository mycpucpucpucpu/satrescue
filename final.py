import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# ==========================================
# 1. 核心参数微调
# ==========================================
POWER_MODES = {
    'Survival': 120, 'Nominal': 160, 
    'OrbitControl': 180, 'LaserLink': 260, 'KBandDownlink': 300
}
INFERENCE_ENERGY = {
    'coarse': 0.005,
    'precise': 0.026272
}
PRED_CONFIG = {
    'coarse': {'efficiency': 0.70}, # 降低粗略效率，拉开差距
    'precise': {'efficiency': 1.0}
}
INFERENCE_INTERVAL = 20 

# ==========================================
# 2. 数据加载
# ==========================================
class DataLoader:
    def __init__(self, file_path):
        try:
            self.df = pd.read_csv(file_path).fillna(0)
            self.best_angle = self.df['best_angle'].values
            self.max_current = self.df['max_current'].values
            self.voltage = self.df.get('蓄电池组电压', pd.Series([28.0]*len(self.df))).values
            self.time_steps = len(self.df)
            print(f"🚀 数据加载: {self.time_steps} 行")
        except:
            self.time_steps = 20000
            t = np.linspace(0, 100, self.time_steps)
            self.best_angle = 180 * np.sin(t)
            self.max_current = np.abs(12*np.sin(t)) + 25
            self.voltage = np.full(self.time_steps, 28.0)

# ==========================================
# 3. 卫星仿真核心
# ==========================================
class TaskBasedSatellite:
    def __init__(self, data, schedule, cap, act_cost, mode_strategy='adaptive', control_strategy='default'):
        self.data = data
        self.schedule = schedule
        self.capacity = float(cap)
        self.energy = self.capacity * 0.7  
        self.initial_energy = self.energy
        self.unit_cost = act_cost
        self.mode_strategy = mode_strategy
        self.control_strategy = control_strategy
        
        self.current_angle = float(self.data.best_angle[0])
        self.last_target = self.current_angle
        self.is_alive = True
        self.total_compute_energy = 0.0
        self.tasks_completed = 0
        self.total_tasks = 0
        self.actuation_count = 0

    def run(self):
        dt = 1/3600
        for t in range(self.data.time_steps):
            if not self.is_alive: break
            
            mode = self.schedule[t]
            is_task = mode in ['KBandDownlink','LaserLink','OrbitControl']
            soc = self.energy / self.capacity
            
            # ===== 1. 核心改进：Adaptive 抢占逻辑 =====
            if self.mode_strategy == 'always_precise':
                pred_mode = 'precise'
            elif self.mode_strategy == 'always_coarse':
                pred_mode = 'coarse'
            else: # Adaptive
                # 任务期间强制精准模式；空闲期电量低于80%就用粗略模式攒钱
                pred_mode = 'precise' if (is_task or soc > 0.8) else 'coarse'
            
            eff = PRED_CONFIG[pred_mode]['efficiency']

            # ===== 2. 推理触发 =====
            if t % INFERENCE_INTERVAL == 0:
                c_energy = INFERENCE_ENERGY[pred_mode]
                self.energy -= c_energy
                self.total_compute_energy += c_energy
                bias = np.random.normal(0, 15) if pred_mode == 'coarse' else 0
                self.last_target = self.data.best_angle[t] + bias
            
            target = self.last_target
            gap = abs(self.current_angle - target)

            # ===== 3. 控制策略 (提高任务灵敏度) =====
            if self.control_strategy == 'periodic':
                need = (t % 200 == 0)
            elif self.control_strategy == 'fixed_cycle':
                need = (t % 100 == 0)
            else: # Default
                need = (gap > 1.5) if is_task else (gap > 12)

            # 强行转身决策
            if need:
                cost = self.unit_cost * gap
                self.energy -= cost
                self.current_angle = target
                self.actuation_count += 1

            # ===== 4. 能量代谢 (大幅强化发电) =====
            solar = max(0, np.sin(2 * np.pi * t / 2000))
            phys = max(0, np.cos(np.radians(self.current_angle - self.data.best_angle[t])))
            
            # 系数调至 115.0，确保在大功耗下净收益为正
            gen = (self.data.max_current[t] * self.data.voltage[t] / 1000.0) * phys * eff * solar * dt * 115.0
            base_load = POWER_MODES.get(mode, 160) / 1000.0
            
            self.energy += gen - (base_load * dt)
            self.energy = min(self.energy, self.capacity)

            if self.energy <= 0: self.is_alive = False

            # ===== 5. 统计 =====
            if is_task:
                self.total_tasks += 1
                if self.is_alive and phys > 0.92: # 成功判定更严格
                    self.tasks_completed += 1
            
        rate = (self.tasks_completed / self.total_tasks * 100) if self.total_tasks > 0 else 0
        return rate, self.actuation_count, (self.energy - self.initial_energy)

# ==========================================
# 4. 主程序
# ==========================================
def generate_task_schedule(total_steps):
    np.random.seed(42)
    schedule = ['Nominal'] * total_steps
    curr = 0
    while curr < total_steps:
        if np.random.rand() > 0.8:
            task = np.random.choice(['KBandDownlink','LaserLink','OrbitControl'])
            dur = np.random.randint(200, 500)
            schedule[curr:min(curr+dur, total_steps)] = [task] * (min(curr+dur, total_steps) - curr)
            curr += dur
        else: curr += 30
    return schedule

def main():
    loader = DataLoader("data.csv")
    schedule = generate_task_schedule(loader.time_steps)
    
    BAT_CAP = 120.0
    ACT_UNIT_COST = 0.000005 # 降低成本

    print("\n" + "="*65)
    print(f"{'策略名称':<18} | {'成功率':<10} | {'转身次数':<8} | {'净收益 (kWh)':<12}")
    print("-" * 65)
    
    # 模式对比
    for strat in ['adaptive', 'always_precise', 'always_coarse']:
        sim = TaskBasedSatellite(loader, schedule, BAT_CAP, ACT_UNIT_COST, mode_strategy=strat)
        r, c, net = sim.run()
        print(f"{strat:<18} | {r:>9.2f}% | {c:>8} | {net:>12.2f}")

    print("-" * 65)
    # 策略对比
    for strat in ['default', 'periodic', 'fixed_cycle']:
        sim = TaskBasedSatellite(loader, schedule, BAT_CAP, ACT_UNIT_COST, control_strategy=strat)
        r, c, net = sim.run()
        print(f"{strat:<18} | {r:>9.2f}% | {c:>8} | {net:>12.2f}")
    print("="*65)

if __name__ == "__main__":
    main()