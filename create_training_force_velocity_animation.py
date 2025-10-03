#!/usr/bin/env python3
"""
为JAX训练数据创建Force Component和Velocity Magnitude动画
特点：训练数据中force是常量（Kolmogorov强迫）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import os
import glob
from typing import List, Tuple
import argparse

class TrainingDataAnimator:
    def __init__(self, data_dir: str):
        """
        初始化训练数据动画创建器
        
        Args:
            data_dir: 训练数据目录路径
        """
        self.data_dir = data_dir
        self.data_files = []
        self.all_frames = []
        self.frame_metadata = []
        
        # 动画参数
        self.frame_skip = 10  # 每隔几帧显示一次（在每个文件内部应用）- 减少跳跃
        
    def load_data_files(self):
        """加载所有训练数据文件"""
        print("📁 扫描训练数据文件...")
        
        # 查找训练数据文件
        pattern = os.path.join(self.data_dir, "*.npz")
        all_files = glob.glob(pattern)
        
        # 排除warmup初始文件（数据结构不同）
        self.data_files = [f for f in all_files if 'warmup_initial_velocity' not in f]
        
        # 排序文件（按step数字顺序）
        def extract_step_number(filename):
            """从文件名提取step后面的数字"""
            import re
            basename = os.path.basename(filename)
            # 查找step_后面的数字
            match = re.search(r'step_(\d+)', basename)
            return int(match.group(1)) if match else 0
        
        self.data_files.sort(key=extract_step_number)
        
        excluded_count = len(all_files) - len(self.data_files)
        print(f"✅ 找到文件:")
        print(f"  训练数据文件: {len(self.data_files)} 个")
        if excluded_count > 0:
            print(f"  排除文件: {excluded_count} 个 (warmup_initial_velocity)")
        
        # 显示前几个文件的排序验证
        print(f"📋 文件排序验证 (前5个):")
        for i, f in enumerate(self.data_files[:5]):
            step_num = extract_step_number(f)
            print(f"  {i+1}. step_{step_num} - {os.path.basename(f)}")
        
        if not self.data_files:
            raise ValueError("未找到训练数据文件!")
    
    def build_frame_index(self):
        """构建所有帧的索引"""
        print("📊 构建帧索引...")
        
        for file_idx, file_path in enumerate(self.data_files):
            try:
                data = np.load(file_path)
                
                if 'u' not in data or 'v' not in data:
                    print(f"⚠️ 跳过文件 {os.path.basename(file_path)}: 缺少velocity数据")
                    continue
                
                # 检查force数据
                has_force = 'fu' in data and 'fv' in data
                if not has_force:
                    print(f"⚠️ 文件 {os.path.basename(file_path)}: 缺少force数据")
                
                u_data = data['u']  # (time, height, width)
                v_data = data['v']
                num_frames = u_data.shape[0]
                
                # 应用frame_skip采样，确保不超出范围
                frame_indices = range(0, num_frames, self.frame_skip)
                
                for frame_idx in frame_indices:
                    # 确保索引不超出范围
                    if frame_idx >= num_frames:
                        break
                        
                    frame_info = {
                        'file_path': file_path,
                        'file_idx': file_idx,
                        'frame_idx': frame_idx,
                        'has_force': has_force,
                        'timestep': data.get('timestep', 0),
                        'num_frames': num_frames  # 添加帧数信息用于调试
                    }
                    self.all_frames.append(frame_info)
                
                print(f"  训练 {file_idx+1}/{len(self.data_files)}: {os.path.basename(file_path)} - {len(frame_indices)} 帧 (采样后)")
                
            except Exception as e:
                print(f"❌ 加载文件失败 {file_path}: {e}")
                continue
        
        print(f"✅ 总共 {len(self.all_frames)} 个动画帧")
    
    def calculate_global_ranges(self):
        """计算全局数据范围以保持colormap一致"""
        print("📊 计算全局数据范围...")
        
        # 采样一些帧来估计范围
        sample_size = min(20, len(self.all_frames))
        sample_indices = np.linspace(0, len(self.all_frames)-1, sample_size, dtype=int)
        
        u_min, u_max = float('inf'), float('-inf')
        v_min, v_max = float('inf'), float('-inf')
        vel_mag_max = 0
        fu_min, fu_max = float('inf'), float('-inf')
        fv_min, fv_max = float('inf'), float('-inf')
        force_mag_max = 0
        
        print(f"  采样 {sample_size} 个帧来估计数据范围...")
        
        for i, idx in enumerate(sample_indices):
            frame_info = self.all_frames[idx]
            data = np.load(frame_info['file_path'])
            
            # 检查帧索引是否有效
            num_frames = data['u'].shape[0]
            frame_idx = frame_info['frame_idx']
            
            if frame_idx >= num_frames:
                print(f"⚠️ 跳过无效帧: 文件 {os.path.basename(frame_info['file_path'])} 帧索引 {frame_idx} >= {num_frames}")
                continue
            
            frame_u = data['u'][frame_idx]
            frame_v = data['v'][frame_idx]
            
            u_min = min(u_min, np.min(frame_u))
            u_max = max(u_max, np.max(frame_u))
            v_min = min(v_min, np.min(frame_v))
            v_max = max(v_max, np.max(frame_v))
            vel_mag_max = max(vel_mag_max, np.max(np.sqrt(frame_u**2 + frame_v**2)))
            
            if frame_info['has_force']:
                if 'fu' not in data or 'fv' not in data:
                    print(f"⚠️ 文件缺少force数据: {os.path.basename(frame_info['file_path'])}")
                    continue
                
                # 检查force数据的维度
                if len(data['fu'].shape) == 2:
                    # 2D force数据（常量）
                    frame_fu = data['fu']
                    frame_fv = data['fv']
                elif len(data['fu'].shape) == 3:
                    # 3D force数据（时间变化）
                    force_num_frames = data['fu'].shape[0]
                    if frame_idx >= force_num_frames:
                        print(f"⚠️ Force帧索引超出范围: {frame_idx} >= {force_num_frames}")
                        continue
                    frame_fu = data['fu'][frame_idx]
                    frame_fv = data['fv'][frame_idx]
                else:
                    print(f"⚠️ 不支持的force数据维度: {data['fu'].shape}")
                    continue
                
                fu_min = min(fu_min, np.min(frame_fu))
                fu_max = max(fu_max, np.max(frame_fu))
                fv_min = min(fv_min, np.min(frame_fv))
                fv_max = max(fv_max, np.max(frame_fv))
                force_mag_max = max(force_mag_max, np.max(np.sqrt(frame_fu**2 + frame_fv**2)))
            
            print(f"\r  进度: {i+1}/{sample_size}", end="")
        
        print()
        
        # 存储范围
        self.vel_ranges = {
            'u': (u_min, u_max),
            'v': (v_min, v_max),
            'mag': (0, vel_mag_max)
        }
        
        self.force_ranges = {
            'fu': (fu_min, fu_max) if fu_min != float('inf') else (0, 1),
            'fv': (fv_min, fv_max) if fv_min != float('inf') else (0, 1),
            'mag': (0, force_mag_max) if force_mag_max > 0 else (0, 1)
        }
        
        print(f"  Velocity范围: u={self.vel_ranges['u']}, v={self.vel_ranges['v']}, mag={self.vel_ranges['mag']}")
        print(f"  Force范围: fu={self.force_ranges['fu']}, fv={self.force_ranges['fv']}, mag={self.force_ranges['mag']}")
    
    def setup_animation(self):
        """设置动画布局"""
        print("🎬 设置动画布局...")
        
        # 创建8面板布局 (2行4列)
        self.fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 上排：Force数据
        self.ax_fu = self.fig.add_subplot(gs[0, 0])
        self.ax_fv = self.fig.add_subplot(gs[0, 1])
        self.ax_force_mag = self.fig.add_subplot(gs[0, 2])
        self.ax_stats = self.fig.add_subplot(gs[0, 3])
        
        # 下排：Velocity数据
        self.ax_u = self.fig.add_subplot(gs[1, 0])
        self.ax_v = self.fig.add_subplot(gs[1, 1])
        self.ax_vel_mag = self.fig.add_subplot(gs[1, 2])
        self.ax_phase = self.fig.add_subplot(gs[1, 3])
        
        # 设置标题
        self.ax_fu.set_title('Force X (fu)', fontweight='bold')
        self.ax_fv.set_title('Force Y (fv)', fontweight='bold')
        self.ax_force_mag.set_title('Force Magnitude |f|', fontweight='bold')
        self.ax_stats.set_title('Statistics & Info', fontweight='bold')
        
        self.ax_u.set_title('Velocity X (u)', fontweight='bold')
        self.ax_v.set_title('Velocity Y (v)', fontweight='bold')
        self.ax_vel_mag.set_title('Velocity Magnitude |v|', fontweight='bold')
        self.ax_phase.set_title('Phase Portrait', fontweight='bold')
        
        # 移除统计轴的刻度
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        self.ax_phase.set_xticks([])
        self.ax_phase.set_yticks([])
    
    def animate(self, frame_num):
        """动画帧更新函数"""
        frame_info = self.all_frames[frame_num]
        data = np.load(frame_info['file_path'])
        
        # 检查帧索引是否有效
        num_frames = data['u'].shape[0]
        frame_idx = frame_info['frame_idx']
        
        if frame_idx >= num_frames:
            print(f"⚠️ 动画帧索引超出范围: {frame_idx} >= {num_frames}")
            # 使用最后一帧
            frame_idx = num_frames - 1
        
        # 获取当前帧数据
        u = data['u'][frame_idx]
        v = data['v'][frame_idx]
        vel_mag = np.sqrt(u**2 + v**2)
        
        # 清理所有轴
        for ax in [self.ax_fu, self.ax_fv, self.ax_force_mag, self.ax_stats, 
                   self.ax_u, self.ax_v, self.ax_vel_mag, self.ax_phase]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 重设标题
        self.ax_fu.set_title('Force X (fu) - CONSTANT', fontweight='bold', color='red')
        self.ax_fv.set_title('Force Y (fv) - CONSTANT', fontweight='bold', color='red')
        self.ax_force_mag.set_title('Force Magnitude |f| - CONSTANT', fontweight='bold', color='red')
        self.ax_stats.set_title('Statistics & Info', fontweight='bold')
        
        self.ax_u.set_title('Velocity X (u)', fontweight='bold', color='blue')
        self.ax_v.set_title('Velocity Y (v)', fontweight='bold', color='blue')
        self.ax_vel_mag.set_title('Velocity Magnitude |v|', fontweight='bold', color='blue')
        self.ax_phase.set_title('Phase Portrait u vs v', fontweight='bold')
        
        # 绘制velocity数据
        im_u = self.ax_u.imshow(u, cmap='RdBu_r', vmin=self.vel_ranges['u'][0], vmax=self.vel_ranges['u'][1])
        im_v = self.ax_v.imshow(v, cmap='RdBu_r', vmin=self.vel_ranges['v'][0], vmax=self.vel_ranges['v'][1])
        im_vel_mag = self.ax_vel_mag.imshow(vel_mag, cmap='viridis', vmin=self.vel_ranges['mag'][0], vmax=self.vel_ranges['mag'][1])
        
        # 绘制force数据（如果存在）
        if frame_info['has_force'] and 'fu' in data and 'fv' in data:
            # 检查force数据的维度
            if len(data['fu'].shape) == 2:
                # 2D force数据（常量）
                fu = data['fu']
                fv = data['fv']
            elif len(data['fu'].shape) == 3:
                # 3D force数据（时间变化）
                fu = data['fu'][frame_idx]
                fv = data['fv'][frame_idx]
            else:
                print(f"⚠️ 不支持的force数据维度: {data['fu'].shape}")
                fu = None
                fv = None
            
            if fu is not None and fv is not None:
                force_mag = np.sqrt(fu**2 + fv**2)
                im_fu = self.ax_fu.imshow(fu, cmap='RdBu_r', vmin=self.force_ranges['fu'][0], vmax=self.force_ranges['fu'][1])
                im_fv = self.ax_fv.imshow(fv, cmap='RdBu_r', vmin=self.force_ranges['fv'][0], vmax=self.force_ranges['fv'][1])
                im_force_mag = self.ax_force_mag.imshow(force_mag, cmap='viridis', vmin=self.force_ranges['mag'][0], vmax=self.force_ranges['mag'][1])
            else:
                # 显示"Force Error"
                self.ax_fu.text(0.5, 0.5, 'Force Error', ha='center', va='center', transform=self.ax_fu.transAxes, fontsize=14)
                self.ax_fv.text(0.5, 0.5, 'Force Error', ha='center', va='center', transform=self.ax_fv.transAxes, fontsize=14)
                self.ax_force_mag.text(0.5, 0.5, 'Force Error', ha='center', va='center', transform=self.ax_force_mag.transAxes, fontsize=14)
        else:
            # 显示"No Force Data"
            self.ax_fu.text(0.5, 0.5, 'No Force Data', ha='center', va='center', transform=self.ax_fu.transAxes, fontsize=14)
            self.ax_fv.text(0.5, 0.5, 'No Force Data', ha='center', va='center', transform=self.ax_fv.transAxes, fontsize=14)
            self.ax_force_mag.text(0.5, 0.5, 'No Force Data', ha='center', va='center', transform=self.ax_force_mag.transAxes, fontsize=14)
        
        # 绘制相位图（速度场的空间采样）
        sample_step = max(1, u.shape[0] // 20)  # 采样20x20点
        u_sample = u[::sample_step, ::sample_step].flatten()
        v_sample = v[::sample_step, ::sample_step].flatten()
        self.ax_phase.scatter(u_sample, v_sample, alpha=0.6, s=10, c='blue')
        self.ax_phase.set_xlabel('u velocity')
        self.ax_phase.set_ylabel('v velocity')
        self.ax_phase.grid(True, alpha=0.3)
        
        # 统计信息
        stats_text = f"""File: {os.path.basename(frame_info['file_path'])}
Frame: {frame_idx} (原索引: {frame_info['frame_idx']})
Timestep: {frame_info['timestep']}

Velocity Stats:
  u: [{np.min(u):.3f}, {np.max(u):.3f}]
  v: [{np.min(v):.3f}, {np.max(v):.3f}]
  |v|: [{np.min(vel_mag):.3f}, {np.max(vel_mag):.3f}]
  RMS: {np.sqrt(np.mean(vel_mag**2)):.3f}"""
        
        if frame_info['has_force'] and 'fu' in data and 'fv' in data and fu is not None and fv is not None:
            stats_text += f"""

Force Stats (CONSTANT):
  fu: [{np.min(fu):.3f}, {np.max(fu):.3f}]
  fv: [{np.min(fv):.3f}, {np.max(fv):.3f}]
  |f|: [{np.min(force_mag):.3f}, {np.max(force_mag):.3f}]
  Force RMS: {np.sqrt(np.mean(force_mag**2)):.3f}
  Force变化: ✗ 常量 (Kolmogorov)"""
        else:
            stats_text += "\n\nForce Stats:\n  No force data"
        
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontfamily='monospace', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.2))
        
        # 更新main title
        self.fig.suptitle(f'Training Data Animation - JAX-CFD - Frame {frame_num+1}/{len(self.all_frames)}', 
                         fontsize=16, fontweight='bold', color='darkblue')
        
        return []
    
    def create_animation(self, output_path: str, fps: int = 10):
        """创建并保存动画"""
        print(f"🎬 创建训练数据动画: {len(self.all_frames)} 帧，{fps} FPS")
        
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, self.animate, frames=len(self.all_frames),
            interval=1000//fps, blit=False, repeat=False
        )
        
        # 保存动画
        print(f"💾 保存动画到: {output_path}")
        
        # 使用更快的编码参数
        writer = animation.FFMpegWriter(
            fps=fps, 
            metadata=dict(artist='Training Data Animator - Constant Kolmogorov Forces'),
            bitrate=3000,
            extra_args=['-pix_fmt', 'yuv420p']
        )
        
        anim.save(output_path, writer=writer, progress_callback=lambda i, n: print(f"\r进度: {i+1}/{n} ({(i+1)/n*100:.1f}%)", end=""))
        print(f"\n✅ 动画保存完成!")
        
        return anim

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建训练数据Force+Velocity动画')
    parser.add_argument('--data_dir', type=str, 
                       default='/Users/yiwei/Projects/Python/thesis/JAX-CFD/data/training_data/256',
                       help='训练数据目录路径')
    parser.add_argument('--output', type=str, 
                       default='/Users/yiwei/Projects/Python/thesis/JAX-CFD/training_force_velocity_animation.mp4',
                       help='输出MP4文件路径')
    parser.add_argument('--fps', type=int, default=10, help='动画帧率')
    parser.add_argument('--frame_skip', type=int, default=10, help='帧采样间隔')
    
    args = parser.parse_args()
    
    print("🎬 训练数据Force+Velocity动画生成器")
    print("🔥 特色：常量Kolmogorov强迫")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出文件: {args.output}")
    print(f"帧率: {args.fps} FPS")
    print(f"帧间隔: {args.frame_skip}")
    print()
    
    # 创建动画器
    animator = TrainingDataAnimator(args.data_dir)
    animator.frame_skip = args.frame_skip
    
    try:
        # 加载数据
        animator.load_data_files()
        animator.build_frame_index()
        
        # 设置动画
        animator.calculate_global_ranges()
        animator.setup_animation()
        
        # 创建动画
        anim = animator.create_animation(args.output, args.fps)
        
        # 清理
        plt.close('all')
        
        print(f"\n🎉 成功生成训练数据动画: {args.output}")
        print("🔥 特色：展示了JAX训练数据中的常量Kolmogorov强迫！")
        
    except Exception as e:
        print(f"\n❌ 生成动画时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
